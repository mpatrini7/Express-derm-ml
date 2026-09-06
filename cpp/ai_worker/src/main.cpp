#include "sha256.hpp"

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

static_assert(NV_TENSORRT_MAJOR >= 10,
              "Express-Derm requires the TensorRT 10 name-based C++ API");

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace express_derm {
namespace {

constexpr int kIpcVersion = 1;
constexpr size_t kMaximumMessageBytes = 1024 * 1024;
constexpr std::string_view kWorkerVersion =
    "express-derm-tensorrt-worker/0.2.0";
constexpr std::string_view kDecisionPolicyVersion = "binary-extremes-v1";

volatile std::sig_atomic_t stop_requested = 0;

void signal_handler(int) { stop_requested = 1; }

void install_signal_handlers() {
  struct sigaction stop_action {};
  stop_action.sa_handler = signal_handler;
  sigemptyset(&stop_action.sa_mask);
  stop_action.sa_flags = 0;
  if (sigaction(SIGINT, &stop_action, nullptr) != 0 ||
      sigaction(SIGTERM, &stop_action, nullptr) != 0) {
    throw std::runtime_error("Unable to install worker stop handlers");
  }

  struct sigaction pipe_action {};
  pipe_action.sa_handler = SIG_IGN;
  sigemptyset(&pipe_action.sa_mask);
  pipe_action.sa_flags = 0;
  if (sigaction(SIGPIPE, &pipe_action, nullptr) != 0) {
    throw std::runtime_error("Unable to ignore SIGPIPE");
  }
}

class WorkerError : public std::runtime_error {
 public:
  WorkerError(std::string code, std::string message, bool retryable)
      : std::runtime_error(std::move(message)),
        code_(std::move(code)),
        retryable_(retryable) {}

  const std::string& code() const noexcept { return code_; }
  bool retryable() const noexcept { return retryable_; }

 private:
  std::string code_;
  bool retryable_;
};

template <typename T>
T required(const json& source, const char* key) {
  const auto found = source.find(key);
  if (found == source.end() || found->is_null()) {
    throw std::runtime_error(std::string("Missing manifest field: ") + key);
  }
  try {
    return found->get<T>();
  } catch (const json::exception&) {
    throw std::runtime_error(std::string("Invalid manifest field: ") + key);
  }
}

bool valid_sha256(const std::string& value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

void require_finite(double value, const char* field) {
  if (!std::isfinite(value)) {
    throw std::runtime_error(std::string("Manifest field is not finite: ") +
                             field);
  }
}

struct Manifest {
  int schema_version = 1;
  std::string version;
  std::string release_version;
  std::string model_sha256;
  std::string engine_sha256;
  std::string deployment_status;
  std::string validation_status;
  std::string domain_status;
  bool thresholds_validated = false;
  std::string input_name;
  std::string output_name;
  std::string engine_filename;
  int image_size = 0;
  std::string resize_geometry = "stretch_square";
  std::array<double, 3> mean{};
  std::array<double, 3> standard_deviation{};
  std::string calibration_method;
  double logit_scale = 1.0;
  double logit_bias = 0.0;
  double temperature = 1.0;
  double low_threshold = 0.0;
  double high_threshold = 1.0;
  double abstention_margin = 0.0;

  static Manifest load(const fs::path& model_directory) {
    const fs::path manifest_path = model_directory / "manifest.json";
    std::ifstream stream(manifest_path);
    if (!stream) {
      throw std::runtime_error("Unable to open manifest: " +
                               manifest_path.string());
    }

    json document;
    try {
      stream >> document;
    } catch (const json::exception& error) {
      throw std::runtime_error("Unable to parse manifest: " +
                               std::string(error.what()));
    }
    if (!document.is_object()) {
      throw std::runtime_error("Manifest must be a JSON object");
    }

    Manifest result;
    result.schema_version = document.value("schema_version", 1);
    result.version = required<std::string>(document, "version");
    result.release_version =
        document.value("release_version", result.version);
    result.model_sha256 = required<std::string>(document, "model_sha256");
    result.engine_sha256 = required<std::string>(document, "engine_sha256");
    result.deployment_status =
        required<std::string>(document, "deployment_status");
    result.validation_status =
        required<std::string>(document, "validation_status");
    result.domain_status = required<std::string>(document, "domain_status");
    result.thresholds_validated =
        required<bool>(document, "thresholds_validated");
    result.input_name = required<std::string>(document, "input_name");
    result.output_name = required<std::string>(document, "output_name");
    result.engine_filename =
        document.value("engine_filename", std::string("model.engine"));
    result.image_size = required<int>(document, "image_size");
    result.mean = required<std::array<double, 3>>(document, "mean");
    result.standard_deviation =
        required<std::array<double, 3>>(document, "std");
    if (result.schema_version >= 3) {
      const std::string model_family =
          required<std::string>(document, "model_family");
      const std::string runtime_identity =
          required<std::string>(document, "runtime_identity");
      const int runtime_model_count =
          required<int>(document, "runtime_model_count");
      const json external_runtime_models =
          required<json>(document, "external_runtime_models");
      if (model_family != "express-derm" ||
          runtime_identity != "express-derm" ||
          runtime_model_count != 1 ||
          !external_runtime_models.is_array() ||
          !external_runtime_models.empty()) {
        throw std::runtime_error("Unsupported model runtime identity contract");
      }
      const json preprocessing = required<json>(document, "preprocessing");
      const std::map<std::string, std::string> expected_common_preprocessing = {
          {"decoder", "opencv_imread_color"},
          {"color_conversion", "bgr_to_rgb"},
          {"resize_interpolation", "area"},
          {"pixel_scale", "uint8_div_255"},
          {"layout", "nchw"},
          {"dtype", "float32"},
      };
      if (!preprocessing.is_object() ||
          preprocessing.size() != expected_common_preprocessing.size() + 2) {
        throw std::runtime_error("Unsupported preprocessing contract");
      }
      for (const auto& [field, expected] : expected_common_preprocessing) {
        if (!preprocessing.contains(field) ||
            !preprocessing.at(field).is_string() ||
            preprocessing.at(field).get<std::string>() != expected) {
          throw std::runtime_error("Unsupported preprocessing contract field: " +
                                   field);
        }
      }
      const std::string preprocessing_version =
          required<std::string>(preprocessing, "version");
      result.resize_geometry =
          required<std::string>(preprocessing, "resize_geometry");
      const bool stretch_contract =
          preprocessing_version ==
              "opencv_imread_bgr2rgb_inter_area_stretch_imagenet_nchw_float32_v1" &&
          result.resize_geometry == "stretch_square";
      const bool letterbox_contract =
          preprocessing_version ==
              "opencv_imread_bgr2rgb_inter_area_letterbox_imagenetmean_nchw_float32_v1" &&
          result.resize_geometry == "letterbox_square_imagenet_mean";
      if (!stretch_contract && !letterbox_contract) {
        throw std::runtime_error(
            "Unsupported preprocessing version/geometry combination");
      }
    }
    result.calibration_method =
        required<std::string>(document, "calibration_method");
    result.low_threshold = required<double>(document, "low_threshold");
    result.high_threshold = required<double>(document, "high_threshold");
    result.abstention_margin =
        required<double>(document, "abstention_margin");

    if (result.calibration_method == "affine_logistic_scaling") {
      result.logit_scale = required<double>(document, "logit_scale");
      result.logit_bias = required<double>(document, "logit_bias");
      if (result.logit_scale <= 0.0) {
        throw std::runtime_error("Manifest logit_scale must be positive");
      }
    } else if (result.calibration_method == "temperature_scaling") {
      result.temperature = required<double>(document, "temperature");
      if (result.temperature <= 0.0) {
        throw std::runtime_error("Manifest temperature must be positive");
      }
    } else {
      throw std::runtime_error("Unsupported calibration method: " +
                               result.calibration_method);
    }

    if (result.version.empty() || result.release_version.empty() ||
        result.input_name.empty() ||
        result.output_name.empty() || result.engine_filename.empty()) {
      throw std::runtime_error("Manifest identity and tensor names cannot be blank");
    }
    const fs::path engine_filename(result.engine_filename);
    if (engine_filename.is_absolute() || engine_filename.has_parent_path() ||
        engine_filename.filename() != engine_filename) {
      throw std::runtime_error("Manifest engine_filename must be one filename");
    }
    if (!valid_sha256(result.model_sha256) ||
        !valid_sha256(result.engine_sha256)) {
      throw std::runtime_error("Manifest contains an invalid SHA-256 value");
    }
    if (result.image_size <= 0 || result.image_size > 4096) {
      throw std::runtime_error("Manifest image_size is outside the supported range");
    }
    for (size_t channel = 0; channel < result.mean.size(); ++channel) {
      require_finite(result.mean[channel], "mean");
      require_finite(result.standard_deviation[channel], "std");
      if (result.standard_deviation[channel] <= 0.0) {
        throw std::runtime_error("Manifest standard deviations must be positive");
      }
    }
    require_finite(result.logit_scale, "logit_scale");
    require_finite(result.logit_bias, "logit_bias");
    require_finite(result.temperature, "temperature");
    require_finite(result.low_threshold, "low_threshold");
    require_finite(result.high_threshold, "high_threshold");
    require_finite(result.abstention_margin, "abstention_margin");
    if (!(0.0 <= result.low_threshold &&
          result.low_threshold < result.high_threshold &&
          result.high_threshold <= 1.0)) {
      throw std::runtime_error("Manifest thresholds are not ordered");
    }
    if (result.abstention_margin < 0.0 || result.abstention_margin > 0.5) {
      throw std::runtime_error("Manifest abstention_margin is invalid");
    }
    return result;
  }

  json identity() const {
    return {
        {"model_version", version},
        {"model_release", release_version},
        {"model_sha256", model_sha256},
        {"engine_sha256", engine_sha256},
        {"deployment_status", deployment_status},
        {"validation_status", validation_status},
        {"domain_status", domain_status},
        {"thresholds_validated", thresholds_validated},
    };
  }
};

class TensorRTLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "TensorRT: " << message << '\n';
    }
  }
};

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA failure during ") + operation +
                             ": " + cudaGetErrorString(status));
  }
}

void set_close_on_exec(int descriptor) {
  const int current = fcntl(descriptor, F_GETFD);
  if (current < 0 || fcntl(descriptor, F_SETFD, current | FD_CLOEXEC) < 0) {
    throw std::runtime_error("Unable to set close-on-exec on a local socket");
  }
}

std::vector<char> read_binary_file(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error("Unable to open TensorRT engine: " + path.string());
  }
  const std::streamsize size = stream.tellg();
  if (size <= 0) {
    throw std::runtime_error("TensorRT engine is empty");
  }
  stream.seekg(0, std::ios::beg);
  std::vector<char> data(static_cast<size_t>(size));
  if (!stream.read(data.data(), size)) {
    throw std::runtime_error("Unable to read TensorRT engine");
  }
  return data;
}

size_t tensor_volume(const nvinfer1::Dims& dimensions) {
  if (dimensions.nbDims <= 0) {
    throw std::runtime_error("TensorRT tensor has no dimensions");
  }
  size_t result = 1;
  for (int index = 0; index < dimensions.nbDims; ++index) {
    const auto dimension = dimensions.d[index];
    if (dimension <= 0) {
      throw std::runtime_error("TensorRT tensor has an unresolved dimension");
    }
    const auto converted = static_cast<size_t>(dimension);
    if (result > std::numeric_limits<size_t>::max() / converted) {
      throw std::runtime_error("TensorRT tensor size overflows host memory");
    }
    result *= converted;
  }
  return result;
}

bool has_dynamic_dimension(const nvinfer1::Dims& dimensions) {
  for (int index = 0; index < dimensions.nbDims; ++index) {
    if (dimensions.d[index] < 0) {
      return true;
    }
  }
  return false;
}

struct InferenceOutput {
  float raw_output = 0.0F;
  double preprocessing_ms = 0.0;
  double inference_ms = 0.0;
};

class TensorRTEngine {
 public:
  TensorRTEngine(const fs::path& engine_path, const Manifest& manifest)
      : manifest_(manifest) {
    check_cuda(cudaFree(nullptr), "CUDA context initialization");
    const std::vector<char> engine_bytes = read_binary_file(engine_path);
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
      throw std::runtime_error("Unable to create the TensorRT runtime");
    }
    engine_.reset(
        runtime_->deserializeCudaEngine(engine_bytes.data(), engine_bytes.size()));
    if (!engine_) {
      throw std::runtime_error("Unable to deserialize the trusted TensorRT engine");
    }
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
      throw std::runtime_error("Unable to create the TensorRT execution context");
    }

    validate_tensor(manifest_.input_name, nvinfer1::TensorIOMode::kINPUT);
    validate_tensor(manifest_.output_name, nvinfer1::TensorIOMode::kOUTPUT);
    if (engine_->getNbIOTensors() != 2) {
      throw std::runtime_error("The worker requires exactly one input and one output");
    }

    nvinfer1::Dims input_dimensions =
        engine_->getTensorShape(manifest_.input_name.c_str());
    if (has_dynamic_dimension(input_dimensions)) {
      const nvinfer1::Dims4 desired{1, 3, manifest_.image_size,
                                    manifest_.image_size};
      if (!context_->setInputShape(manifest_.input_name.c_str(), desired)) {
        throw std::runtime_error("TensorRT rejected the configured input shape");
      }
    }
    input_dimensions =
        context_->getTensorShape(manifest_.input_name.c_str());
    const size_t expected_elements =
        static_cast<size_t>(3) * static_cast<size_t>(manifest_.image_size) *
        static_cast<size_t>(manifest_.image_size);
    input_elements_ = tensor_volume(input_dimensions);
    if (input_elements_ != expected_elements) {
      throw std::runtime_error("TensorRT input shape does not match the manifest");
    }

    const nvinfer1::Dims output_dimensions =
        context_->getTensorShape(manifest_.output_name.c_str());
    output_elements_ = tensor_volume(output_dimensions);
    if (output_elements_ != 1) {
      throw std::runtime_error("The TensorRT model must return one scalar logit");
    }

    check_cuda(cudaStreamCreate(&stream_), "stream creation");
    check_cuda(cudaMalloc(&device_input_, input_elements_ * sizeof(float)),
               "input allocation");
    check_cuda(cudaMalloc(&device_output_, output_elements_ * sizeof(float)),
               "output allocation");
    if (!context_->setTensorAddress(manifest_.input_name.c_str(), device_input_) ||
        !context_->setTensorAddress(manifest_.output_name.c_str(), device_output_)) {
      throw std::runtime_error("Unable to bind TensorRT tensor addresses");
    }
  }

  TensorRTEngine(const TensorRTEngine&) = delete;
  TensorRTEngine& operator=(const TensorRTEngine&) = delete;

  ~TensorRTEngine() {
    if (stream_ != nullptr) {
      cudaStreamSynchronize(stream_);
    }
    if (device_output_ != nullptr) {
      cudaFree(device_output_);
    }
    if (device_input_ != nullptr) {
      cudaFree(device_input_);
    }
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
    context_.reset();
    engine_.reset();
    runtime_.reset();
  }

  InferenceOutput predict(const fs::path& image_path) {
    const auto preprocessing_started = std::chrono::steady_clock::now();
    std::vector<float> input = preprocess(image_path);
    const auto preprocessing_finished = std::chrono::steady_clock::now();

    float output = 0.0F;
    const auto inference_started = preprocessing_finished;
    check_cuda(cudaMemcpyAsync(device_input_, input.data(),
                               input.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream_),
               "input transfer");
    if (!context_->enqueueV3(stream_)) {
      throw std::runtime_error("TensorRT enqueueV3 returned false");
    }
    check_cuda(cudaMemcpyAsync(&output, device_output_, sizeof(float),
                               cudaMemcpyDeviceToHost, stream_),
               "output transfer");
    check_cuda(cudaStreamSynchronize(stream_), "inference synchronization");
    const auto inference_finished = std::chrono::steady_clock::now();
    if (!std::isfinite(output)) {
      throw std::runtime_error("TensorRT returned a non-finite scalar output");
    }

    return {
        output,
        std::chrono::duration<double, std::milli>(preprocessing_finished -
                                                  preprocessing_started)
            .count(),
        std::chrono::duration<double, std::milli>(inference_finished -
                                                  inference_started)
            .count(),
    };
  }

 private:
  void validate_tensor(const std::string& name,
                       nvinfer1::TensorIOMode wanted_mode) const {
    if (engine_->getTensorIOMode(name.c_str()) != wanted_mode) {
      throw std::runtime_error("TensorRT tensor is missing or has the wrong mode: " +
                               name);
    }
    if (engine_->getTensorDataType(name.c_str()) != nvinfer1::DataType::kFLOAT) {
      throw std::runtime_error("TensorRT worker currently requires FP32 tensor I/O: " +
                               name);
    }
    if (engine_->getTensorLocation(name.c_str()) !=
        nvinfer1::TensorLocation::kDEVICE) {
      throw std::runtime_error("TensorRT worker requires device tensor I/O: " +
                               name);
    }
  }

  std::vector<float> preprocess(const fs::path& image_path) const {
    const cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) {
      throw WorkerError("IMAGE_DECODE_FAILED",
                        "Unable to decode the microscope image", false);
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    if (manifest_.resize_geometry == "stretch_square") {
      cv::resize(rgb, resized,
                 cv::Size(manifest_.image_size, manifest_.image_size), 0.0,
                 0.0, cv::INTER_AREA);
    } else if (manifest_.resize_geometry ==
               "letterbox_square_imagenet_mean") {
      const double scale = std::min(
          static_cast<double>(manifest_.image_size) / rgb.cols,
          static_cast<double>(manifest_.image_size) / rgb.rows);
      const int target_width = std::clamp(
          static_cast<int>(std::lround(rgb.cols * scale)), 1,
          manifest_.image_size);
      const int target_height = std::clamp(
          static_cast<int>(std::lround(rgb.rows * scale)), 1,
          manifest_.image_size);
      cv::Mat content;
      cv::resize(rgb, content, cv::Size(target_width, target_height), 0.0,
                 0.0, cv::INTER_AREA);
      const cv::Scalar fill(
          std::lround(manifest_.mean[0] * 255.0),
          std::lround(manifest_.mean[1] * 255.0),
          std::lround(manifest_.mean[2] * 255.0));
      resized = cv::Mat(manifest_.image_size, manifest_.image_size, CV_8UC3,
                        fill);
      const int left = (manifest_.image_size - target_width) / 2;
      const int top = (manifest_.image_size - target_height) / 2;
      content.copyTo(
          resized(cv::Rect(left, top, target_width, target_height)));
    } else {
      throw std::runtime_error("Unsupported resize geometry");
    }

    const size_t plane = static_cast<size_t>(manifest_.image_size) *
                         static_cast<size_t>(manifest_.image_size);
    std::vector<float> tensor(plane * 3);
    for (int row = 0; row < manifest_.image_size; ++row) {
      for (int column = 0; column < manifest_.image_size; ++column) {
        const cv::Vec3b pixel = resized.at<cv::Vec3b>(row, column);
        const size_t offset = static_cast<size_t>(row) *
                                  static_cast<size_t>(manifest_.image_size) +
                              static_cast<size_t>(column);
        for (size_t channel = 0; channel < 3; ++channel) {
          const double normalized =
              (static_cast<double>(pixel[channel]) / 255.0 -
               manifest_.mean[channel]) /
              manifest_.standard_deviation[channel];
          tensor[channel * plane + offset] = static_cast<float>(normalized);
        }
      }
    }
    return tensor;
  }

  const Manifest& manifest_;
  TensorRTLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_ = nullptr;
  void* device_input_ = nullptr;
  void* device_output_ = nullptr;
  size_t input_elements_ = 0;
  size_t output_elements_ = 0;
};

struct Arguments {
  fs::path model_directory;
  fs::path image_root;
  fs::path socket_path;
};

Arguments parse_arguments(int argc, char** argv) {
  Arguments arguments;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (option == "--help") {
      std::cout << "Usage: express-derm-ai-worker --model-dir DIR "
                   "--image-root DIR --socket PATH\n";
      std::exit(0);
    }
    if (index + 1 >= argc) {
      throw std::runtime_error("Missing value for argument: " + option);
    }
    const fs::path value = argv[++index];
    if (option == "--model-dir") {
      arguments.model_directory = value;
    } else if (option == "--image-root") {
      arguments.image_root = value;
    } else if (option == "--socket") {
      arguments.socket_path = value;
    } else {
      throw std::runtime_error("Unknown argument: " + option);
    }
  }
  if (arguments.model_directory.empty() || arguments.image_root.empty() ||
      arguments.socket_path.empty()) {
    throw std::runtime_error(
        "--model-dir, --image-root and --socket are all required");
  }
  arguments.model_directory = fs::canonical(arguments.model_directory);
  arguments.image_root = fs::canonical(arguments.image_root);
  arguments.socket_path = fs::absolute(arguments.socket_path).lexically_normal();
  if (!fs::is_directory(arguments.model_directory) ||
      !fs::is_directory(arguments.image_root)) {
    throw std::runtime_error("Model directory and image root must be directories");
  }
  return arguments;
}

bool path_is_within(const fs::path& candidate, const fs::path& root) {
  const fs::path relative = fs::relative(candidate, root);
  if (relative.empty() || relative.is_absolute()) {
    return false;
  }
  for (const auto& component : relative) {
    if (component == "..") {
      return false;
    }
  }
  return true;
}

fs::path validated_image_path(const json& input, const fs::path& image_root) {
  if (!input.is_object()) {
    throw WorkerError("INVALID_REQUEST", "input must be an object", false);
  }
  const auto path_value = input.find("image_path");
  if (path_value == input.end() || !path_value->is_string()) {
    throw WorkerError("INVALID_REQUEST", "image_path must be text", false);
  }
  try {
    const fs::path requested = path_value->get<std::string>();
    if (!requested.is_absolute()) {
      throw WorkerError("IMAGE_PATH_REJECTED",
                        "Microscope image path must be absolute", false);
    }
    const fs::path resolved = fs::canonical(requested);
    if (!path_is_within(resolved, image_root) || !fs::is_regular_file(resolved)) {
      throw WorkerError("IMAGE_PATH_REJECTED",
                        "Microscope image must be a regular file below image root",
                        false);
    }
    return resolved;
  } catch (const WorkerError&) {
    throw;
  } catch (const fs::filesystem_error& error) {
    throw WorkerError("IMAGE_PATH_REJECTED",
                      std::string("Microscope image is unavailable: ") +
                          error.what(),
                      false);
  }
}

bool valid_request_id(const std::string& value) {
  return value.size() == 32 &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

bool expected_identity_matches(const json& request, const Manifest& manifest) {
  const auto found = request.find("expected_model");
  return found != request.end() && found->is_object() &&
         *found == manifest.identity();
}

json success_response(const std::string& request_id, json result) {
  return {
      {"ipc_version", kIpcVersion},
      {"request_id", request_id},
      {"ok", true},
      {"result", std::move(result)},
  };
}

json error_response(const std::string& request_id, const std::string& code,
                    const std::string& message, bool retryable) {
  return {
      {"ipc_version", kIpcVersion},
      {"request_id", request_id},
      {"ok", false},
      {"error",
       {{"code", code}, {"message", message}, {"retryable", retryable}}},
  };
}

double calibrated_score(float raw_output, const Manifest& manifest) {
  double calibrated_logit = 0.0;
  if (manifest.calibration_method == "affine_logistic_scaling") {
    calibrated_logit = manifest.logit_scale * static_cast<double>(raw_output) +
                       manifest.logit_bias;
  } else {
    calibrated_logit = static_cast<double>(raw_output) / manifest.temperature;
  }
  calibrated_logit = std::clamp(calibrated_logit, -50.0, 50.0);
  return 1.0 / (1.0 + std::exp(-calibrated_logit));
}

json prediction_result(const InferenceOutput& inference,
                       const Manifest& manifest) {
  const double score = calibrated_score(inference.raw_output, manifest);
  const bool threshold_gap =
      score >= manifest.low_threshold && score < manifest.high_threshold;
  const bool configured_margin =
      std::abs(score - 0.5) < manifest.abstention_margin;
  const bool abstained = threshold_gap || configured_margin;
  std::string attention_level;
  if (abstained) {
    attention_level = "uncertain";
  } else if (score < manifest.low_threshold) {
    attention_level = "low";
  } else {
    attention_level = "high";
  }

  json abstention_reason = nullptr;
  if (threshold_gap) {
    abstention_reason =
        "Score falls between the configured low and high decision thresholds";
  } else if (configured_margin) {
    abstention_reason =
        "Score falls inside the configured uncertainty region";
  }

  json result = manifest.identity();
  result.update({
      {"inference_backend", "tensorrt_cpp"},
      {"raw_output", inference.raw_output},
      {"score", score},
      {"attention_level", attention_level},
      {"abstained", abstained},
      {"abstention_reason", abstention_reason},
      {"decision_policy_version", std::string(kDecisionPolicyVersion)},
      {"preprocessing_ms", inference.preprocessing_ms},
      {"inference_ms", inference.inference_ms},
      {"latency_ms", inference.preprocessing_ms + inference.inference_ms},
  });
  return result;
}

json process_request(const json& request, const Manifest& manifest,
                     TensorRTEngine& engine, const fs::path& image_root) {
  if (!request.is_object()) {
    throw WorkerError("INVALID_REQUEST", "Request must be an object", false);
  }
  const auto ipc_version = request.find("ipc_version");
  if (ipc_version == request.end() || !ipc_version->is_number_integer() ||
      ipc_version->get<int>() != kIpcVersion) {
    throw WorkerError("INVALID_REQUEST", "IPC version mismatch", false);
  }
  const auto request_id_value = request.find("request_id");
  if (request_id_value == request.end() || !request_id_value->is_string()) {
    throw WorkerError("INVALID_REQUEST", "request_id is invalid", false);
  }
  const std::string request_id = request_id_value->get<std::string>();
  if (!valid_request_id(request_id)) {
    throw WorkerError("INVALID_REQUEST", "request_id is invalid", false);
  }
  const auto operation_value = request.find("operation");
  if (operation_value == request.end() || !operation_value->is_string()) {
    throw WorkerError("INVALID_REQUEST", "operation must be text", false);
  }
  const std::string operation = operation_value->get<std::string>();
  if (operation == "status") {
    if (!expected_identity_matches(request, manifest)) {
      json result = manifest.identity();
      result.update({
          {"ready", false},
          {"worker_version", std::string(kWorkerVersion)},
          {"decision_policy_version", std::string(kDecisionPolicyVersion)},
          {"reason", "Requested model identity does not match the loaded model"},
      });
      return success_response(request_id, std::move(result));
    }
    json result = manifest.identity();
    result.update({
        {"ready", true},
        {"worker_version", std::string(kWorkerVersion)},
        {"decision_policy_version", std::string(kDecisionPolicyVersion)},
        {"reason", nullptr},
    });
    return success_response(request_id, std::move(result));
  }
  if (operation != "predict") {
    throw WorkerError("INVALID_REQUEST", "Unknown operation", false);
  }
  if (!expected_identity_matches(request, manifest)) {
    throw WorkerError("MODEL_IDENTITY_MISMATCH",
                      "Registered model identity does not match the request",
                      false);
  }

  const auto input = request.find("input");
  if (input == request.end() || !input->is_object()) {
    throw WorkerError("INVALID_REQUEST", "input must be an object", false);
  }
  const auto quality = input->find("quality_status");
  const auto source_confirmed = input->find("microscope_source_confirmed");
  const auto protocol = input->find("acquisition_protocol_status");
  const auto recorded_protocol =
      input->find("recorded_acquisition_protocol_status");
  if (quality == input->end() || !quality->is_string() ||
      source_confirmed == input->end() || !source_confirmed->is_boolean() ||
      protocol == input->end() || !protocol->is_string() ||
      (recorded_protocol != input->end() &&
       !recorded_protocol->is_string())) {
    throw WorkerError("INVALID_REQUEST", "Eligibility fields have invalid types",
                      false);
  }
  const bool eligible = quality->get<std::string>() == "accepted" &&
                        source_confirmed->get<bool>();
  const auto observation_id = input->find("observation_id");
  if (observation_id == input->end() || !observation_id->is_number_integer() ||
      observation_id->get<std::int64_t>() <= 0) {
    throw WorkerError("INVALID_REQUEST", "observation_id must be positive", false);
  }
  if (!eligible) {
    throw WorkerError("INELIGIBLE_OBSERVATION",
                      "Only accepted, source-confirmed microscope images can "
                      "reach inference",
                      false);
  }

  const fs::path image_path = validated_image_path(*input, image_root);
  try {
    return success_response(request_id,
                            prediction_result(engine.predict(image_path), manifest));
  } catch (const WorkerError&) {
    throw;
  } catch (const std::exception& error) {
    throw WorkerError("INFERENCE_FAILED", error.what(), true);
  }
}

std::string request_id_for_error(const json& request) {
  if (request.is_object()) {
    const auto found = request.find("request_id");
    if (found != request.end() && found->is_string()) {
      const std::string value = found->get<std::string>();
      if (valid_request_id(value)) {
        return value;
      }
    }
  }
  return std::string(32, '0');
}

std::string read_message(int connection) {
  std::string message;
  std::array<char, 65536> buffer{};
  while (true) {
    const ssize_t count = recv(connection, buffer.data(), buffer.size(), 0);
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw WorkerError("INVALID_REQUEST", "Unable to read request", true);
    }
    if (count == 0) {
      throw WorkerError("INVALID_REQUEST", "Request ended before newline", false);
    }
    message.append(buffer.data(), static_cast<size_t>(count));
    const size_t newline = message.find('\n');
    if (newline != std::string::npos) {
      if (newline > kMaximumMessageBytes) {
        throw WorkerError("INVALID_REQUEST", "Request exceeds 1 MiB", false);
      }
      message.resize(newline);
      return message;
    }
    if (message.size() > kMaximumMessageBytes) {
      throw WorkerError("INVALID_REQUEST", "Request exceeds 1 MiB", false);
    }
  }
}

void send_message(int connection, const json& response) {
  const std::string rendered = response.dump() + "\n";
  if (rendered.size() > kMaximumMessageBytes) {
    throw std::runtime_error("Response exceeds 1 MiB");
  }
  size_t sent = 0;
  while (sent < rendered.size()) {
    const ssize_t count = send(connection, rendered.data() + sent,
                               rendered.size() - sent, MSG_NOSIGNAL);
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error("Unable to send worker response");
    }
    sent += static_cast<size_t>(count);
  }
}

class UnixServer {
 public:
  explicit UnixServer(fs::path socket_path)
      : socket_path_(std::move(socket_path)) {
    if (socket_path_.string().size() >= sizeof(sockaddr_un::sun_path)) {
      throw std::runtime_error("Unix socket path is too long");
    }
    fs::create_directories(socket_path_.parent_path());
    remove_stale_socket();

    descriptor_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (descriptor_ < 0) {
      throw std::runtime_error("Unable to create Unix socket: " +
                               std::string(std::strerror(errno)));
    }
    set_close_on_exec(descriptor_);
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::strncpy(address.sun_path, socket_path_.c_str(),
                 sizeof(address.sun_path) - 1);
    if (bind(descriptor_, reinterpret_cast<sockaddr*>(&address),
             sizeof(address)) != 0) {
      const std::string reason = std::strerror(errno);
      close(descriptor_);
      descriptor_ = -1;
      throw std::runtime_error("Unable to bind Unix socket: " + reason);
    }
    owns_socket_ = true;
    if (chmod(socket_path_.c_str(), S_IRUSR | S_IWUSR) != 0) {
      throw std::runtime_error("Unable to restrict Unix socket permissions");
    }
    if (listen(descriptor_, 16) != 0) {
      throw std::runtime_error("Unable to listen on Unix socket");
    }
  }

  UnixServer(const UnixServer&) = delete;
  UnixServer& operator=(const UnixServer&) = delete;

  ~UnixServer() {
    if (descriptor_ >= 0) {
      close(descriptor_);
    }
    if (owns_socket_) {
      unlink(socket_path_.c_str());
    }
  }

  void run(const Manifest& manifest, TensorRTEngine& engine,
           const fs::path& image_root) {
    while (!stop_requested) {
      const int connection = accept(descriptor_, nullptr, nullptr);
      if (connection < 0) {
        if (errno == EINTR) {
          continue;
        }
        throw std::runtime_error("Unable to accept worker connection: " +
                                 std::string(std::strerror(errno)));
      }
      try {
        set_close_on_exec(connection);
      } catch (const std::exception&) {
        close(connection);
        throw;
      }
      handle_connection(connection, manifest, engine, image_root);
      close(connection);
    }
  }

 private:
  void remove_stale_socket() {
    struct stat metadata {};
    if (lstat(socket_path_.c_str(), &metadata) != 0) {
      if (errno == ENOENT) {
        return;
      }
      throw std::runtime_error("Unable to inspect existing socket path");
    }
    if (!S_ISSOCK(metadata.st_mode)) {
      throw std::runtime_error(
          "Refusing to replace a non-socket path: " + socket_path_.string());
    }

    const int probe = socket(AF_UNIX, SOCK_STREAM, 0);
    if (probe < 0) {
      throw std::runtime_error("Unable to probe existing Unix socket");
    }
    set_close_on_exec(probe);
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::strncpy(address.sun_path, socket_path_.c_str(),
                 sizeof(address.sun_path) - 1);
    if (connect(probe, reinterpret_cast<sockaddr*>(&address), sizeof(address)) ==
        0) {
      close(probe);
      throw std::runtime_error("A worker is already listening on " +
                               socket_path_.string());
    }
    const int connection_error = errno;
    close(probe);
    if (connection_error != ECONNREFUSED && connection_error != ENOENT) {
      throw std::runtime_error("Existing socket cannot be safely replaced: " +
                               std::string(std::strerror(connection_error)));
    }
    if (unlink(socket_path_.c_str()) != 0 && errno != ENOENT) {
      throw std::runtime_error("Unable to remove stale Unix socket");
    }
  }

  static void handle_connection(int connection, const Manifest& manifest,
                                TensorRTEngine& engine,
                                const fs::path& image_root) {
    json request;
    try {
      const std::string incoming = read_message(connection);
      try {
        request = json::parse(incoming);
      } catch (const json::exception&) {
        throw WorkerError("INVALID_REQUEST", "Request is not valid JSON", false);
      }
      send_message(connection,
                   process_request(request, manifest, engine, image_root));
    } catch (const WorkerError& error) {
      try {
        send_message(connection,
                     error_response(request_id_for_error(request), error.code(),
                                    error.what(), error.retryable()));
      } catch (const std::exception&) {
        // The peer may already have disconnected; the worker remains available.
      }
    } catch (const std::exception& error) {
      try {
        send_message(connection,
                     error_response(request_id_for_error(request),
                                    "INFERENCE_FAILED", error.what(), true));
      } catch (const std::exception&) {
        // The peer may already have disconnected; the worker remains available.
      }
    }
  }

  fs::path socket_path_;
  int descriptor_ = -1;
  bool owns_socket_ = false;
};

void verify_model_package(const fs::path& model_directory,
                          const Manifest& manifest) {
  const fs::path model_path = model_directory / "model.onnx";
  const fs::path engine_path = model_directory / manifest.engine_filename;
  if (fs::is_symlink(model_path) || fs::is_symlink(engine_path) ||
      !fs::is_regular_file(model_path) || !fs::is_regular_file(engine_path)) {
    throw std::runtime_error(
        "Registered package must contain regular model.onnx and engine files");
  }
  if (sha256_file(model_path) != manifest.model_sha256) {
    throw std::runtime_error("ONNX SHA-256 does not match manifest");
  }
  if (sha256_file(engine_path) != manifest.engine_sha256) {
    throw std::runtime_error("TensorRT engine SHA-256 does not match manifest");
  }
}

}  // namespace
}  // namespace express_derm

int main(int argc, char** argv) {
  using namespace express_derm;
  try {
    install_signal_handlers();

    const Arguments arguments = parse_arguments(argc, argv);
    const Manifest manifest = Manifest::load(arguments.model_directory);
    verify_model_package(arguments.model_directory, manifest);
    TensorRTEngine engine(arguments.model_directory / manifest.engine_filename,
                          manifest);
    UnixServer server(arguments.socket_path);

    std::cerr << kWorkerVersion << " ready; model=" << manifest.version
              << " socket=" << arguments.socket_path << '\n';
    server.run(manifest, engine, arguments.image_root);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "express-derm-ai-worker: " << error.what() << '\n';
    return 1;
  }
}
