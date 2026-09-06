#include "sha256.hpp"

#include <openssl/evp.h>

#include <array>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace express_derm {
namespace {

struct DigestContextDeleter {
  void operator()(EVP_MD_CTX* context) const noexcept {
    EVP_MD_CTX_free(context);
  }
};

void require_openssl(int result, const char* operation) {
  if (result != 1) {
    throw std::runtime_error(std::string("OpenSSL failure during ") + operation);
  }
}

}  // namespace

std::string sha256_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Unable to open file for SHA-256: " + path.string());
  }

  std::unique_ptr<EVP_MD_CTX, DigestContextDeleter> context(EVP_MD_CTX_new());
  if (!context) {
    throw std::runtime_error("Unable to allocate SHA-256 context");
  }
  require_openssl(EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr),
                  "SHA-256 initialization");

  std::array<char, 1024 * 1024> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count > 0) {
      require_openssl(
          EVP_DigestUpdate(context.get(), buffer.data(), static_cast<size_t>(count)),
          "SHA-256 update");
    }
  }
  if (!input.eof()) {
    throw std::runtime_error("Unable to read file for SHA-256: " + path.string());
  }

  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int digest_size = 0;
  require_openssl(EVP_DigestFinal_ex(context.get(), digest.data(), &digest_size),
                  "SHA-256 finalization");

  std::ostringstream rendered;
  rendered << std::hex << std::setfill('0');
  for (unsigned int index = 0; index < digest_size; ++index) {
    rendered << std::setw(2) << static_cast<unsigned int>(digest[index]);
  }
  return rendered.str();
}

}  // namespace express_derm
