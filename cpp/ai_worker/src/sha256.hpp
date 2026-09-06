#pragma once

#include <filesystem>
#include <string>

namespace express_derm {

std::string sha256_file(const std::filesystem::path& path);

}  // namespace express_derm
