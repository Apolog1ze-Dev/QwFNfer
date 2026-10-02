#pragma once
// The llama.cpp backend directory this project's tools default to. POSIX
// tools read $HOME; on Windows the same tree lives under %USERPROFILE%
// (set by the system), with $HOME honored when present (git-bash, msys).
#include <cstdlib>
#include <string>

namespace qwfn {

inline const char * home_env() {
#ifdef _WIN32
    const char * home = std::getenv("USERPROFILE");
    if (!home) home = std::getenv("HOME");
    return home;
#else
    return std::getenv("HOME");
#endif
}

// The default llama.cpp build/bin directory (ggml backends live there).
inline std::string llama_backend_dir() {
    const char * home = home_env();
    return std::string(home ? home : "") + "/.unsloth/llama.cpp/build/bin";
}

} // namespace qwfn
