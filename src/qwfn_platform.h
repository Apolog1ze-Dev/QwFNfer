#pragma once
// The few platform facts every module shares. Keep this minimal: one signed
// read-size type on Windows (where ssize_t does not exist) and nothing else.
#include <cstdint>

#ifdef _WIN32
#ifndef _SSIZE_T_DEFINED
#define _SSIZE_T_DEFINED
typedef long long ssize_t;
#endif
#endif
