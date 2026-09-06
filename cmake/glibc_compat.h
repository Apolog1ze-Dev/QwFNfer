/* Portable build: keep the glibc floor at 2.34. glibc 2.38+ redirects strtol/strtoll/
   strtoull/sscanf to __isoc23_* whenever _GNU_SOURCE is set (g++ always sets it); pinning
   the C23 switch off keeps the classic symbols. sqrtf's 2.43 version is avoided with
   -fno-math-errno (the call only exists to set errno on negative input). */
#include <features.h>
#undef __GLIBC_USE_C23_STRTOL
#define __GLIBC_USE_C23_STRTOL 0
