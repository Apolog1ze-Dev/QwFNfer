cmake_minimum_required(VERSION 3.20)
project(qwfnfer LANGUAGES C CXX)

# qwfnfer -- a purpose-built inference engine for Qwen3.8-Flash-Next (GGUF arch "qwen4exp").
#
# We own the model graph and the memory hierarchy; ggml supplies the quantized
# math kernels. The entire performance thesis of this engine lives in
# qwfn_io / qwfn_expert_cache / qwfn_ple: fetch expert slices at their natural
# 627KB-921KB granularity over io_uring instead of 4KiB demand paging.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Where llama.cpp/ggml lives. Override with -DLLAMA_CPP_ROOT=...
set(LLAMA_CPP_ROOT "$ENV{HOME}/.unsloth/llama.cpp" CACHE PATH "llama.cpp source root")
set(LLAMA_CPP_BUILD "${LLAMA_CPP_ROOT}/build/bin"   CACHE PATH "llama.cpp built libraries")
# QWFN_PORTABLE builds the release bundle (scripts/package.sh): baseline x86-64-v3 code
# instead of this machine's -march=native, a glibc 2.34 floor (cmake/glibc_compat.h keeps
# the pre-C23 strtol/sscanf symbols, -fno-math-errno the pre-2.43 sqrtf), libstdc++ and
# libgcc linked in, and the ggml/CUDA libraries found next to the binaries ($ORIGIN).
# Windows carries no portable bundle yet: the flag set is Linux-only and stays gated.
option(QWFN_PORTABLE "Relocatable release build" OFF)
if(QWFN_PORTABLE AND NOT WIN32)
  set(QWFN_ARCH_FLAGS -march=x86-64-v3 -mtune=generic)
  add_compile_options(-include "${CMAKE_SOURCE_DIR}/cmake/glibc_compat.h" -fno-math-errno)
  add_link_options(-static-libstdc++ -static-libgcc)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
  set(CMAKE_INSTALL_RPATH "$ORIGIN")
elseif(NOT WIN32)
  set(QWFN_ARCH_FLAGS -march=native)
else()
  # clang-cl / MSVC: the toolchain's own optimization flags; no -march (the
  # user's /clang:-march flags arrive through CMAKE_CXX_FLAGS when wanted).
  set(QWFN_ARCH_FLAGS "")
  # getenv/fopen POSIX usage (and the CRT's other "secure" variants of calls
  # this project deliberately keeps portable) trip MSVC's deprecation wall.
  add_compile_options(/W4 /wd4996)
endif()

if(NOT EXISTS "${LLAMA_CPP_ROOT}/ggml/include/ggml.h")
  message(FATAL_ERROR "ggml headers not found under ${LLAMA_CPP_ROOT}. Pass -DLLAMA_CPP_ROOT=<path>.")
endif()

# The imported ggml/llama libraries carry the platform's own suffix and name.
# Windows: the DLLs land in bin/, the import libraries (.lib) one directory up;
# link through the .lib, find the DLL at runtime (same dir as our binaries).
if(WIN32)
  set(GGML_BASE_LIB "${LLAMA_CPP_BUILD}/ggml-base.dll")
  set(GGML_BASE_IMPLIB "${LLAMA_CPP_BUILD}/../ggml/src/ggml-base.lib")
  set(GGML_ALL_LIB  "${LLAMA_CPP_BUILD}/ggml.dll")
  set(GGML_ALL_IMPLIB  "${LLAMA_CPP_BUILD}/../ggml/src/ggml.lib")
  set(LLAMA_LIB     "${LLAMA_CPP_BUILD}/llama.dll")
  set(LLAMA_IMPLIB     "${LLAMA_CPP_BUILD}/../src/llama.lib")
else()
  set(GGML_BASE_LIB "${LLAMA_CPP_BUILD}/libggml-base.so.0")
  set(GGML_ALL_LIB  "${LLAMA_CPP_BUILD}/libggml.so.0")
  set(LLAMA_LIB     "${LLAMA_CPP_BUILD}/libllama.so.0")
endif()
if(NOT EXISTS "${GGML_BASE_LIB}" AND NOT EXISTS "${GGML_BASE_IMPLIB}")
  message(FATAL_ERROR "built ggml not found under ${LLAMA_CPP_BUILD} (${GGML_BASE_LIB}). Build llama.cpp first or pass -DLLAMA_CPP_BUILD=<dir>.")
endif()

add_library(ggml_base SHARED IMPORTED)
if(WIN32)
  set_target_properties(ggml_base PROPERTIES IMPORTED_IMPLIB "${GGML_BASE_IMPLIB}" IMPORTED_LOCATION "${GGML_BASE_LIB}")
else()
  set_target_properties(ggml_base PROPERTIES IMPORTED_LOCATION "${GGML_BASE_LIB}")
endif()
add_library(ggml_all SHARED IMPORTED)
if(WIN32)
  set_target_properties(ggml_all PROPERTIES IMPORTED_IMPLIB "${GGML_ALL_IMPLIB}" IMPORTED_LOCATION "${GGML_ALL_LIB}")
else()
  set_target_properties(ggml_all PROPERTIES IMPORTED_LOCATION "${GGML_ALL_LIB}")
endif()

# Two things need libllama: qwfn-refdump drives llama.cpp's own graph to produce
# the reference intermediates our graph is bisected against, and qwfn-chat
# borrows its tokenizer (opened vocab_only, so no weights are loaded).
add_library(llama_ref SHARED IMPORTED)
if(WIN32)
  set_target_properties(llama_ref PROPERTIES IMPORTED_IMPLIB "${LLAMA_IMPLIB}" IMPORTED_LOCATION "${LLAMA_LIB}")
else()
  set_target_properties(llama_ref PROPERTIES IMPORTED_LOCATION "${LLAMA_LIB}")
endif()

find_package(Threads REQUIRED)

# liburing is the Linux io_uring backend. Windows has no io_uring: the io
# layer runs the thread-pool backend there (see src/qwfn_io.cpp), so the
# library is mandatory only where that backend exists.
if(NOT WIN32)
  find_path(URING_INCLUDE_DIR liburing.h)
  find_library(URING_LIBRARY NAMES uring liburing)
  if(NOT URING_INCLUDE_DIR OR NOT URING_LIBRARY)
    message(FATAL_ERROR "liburing not found. Install liburing (Arch: pacman -S liburing).")
  endif()
  message(STATUS "liburing: ${URING_LIBRARY}")
endif()

add_library(qwfn_core STATIC
  src/qwfn_model.cpp
  src/qwfn_io.cpp
  src/qwfn_expert_cache.cpp
  src/qwfn_ple.cpp
  src/qwfn_weights.cpp
  src/qwfn_graph.cpp
  src/qwfn_state.cpp
  src/qwfn_engine.cpp
  src/qwfn_prefill.cpp
  src/qwfn_vision.cpp
)
target_include_directories(qwfn_core PUBLIC
  "${LLAMA_CPP_ROOT}/vendor/stb"
  src
  "${LLAMA_CPP_ROOT}/ggml/include"
  "${LLAMA_CPP_ROOT}/ggml/src"       # ggml-impl.h: the graph struct, for the per-node profiler only
)
if(NOT WIN32)
  target_include_directories(qwfn_core PUBLIC "${URING_INCLUDE_DIR}")
  target_link_libraries(qwfn_core PUBLIC ggml_base ggml_all Threads::Threads "${URING_LIBRARY}")
  target_compile_options(qwfn_core PRIVATE -O3 ${QWFN_ARCH_FLAGS} -Wall -Wextra -Wno-unused-parameter)
else()
  target_link_libraries(qwfn_core PUBLIC ggml_base ggml_all Threads::Threads)
  target_compile_options(qwfn_core PRIVATE /O2 /W4 /permissive- /wd4100)   # /wd4100 = unused parameter, matches -Wno-unused-parameter
endif()

# Per-tool optimization flags: GNU flags on Linux, MSVC/clang-cl flags on
# Windows. clang-cl accepts -O2/-O3 directly; /W4 replaces -Wall -Wextra for
# the tools (qwfn_core carries the full warning set above).
function(qwfn_opts target)
  if(WIN32)
    target_compile_options(${target} PRIVATE /O2 /W4 /permissive-)
  else()
    target_compile_options(${target} PRIVATE -O3 ${QWFN_ARCH_FLAGS})
  endif()
endfunction()

add_executable(qwfn-iobench tools/qwfn_iobench.cpp)
target_link_libraries(qwfn-iobench PRIVATE qwfn_core)
qwfn_opts(qwfn-iobench)

# Public-seam tests for the io layer (runs identically on Linux and Windows).
add_executable(qwfn-io-test tools/qwfn_io_test.cpp)
target_link_libraries(qwfn-io-test PRIVATE qwfn_core)
add_test(NAME io COMMAND qwfn-io-test)

add_executable(qwfn-inspect tools/qwfn_inspect.cpp)
target_link_libraries(qwfn-inspect PRIVATE qwfn_core)
qwfn_opts(qwfn-inspect)

add_executable(qwfn-test-hc tools/qwfn_test_hc.cpp)
target_link_libraries(qwfn-test-hc PRIVATE qwfn_core)
qwfn_opts(qwfn-test-hc)

add_executable(qwfn-logits tools/qwfn_logits.cpp)
target_link_libraries(qwfn-logits PRIVATE qwfn_core)
qwfn_opts(qwfn-logits)

add_executable(qwfn-refdump tools/qwfn_refdump.cpp)
target_include_directories(qwfn-refdump PRIVATE "${LLAMA_CPP_ROOT}/include" "${LLAMA_CPP_ROOT}/ggml/include" src)
target_link_libraries(qwfn-refdump PRIVATE llama_ref ggml_base ggml_all)
qwfn_opts(qwfn-refdump)

add_executable(qwfn-run tools/qwfn_run.cpp)
target_link_libraries(qwfn-run PRIVATE qwfn_core)
qwfn_opts(qwfn-run)

# qwfn_vocab.cpp is deliberately NOT in qwfn_core: only this binary should
# carry a libllama dependency.
add_executable(qwfn-vtest tools/qwfn_vtest.cpp)
target_link_libraries(qwfn-vtest PRIVATE qwfn_core)
qwfn_opts(qwfn-vtest)

# cpp-httplib 0.54 is split header/source, so its .cpp joins the target.
add_executable(qwfn-server tools/qwfn_server.cpp src/qwfn_vocab.cpp
               "${LLAMA_CPP_ROOT}/vendor/cpp-httplib/httplib.cpp")
target_include_directories(qwfn-server PRIVATE
  "${LLAMA_CPP_ROOT}/include"
  "${LLAMA_CPP_ROOT}/vendor"
  "${LLAMA_CPP_ROOT}/vendor/cpp-httplib")
target_link_libraries(qwfn-server PRIVATE qwfn_core llama_ref)
qwfn_opts(qwfn-server)

add_executable(qwfn-chat tools/qwfn_chat.cpp src/qwfn_vocab.cpp)
target_include_directories(qwfn-chat PRIVATE "${LLAMA_CPP_ROOT}/include")
target_link_libraries(qwfn-chat PRIVATE qwfn_core llama_ref)
qwfn_opts(qwfn-chat)

add_executable(qwfn-tok tools/qwfn_tok.cpp src/qwfn_vocab.cpp)
target_include_directories(qwfn-tok PRIVATE "${LLAMA_CPP_ROOT}/include")
target_link_libraries(qwfn-tok PRIVATE qwfn_core llama_ref)
qwfn_opts(qwfn-tok)

add_executable(qwfn-gen tools/qwfn_gen.cpp)
target_link_libraries(qwfn-gen PRIVATE qwfn_core)
qwfn_opts(qwfn-gen)

add_executable(qwfn-train-predictor tools/qwfn_train_predictor.cpp)
target_link_libraries(qwfn-train-predictor PRIVATE qwfn_core)
qwfn_opts(qwfn-train-predictor)
target_compile_definitions(qwfn-train-predictor PRIVATE LLAMA_CPP_BUILD_DIR="${LLAMA_CPP_BUILD}")

# --- Windows: stage the shared libraries next to the executables --------------------
# The same contract as scripts/package.sh on Linux (its bundle copies every .so
# into bin/): the loader, the engine's backend directory and the console's PATH
# injection all find the libraries in one place without PATH setup. In a source
# checkout that keeps a plain `cmake --build build-win` runnable everywhere.
if(WIN32)
  set(QWFN_STAGE_LIBS "")
  foreach(dll ggml-base ggml ggml-cpu ggml-cuda llama)
    if(EXISTS "${LLAMA_CPP_BUILD}/${dll}.dll")
      list(APPEND QWFN_STAGE_LIBS "${LLAMA_CPP_BUILD}/${dll}.dll")
    endif()
  endforeach()
  # ggml-cuda's cuBLAS imports (cudart is linked statically into it). CUDA
  # 13.4 keeps the x64 runtimes in bin/x64; older toolkits put them in bin/.
  set(_cuda_root "${CUDA_TOOLKIT_ROOT}")
  if(NOT _cuda_root AND DEFINED ENV{CUDA_PATH})
    set(_cuda_root "$ENV{CUDA_PATH}")
  endif()
  if(NOT _cuda_root)
    set(_cuda_root "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4")
  endif()
  foreach(cub cublas64_13 cublasLt64_13)
    foreach(sub x64 .)
      if(EXISTS "${_cuda_root}/bin/${sub}/${cub}.dll")
        list(APPEND QWFN_STAGE_LIBS "${_cuda_root}/bin/${sub}/${cub}.dll")
        break()
      endif()
    endforeach()
  endforeach()
  if(QWFN_STAGE_LIBS)
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/stage-dlls")
    foreach(lib ${QWFN_STAGE_LIBS})
      file(COPY "${lib}" DESTINATION "${CMAKE_BINARY_DIR}/stage-dlls")
    endforeach()
    add_custom_target(qwfn-stage-dlls ALL DEPENDS ${QWFN_STAGE_LIBS}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${QWFN_STAGE_LIBS} $<TARGET_FILE_DIR:qwfn-server>
      COMMENT "staging ggml/llama/CUDA DLLs next to the executables")
    add_dependencies(qwfn-server qwfn-stage-dlls)
  endif()
endif()
