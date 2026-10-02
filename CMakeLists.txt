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
# the pre-C23 strtol/sscanf symbols, -fno-math-errno the pre-2.43 sqrtf), and the ggml/CUDA
# libraries found next to the binaries ($ORIGIN). Every target the bundle ships takes
# QWFN_ARCH_FLAGS, the -O2 ones included: without it they are built for whatever -march the
# build machine's compiler happens to default to, which is nobody's guarantee of x86-64-v3.
#
# -static-libstdc++ -static-libgcc used to be here as well, to keep the bundle off the user's
# C++ runtime. They also copy the build machine's runtime into the binaries, and the linker
# ORs the ISA property of every object it links into the output: on a distribution that
# compiles libstdc++.a and libgcc.a for AVX-512 that leaves the binaries marked "x86-64-v4
# needed" whatever -march the engine itself was built with, after which glibc's loader
# refuses to start them on any CPU without AVX-512. It bought little in return -- libllama
# and the ggml libraries link libstdc++ dynamically, so the bundle needs the user's copy
# either way -- and the C++ runtime is one thing every distribution has.
#
# Windows builds with MSVC or clang-cl (MSVC is true for both) or with MinGW-w64; the flags
# follow the compiler, the platform pieces (no liburing, DLLs and import libraries) follow
# WIN32. There, portable or not, the target is AVX2 (/arch:AVX2, -march=x86-64-v3): MSVC has
# no -march=native, and it is the baseline the Linux bundle already requires.
option(QWFN_PORTABLE "Relocatable release build" OFF)
if(MSVC)
  set(QWFN_ARCH_FLAGS /arch:AVX2)
  # The POSIX getenv/fopen/strerror calls this code keeps portable trip the CRT's
  # deprecation warnings; /wd4100 is -Wno-unused-parameter.
  add_compile_options(/wd4996 /wd4100)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
elseif(WIN32)
  set(QWFN_ARCH_FLAGS -march=x86-64-v3 -mtune=generic)
elseif(QWFN_PORTABLE)
  set(QWFN_ARCH_FLAGS -march=x86-64-v3 -mtune=generic)
  add_compile_options(-include "${CMAKE_SOURCE_DIR}/cmake/glibc_compat.h" -fno-math-errno)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
  set(CMAKE_INSTALL_RPATH "$ORIGIN")
else()
  set(QWFN_ARCH_FLAGS -march=native)
endif()

if(NOT EXISTS "${LLAMA_CPP_ROOT}/ggml/include/ggml.h")
  message(FATAL_ERROR "ggml headers not found under ${LLAMA_CPP_ROOT}. Pass -DLLAMA_CPP_ROOT=<path>.")
endif()

# The ggml/llama shared libraries, under the platform's names. On Windows the DLLs are in
# LLAMA_CPP_BUILD and are linked through import libraries that llama.cpp's build writes
# elsewhere: <build>/ggml/src and <build>/src (Release/ under them with a multi-config
# generator, whose DLLs are then in bin/Release), lib/ in an installed tree; name.lib from
# MSVC and clang-cl, libname.dll.a from MinGW.
function(qwfn_import_lib target name subdir)
  add_library(${target} SHARED IMPORTED)
  if(WIN32)
    set(implib "")
    foreach(d "${LLAMA_CPP_BUILD}/../${subdir}" "${LLAMA_CPP_BUILD}/../../${subdir}/Release"
              "${LLAMA_CPP_BUILD}/../../${subdir}/RelWithDebInfo" "${LLAMA_CPP_BUILD}/../lib" "${LLAMA_CPP_BUILD}")
      foreach(f "${name}.lib" "lib${name}.dll.a")
        if(NOT implib AND EXISTS "${d}/${f}")
          get_filename_component(implib "${d}/${f}" ABSOLUTE)
        endif()
      endforeach()
    endforeach()
    set(dll "${LLAMA_CPP_BUILD}/${name}.dll")
    if(NOT EXISTS "${dll}")
      set(dll "${LLAMA_CPP_BUILD}/lib${name}.dll")   # MinGW names libllama this way
    endif()
    if(NOT implib OR NOT EXISTS "${dll}")
      message(FATAL_ERROR "${name}.dll or its import library not found from ${LLAMA_CPP_BUILD}. "
                          "Build llama.cpp with -DBUILD_SHARED_LIBS=ON first, and pass -DLLAMA_CPP_BUILD=<the directory holding ${name}.dll>.")
    endif()
    set_target_properties(${target} PROPERTIES IMPORTED_IMPLIB "${implib}" IMPORTED_LOCATION "${dll}")
  else()
    if(NOT EXISTS "${LLAMA_CPP_BUILD}/lib${name}.so.0")
      message(FATAL_ERROR "lib${name}.so.0 not found in ${LLAMA_CPP_BUILD}. Build llama.cpp first or pass -DLLAMA_CPP_BUILD=<dir>.")
    endif()
    set_target_properties(${target} PROPERTIES IMPORTED_LOCATION "${LLAMA_CPP_BUILD}/lib${name}.so.0")
  endif()
endfunction()

qwfn_import_lib(ggml_base ggml-base ggml/src)
qwfn_import_lib(ggml_all  ggml      ggml/src)
# Two things need libllama: qwfn-refdump drives llama.cpp's own graph to produce
# the reference intermediates our graph is bisected against, and qwfn-chat
# borrows its tokenizer (opened vocab_only, so no weights are loaded).
qwfn_import_lib(llama_ref llama     src)

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
target_link_libraries(qwfn_core PUBLIC ggml_base ggml_all Threads::Threads)
if(NOT WIN32)
  target_include_directories(qwfn_core PUBLIC "${URING_INCLUDE_DIR}")
  target_link_libraries(qwfn_core PUBLIC "${URING_LIBRARY}")
endif()
if(MSVC)
  target_compile_options(qwfn_core PRIVATE /O2 ${QWFN_ARCH_FLAGS} /W4 /permissive-)
else()
  target_compile_options(qwfn_core PRIVATE -O3 ${QWFN_ARCH_FLAGS} -Wall -Wextra -Wno-unused-parameter)
endif()

# Per-tool optimization flags, in the compiler's own syntax.
function(qwfn_opts target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /O2 ${QWFN_ARCH_FLAGS} /permissive-)
  else()
    target_compile_options(${target} PRIVATE -O3 ${QWFN_ARCH_FLAGS})
  endif()
endfunction()

add_executable(qwfn-iobench tools/qwfn_iobench.cpp)
target_link_libraries(qwfn-iobench PRIVATE qwfn_core)
qwfn_opts(qwfn-iobench)

# Public-seam tests for the io layer (runs identically on Linux and Windows): ctest, or
# build/qwfn-io-test directly.
enable_testing()
add_executable(qwfn-io-test tools/qwfn_io_test.cpp)
target_link_libraries(qwfn-io-test PRIVATE qwfn_core)
qwfn_opts(qwfn-io-test)
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
if(WIN32)
  target_link_libraries(qwfn-server PRIVATE ws2_32)   # httplib's sockets; MSVC links it by pragma, MinGW does not
endif()
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
# The same contract as scripts/package.sh on Linux (its bundle copies every .so into
# bin/): the loader, ggml's backend search (the executable's directory) and the console
# all find the libraries in one place, so a plain build runs with no PATH setup. Every
# ggml/llama DLL of the llama.cpp build -- the CPU backend may be several, one per ISA
# level -- and, when the CUDA backend is there, the CUDA runtime DLLs NVIDIA lets
# applications redistribute (cudart, cublas, cublasLt), from CUDA_PATH's bin/ or bin/x64/
# (where CUDA 13 keeps them).
if(WIN32)
  file(GLOB QWFN_STAGE_LIBS "${LLAMA_CPP_BUILD}/ggml*.dll" "${LLAMA_CPP_BUILD}/llama*.dll" "${LLAMA_CPP_BUILD}/libllama*.dll")
  if(EXISTS "${LLAMA_CPP_BUILD}/ggml-cuda.dll")
    set(_cuda_root "${CUDAToolkit_ROOT}")
    if(NOT _cuda_root AND DEFINED ENV{CUDA_PATH})
      file(TO_CMAKE_PATH "$ENV{CUDA_PATH}" _cuda_root)
    endif()
    if(_cuda_root)
      file(GLOB _cuda_dlls "${_cuda_root}/bin/x64/cudart64_*.dll" "${_cuda_root}/bin/x64/cublas64_*.dll" "${_cuda_root}/bin/x64/cublasLt64_*.dll"
                           "${_cuda_root}/bin/cudart64_*.dll" "${_cuda_root}/bin/cublas64_*.dll" "${_cuda_root}/bin/cublasLt64_*.dll")
      list(APPEND QWFN_STAGE_LIBS ${_cuda_dlls})
    else()
      message(WARNING "ggml-cuda.dll found but not the CUDA toolkit (set CUDA_PATH): the CUDA runtime DLLs are not staged")
    endif()
  endif()
  if(QWFN_STAGE_LIBS)
    add_custom_target(qwfn-stage-dlls ALL
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${QWFN_STAGE_LIBS} "$<TARGET_FILE_DIR:qwfn-server>"
      COMMENT "staging the ggml/llama/CUDA DLLs next to the executables")
    add_dependencies(qwfn-server qwfn-stage-dlls)
  endif()
endif()
