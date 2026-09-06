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
option(QWFN_PORTABLE "Relocatable release build" OFF)
if(QWFN_PORTABLE)
  set(QWFN_ARCH_FLAGS -march=x86-64-v3 -mtune=generic)
  add_compile_options(-include "${CMAKE_SOURCE_DIR}/cmake/glibc_compat.h" -fno-math-errno)
  add_link_options(-static-libstdc++ -static-libgcc)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
  set(CMAKE_INSTALL_RPATH "$ORIGIN")
else()
  set(QWFN_ARCH_FLAGS -march=native)
endif()

if(NOT EXISTS "${LLAMA_CPP_ROOT}/ggml/include/ggml.h")
  message(FATAL_ERROR "ggml headers not found under ${LLAMA_CPP_ROOT}. Pass -DLLAMA_CPP_ROOT=<path>.")
endif()

add_library(ggml_base SHARED IMPORTED)
set_target_properties(ggml_base PROPERTIES IMPORTED_LOCATION "${LLAMA_CPP_BUILD}/libggml-base.so.0")
add_library(ggml_all SHARED IMPORTED)
set_target_properties(ggml_all PROPERTIES IMPORTED_LOCATION "${LLAMA_CPP_BUILD}/libggml.so.0")

# Two things need libllama: qwfn-refdump drives llama.cpp's own graph to produce
# the reference intermediates our graph is bisected against, and qwfn-chat
# borrows its tokenizer (opened vocab_only, so no weights are loaded).
add_library(llama_ref SHARED IMPORTED)
set_target_properties(llama_ref PROPERTIES IMPORTED_LOCATION "${LLAMA_CPP_BUILD}/libllama.so.0")

find_package(Threads REQUIRED)

# liburing is mandatory: it is the whole point of the engine.
find_path(URING_INCLUDE_DIR liburing.h)
find_library(URING_LIBRARY NAMES uring liburing)
if(NOT URING_INCLUDE_DIR OR NOT URING_LIBRARY)
  message(FATAL_ERROR "liburing not found. Install liburing (Arch: pacman -S liburing).")
endif()
message(STATUS "liburing: ${URING_LIBRARY}")

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
  "${URING_INCLUDE_DIR}"
)
target_link_libraries(qwfn_core PUBLIC ggml_base ggml_all Threads::Threads "${URING_LIBRARY}")
target_compile_options(qwfn_core PRIVATE -O3 ${QWFN_ARCH_FLAGS} -Wall -Wextra -Wno-unused-parameter)

add_executable(qwfn-iobench tools/qwfn_iobench.cpp)
target_link_libraries(qwfn-iobench PRIVATE qwfn_core)
target_compile_options(qwfn-iobench PRIVATE -O3 ${QWFN_ARCH_FLAGS})

add_executable(qwfn-inspect tools/qwfn_inspect.cpp)
target_link_libraries(qwfn-inspect PRIVATE qwfn_core)
target_compile_options(qwfn-inspect PRIVATE -O3 ${QWFN_ARCH_FLAGS})

add_executable(qwfn-test-hc tools/qwfn_test_hc.cpp)
target_link_libraries(qwfn-test-hc PRIVATE qwfn_core)
target_compile_options(qwfn-test-hc PRIVATE -O3 ${QWFN_ARCH_FLAGS})

add_executable(qwfn-logits tools/qwfn_logits.cpp)
target_link_libraries(qwfn-logits PRIVATE qwfn_core)
target_compile_options(qwfn-logits PRIVATE -O3 ${QWFN_ARCH_FLAGS})

add_executable(qwfn-refdump tools/qwfn_refdump.cpp)
target_include_directories(qwfn-refdump PRIVATE "${LLAMA_CPP_ROOT}/include" "${LLAMA_CPP_ROOT}/ggml/include")
target_link_libraries(qwfn-refdump PRIVATE llama_ref ggml_base ggml_all)
target_compile_options(qwfn-refdump PRIVATE -O2)

add_executable(qwfn-run tools/qwfn_run.cpp)
target_link_libraries(qwfn-run PRIVATE qwfn_core)
target_compile_options(qwfn-run PRIVATE -O3 ${QWFN_ARCH_FLAGS})

# qwfn_vocab.cpp is deliberately NOT in qwfn_core: only this binary should
# carry a libllama dependency.
add_executable(qwfn-vtest tools/qwfn_vtest.cpp)
target_link_libraries(qwfn-vtest PRIVATE qwfn_core)
target_compile_options(qwfn-vtest PRIVATE -O3 ${QWFN_ARCH_FLAGS})

# cpp-httplib 0.54 is split header/source, so its .cpp joins the target.
add_executable(qwfn-server tools/qwfn_server.cpp src/qwfn_vocab.cpp
               "${LLAMA_CPP_ROOT}/vendor/cpp-httplib/httplib.cpp")
target_include_directories(qwfn-server PRIVATE
  "${LLAMA_CPP_ROOT}/include"
  "${LLAMA_CPP_ROOT}/vendor"
  "${LLAMA_CPP_ROOT}/vendor/cpp-httplib")
target_link_libraries(qwfn-server PRIVATE qwfn_core llama_ref)
target_compile_options(qwfn-server PRIVATE -O2)

add_executable(qwfn-chat tools/qwfn_chat.cpp src/qwfn_vocab.cpp)
target_include_directories(qwfn-chat PRIVATE "${LLAMA_CPP_ROOT}/include")
target_link_libraries(qwfn-chat PRIVATE qwfn_core llama_ref)
target_compile_options(qwfn-chat PRIVATE -O3 ${QWFN_ARCH_FLAGS})

add_executable(qwfn-tok tools/qwfn_tok.cpp src/qwfn_vocab.cpp)
target_include_directories(qwfn-tok PRIVATE "${LLAMA_CPP_ROOT}/include")
target_link_libraries(qwfn-tok PRIVATE qwfn_core llama_ref)
target_compile_options(qwfn-tok PRIVATE -O2)

add_executable(qwfn-gen tools/qwfn_gen.cpp)
target_link_libraries(qwfn-gen PRIVATE qwfn_core)
target_compile_options(qwfn-gen PRIVATE -O3 ${QWFN_ARCH_FLAGS})

add_executable(qwfn-train-predictor tools/qwfn_train_predictor.cpp)
target_link_libraries(qwfn-train-predictor PRIVATE qwfn_core)
target_compile_options(qwfn-train-predictor PRIVATE -O3 ${QWFN_ARCH_FLAGS})
target_compile_definitions(qwfn-train-predictor PRIVATE LLAMA_CPP_BUILD_DIR="${LLAMA_CPP_BUILD}")
