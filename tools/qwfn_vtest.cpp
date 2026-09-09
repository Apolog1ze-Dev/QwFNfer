// qwfn-vtest -- run the vision tower on one image and report the embeddings.
#include "qwfn_vision.h"
#include "ggml-backend.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace qwfn;

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: qwfn-vtest <mmproj.gguf> <image> [--cpu] [--reps N]\n"); return 1; }
    bool use_gpu = true; int reps = 2;
    for (int i = 3; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--cpu") use_gpu = false;
        else if (a == "--reps" && i + 1 < argc) reps = atoi(argv[++i]);
    }
    ggml_backend_load_all_from_path((std::string(getenv("HOME")) + "/.unsloth/llama.cpp/build/bin").c_str());

    ggml_backend_t be = nullptr;
    if (use_gpu) {
        for (size_t i = 0; i < ggml_backend_dev_count() && !be; i++) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) be = ggml_backend_dev_init(d, nullptr);
        }
    }
    if (!be) be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    printf("backend: %s\n", ggml_backend_name(be));

    std::string err;
    vision_encoder v;
    if (!v.load(argv[1], be, ggml_backend_get_default_buffer_type(be), err)) {
        fprintf(stderr, "load: %s\n", err.c_str()); return 1;
    }
    image_u8 img;
    if (!img.load(argv[2], err)) { fprintf(stderr, "image: %s\n", err.c_str()); return 1; }
    printf("image: %dx%d\n", img.nx, img.ny);

    std::vector<float> emb; int n_out = 0, gw = 0, gh = 0;
    // Encode twice: the first call pays one-time costs (graph allocation, CUDA warm-up).
    double t_last = 0;
    for (int r = 0; r < reps; r++) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!v.encode(img, emb, n_out, gw, gh, err)) { fprintf(stderr, "encode: %s\n", err.c_str()); return 1; }
        t_last = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("encode %d: %.3f s\n", r, t_last);
    }

    double sum = 0, sq = 0; float mx = 0; int nonfinite = 0;
    for (float x : emb) {
        if (!std::isfinite(x)) { nonfinite++; continue; }
        sum += x; sq += (double) x * x; if (std::fabs(x) > mx) mx = std::fabs(x);
    }
    const double n = (double) emb.size();
    printf("grid %dx%d -> %d tokens of %u dims\n", gw, gh, n_out, v.hp().proj_dim);
    printf("mean %.5f  rms %.5f  absmax %.4f  nonfinite %d\n",
           sum / n, std::sqrt(sq / n), mx, nonfinite);
    printf("first 8: ");
    for (int i = 0; i < 8 && i < (int) emb.size(); i++) printf("%.4f ", emb[i]);
    printf("\n");
    return nonfinite ? 1 : 0;
}
