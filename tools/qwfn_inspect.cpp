// qwfn-inspect -- read a Qwen3.8-Flash-Next checkpoint and print exactly what
// the engine will have to move around at run time.

#include "qwfn_model.h"
#include "qwfn_expert_cache.h"

#include <cinttypes>
#include <cstdio>
#include <map>
#include <string>

using namespace qwfn;

static const char * tname(ggml_type t) { return ggml_type_name(t); }
static double GB(uint64_t b) { return (double) b / 1e9; }

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: qwfn-inspect <any-shard.gguf> [cold-shard.gguf]\n");
        return 1;
    }
    model_index mi;
    std::string err;
    if (!mi.load(argv[1], err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }

    printf("=== %s ===\n%s\n\n", mi.arch.c_str(), mi.hp().summary().c_str());

    printf("shards (%zu):\n", mi.shard_paths().size());
    for (size_t i = 0; i < mi.shard_paths().size(); i++)
        printf("  [%zu] %s\n", i, mi.shard_paths()[i].c_str());

    printf("\nweight budget:\n");
    printf("  routed experts : %8.2f GB\n", GB(mi.bytes_experts()));
    printf("  PLE n-gram tbl : %8.2f GB\n", GB(mi.bytes_ple()));
    printf("  dense core     : %8.2f GB   <- fits in VRAM\n", GB(mi.bytes_dense()));
    printf("  total          : %8.2f GB\n",
           GB(mi.bytes_experts() + mi.bytes_ple() + mi.bytes_dense()));

    // Per-layer expert block layout, grouped by identical quant signature.
    printf("\nexpert block layout (per layer, one expert = gate+up+down):\n");
    std::map<std::string, std::pair<int, uint32_t>> sig;
    for (uint32_t il = 0; il < mi.hp().n_layer; il++) {
        char buf[160];
        uint32_t tot = 0;
        std::string s;
        for (int q = 0; q < EXPERT_NPARTS; q++) {
            const byte_range r = mi.expert_range(il, 0, (expert_part) q);
            const tensor_ref * t = mi.find("blk." + std::to_string(il) + "." +
                (q == 0 ? "ffn_gate_exps.weight" : q == 1 ? "ffn_up_exps.weight" : "ffn_down_exps.weight"));
            snprintf(buf, sizeof(buf), "%s:%uK ", t ? tname(t->type) : "?", r.nbytes / 1024);
            s += buf;
            tot += r.nbytes;
        }
        snprintf(buf, sizeof(buf), "%s= %u KiB", s.c_str(), tot / 1024);
        auto & e = sig[buf];
        e.first++;
        e.second = tot;
    }
    for (auto & kv : sig)
        printf("  %-58s  x%d layers\n", kv.first.c_str(), kv.second.first);

    // O_DIRECT viability: the slice stride must be a whole number of 512B blocks.
    printf("\nO_DIRECT alignment check (slice stride %% 512):\n");
    bool ok = true;
    for (uint32_t il = 0; il < mi.hp().n_layer; il++) {
        for (int q = 0; q < EXPERT_NPARTS; q++) {
            const byte_range a = mi.expert_range(il, 0, (expert_part) q);
            const byte_range b = mi.expert_range(il, 1, (expert_part) q);
            if ((b.offset - a.offset) % 512 != 0) {
                printf("  layer %u part %d: stride %" PRIu64 " NOT aligned\n", il, q, b.offset - a.offset);
                ok = false;
            }
        }
    }
    printf("  %s\n", ok ? "all expert slices are 512B-aligned -> O_DIRECT fast path usable"
                        : "misaligned slices present -> buffered I/O fallback required");

    if (const tensor_ref * t = mi.ple_tensor()) {
        const byte_range r0 = mi.ple_row_range(0), r1 = mi.ple_row_range(1);
        printf("\nPLE table:\n");
        printf("  tensor       : %s [%" PRId64 ", %" PRId64 "] %s\n",
               t->name.c_str(), t->ne[0], t->ne[1], tname(t->type));
        printf("  rows         : %" PRId64 "  (%u heads)\n", t->ne[1], mi.hp().ple_n_head());
        printf("  row size     : %u bytes   stride %" PRIu64 "\n", r0.nbytes, r1.offset - r0.offset);
        printf("  per token    : %u rows = %u bytes payload, %u random reads\n",
               mi.hp().ple_n_head(), mi.hp().ple_n_head() * r0.nbytes, mi.hp().ple_n_head());
        printf("  head vocab   :");
        for (size_t i = 0; i < mi.hp().ple_head_vocab_sizes.size() && i < 4; i++)
            printf(" %" PRIu64, mi.hp().ple_head_vocab_sizes[i]);
        printf(" ... (%zu heads)\n", mi.hp().ple_head_vocab_sizes.size());
    }

    // Traffic implied by one generated token.
    const hparams & hp = mi.hp();
    uint64_t per_token = 0;
    for (uint32_t il = 0; il < hp.n_layer; il++) {
        uint32_t blkb = 0;
        for (int q = 0; q < EXPERT_NPARTS; q++) blkb += mi.expert_range(il, 0, (expert_part) q).nbytes;
        per_token += (uint64_t) blkb * hp.n_expert_used;
    }
    printf("\nper generated token:\n");
    printf("  expert weights : %.0f MB  (%u layers x %u experts)\n",
           per_token / 1e6, hp.n_layer, hp.n_expert_used);
    printf("  at 15 tok/s    : %.1f GB/s of expert bandwidth required\n", per_token * 15 / 1e9);
    printf("  at 20 tok/s    : %.1f GB/s\n", per_token * 20 / 1e9);

    if (argc >= 3) {
        model_index cold;
        if (cold.load(argv[2], err)) {
            uint64_t cold_pt = 0;
            for (uint32_t il = 0; il < hp.n_layer; il++) {
                uint32_t b = 0;
                for (int q = 0; q < EXPERT_NPARTS; q++) b += cold.expert_range(il, 0, (expert_part) q).nbytes;
                cold_pt += (uint64_t) b * hp.n_expert_used;
            }
            printf("\ncold tier (%s):\n", argv[2]);
            printf("  experts        : %.2f GB (vs %.2f GB hot)\n", GB(cold.bytes_experts()), GB(mi.bytes_experts()));
            printf("  per token      : %.0f MB  -> miss cost x%.2f\n",
                   cold_pt / 1e6, (double) cold_pt / (double) per_token);
        } else {
            printf("\ncold tier: failed to load (%s)\n", err.c_str());
        }
    }
    return 0;
}
