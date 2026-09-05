#pragma once
// The engine: position-tracked evaluation over a growing sequence.
//
// Two paths, because the MoE behaves differently by batch size:
//
//   decode  (1 token)  every layer selects exactly n_expert_used experts, so the
//                      expert cache serves them and each runs one matmul in
//                      place from VRAM or RAM.
//   prefill (N tokens) each token picks its own experts, so the per-layer union
//                      approaches all 512. That is a streaming pass, not a
//                      caching one -- each expert is read once and reused across
//                      all its tokens -- so it reads from the mmap'd file and
//                      leaves the cache undisturbed for decode.
//
// State (KV, indexer keys, DeltaNet recurrence, conv history) persists across
// calls; n_past advances.

#include "qwfn_expert_cache.h"
#include "qwfn_graph.h"
#include "qwfn_model.h"
#include "qwfn_prefill.h"
#include "qwfn_state.h"
#include "qwfn_weights.h"

#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace qwfn {

// Widest speculative prediction the engine will read back per layer.
static constexpr uint32_t QWFN_SPEC_MAX = 16;

struct engine_config {
    uint32_t  n_ctx      = 4096;
    uint32_t  n_batch    = 256;          // largest prefill ubatch
    size_t    ram_bytes  = 8ull  << 30;
    size_t    vram_bytes = 0;
    bool      use_gpu    = true;
    bool      use_qsa    = true;
    bool      use_cold_tier = false;
    int       n_threads  = 8;
    int       evict_policy = 0;   // 0 lru, 1 lfu, 2 hybrid
    uint32_t  promote_per_layer = 2;
    // Thread pool by default. On this filesystem io_uring_submit() executes the
    // reads inline instead of queueing them, so a burst never gets the
    // concurrency it asked for; blocking preads on worker threads do. Measured
    // +10% end to end on identical work. `--io-uring` selects the old path.
    // Stage prefill experts in VRAM and run the batched MoE on the GPU.
    bool      prefill_on_gpu = true;
    // Cap on n_kv * ubatch, bounding the per-call input arena. 0 disables.
    // 48M: at 128K context that is a 384-query attention chunk. The chunk's
    // graph holds several [n_kv, T] tensors; at 96M they needed a 1.3 GB
    // arena on top of the MoE chunk's and the tier had not left that free.
    uint64_t  ubatch_kv_product = 48ull << 20;
    bool      reuse_graphs = true;   // replay the position-independent layer graphs
    bool      io_threads = true;
    unsigned  io_workers = 16;
    bool      speculate = true;     // prefetch the next layer's predicted experts
    // How many of the predicted top-k to actually fetch (clamped to
    // n_expert_used). Measured on a replayed fixed sequence: depth 0 -> 11.6
    // tok/s / 85.2% hit, depth 8 -> 12.1 / 92.8%, depth 10 -> 12.4 / 94.7%.
    // Prefetching every prediction wins despite ~23% of them being wrong,
    // because a wrong prefetch only costs bandwidth while a right one removes a
    // blocking read from the critical path.
    uint32_t  speculate_depth = 10;   // may exceed n_expert_used, up to QWFN_SPEC_MAX
    // How many layers ahead to predict. 1 = the L+1 lookahead above. 2 also
    // predicts L+2 from the same residual, so its reads get two layers of
    // compute to land in rather than one.
    //
    // Measured on a replayed sequence (identical routing): 12.46 tok/s at 1,
    // 12.48 at 2 with depth2=4, 12.4 at 2 with depth2=10. It is a wash. The
    // L+2 prediction is only 69% accurate against 79% for L+1, so the extra
    // reads mostly miss -- 3,600 wasted blocks against 2,300 -- and the second
    // router matmul adds ~0.6 s of graphA per 250 tokens, which cancels what
    // the wider window buys. Off by default; the machinery stays because it
    // becomes worthwhile if prediction accuracy improves.
    uint32_t  speculate_ahead = 1;
    // Depth for the L+2 set alone, when speculate_ahead >= 2. Fetching all ten
    // of its picks measured slower than four: the misses burn NVMe bandwidth
    // the demand fetches need and evict blocks that were still useful.
    uint32_t  speculate_depth2 = 4;
    // Learned routing predictor file (scripts/train_predictor.py); empty = the
    // next layer's own router on the current residual.
    std::string predictor_path;
    // Confidence gate on the speculative reads. A predicted candidate's margin
    // is its distance in router logits from the routing cut-off: for a rank
    // inside the predicted top-n_expert_used, its logit minus the first
    // excluded one; for a rank past it, the last included logit minus its own.
    // A candidate whose margin is below spec_margin is not read. Changes only
    // which reads are issued, never what is computed. 0 = off.
    float     spec_margin = 0.0f;
    // Apply the gate only while at least this many speculative reads are in
    // flight, i.e. when the disk is the constraint; with an idle disk every
    // prediction is worth reading (measured). 0 = always.
    uint32_t  spec_gate_inflight = 0;
    // Predict layer L+1's routing by running L+1's own token mixer (PLE
    // injection, attention mixer, DeltaNet or decode sparse attention, combine)
    // on the approximate residual inside layer L's graph, with every state
    // write suppressed, and applying the router to what that produces. The
    // block is the term the residual-only predictor cannot see. Costs one
    // extra block per layer of graph A, synchronous.
    bool      spec_block = false;
    // Which predicted layers get the block: "" = all, else "1,2,5-9,47".
    std::string spec_block_layers;
    // MTP draft head, experiment stage: the nextn GGUF (the checkpoint ships one
    // block, MTP/mtp-...gguf). Loaded whole on the GPU, run after every decoded
    // token on the trunk's last residual and the sampled token, and its draft
    // scored against the token that actually followed. Nothing is verified or
    // rolled back yet: this measures the acceptance rate the rest depends on.
    std::string mtp_path;
    // Decode without waiting for expert misses: the token is computed from the
    // experts that are resident (gates renormalised), the misses' reads still
    // go out and land for next time. An approximation -- measure its NLL.
    bool      skip_miss = false;
    // Overlap the next layer's bulk expert read with this layer's upload and
    // compute during prefill. Costs a second host staging buffer (~1.8 GB),
    // taken before the RAM tier sizes itself so the memory guard sees it.
    bool      prefill_overlap = true;
    // Below this many new tokens, prefill runs as repeated single-token decodes
    // rather than the streaming batched path.
    //
    // The streaming path reads every expert of every layer -- ~53 GB -- no
    // matter how many tokens are in the ubatch, because at a useful ubatch
    // essentially all 512 are selected anyway. That is the right trade at 1024
    // tokens (~0.013 s/token) and a bad one at 25 (~0.45 s/token). The decode
    // path instead fetches only what routing actually asks for, and warms the
    // expert cache for the generation that follows instead of bypassing it.
    // This is the common case for a chat server: with prefix reuse, every turn
    // after the first feeds ~25 tokens.
    uint32_t  prefill_decode_max = 96;
    // MemAvailable clamp on the expert RAM tier: frac of MemAvailable minus
    // headroom. The default is deliberately conservative for a desktop with
    // zram; raise ram_frac to trade the rest of the machine for hit rate.
    double    ram_frac     = 0.60;
    size_t    ram_headroom = 3ull << 30;
    // Device memory the VRAM expert tier must leave behind for everything
    // allocated after it: the prefill MoE graph arena (~0.31 MB per token of
    // ubatch), the per-call input arena (kq_mask and the QSA bias, bounded by
    // ubatch_kv_product), and the 36 lazily-created replay allocators. 0 sizes
    // it from n_batch, which is what makes a large ubatch safe -- a fixed
    // reserve failed at ubatch 2048 with "prefill galloc failed".
    size_t    vram_reserve = 0;
    ggml_type type_k     = GGML_TYPE_F16;
    ggml_type type_v     = GGML_TYPE_F16;
    // After each streamed prefill layer, copy this many of the experts the
    // prompt routed to (most recent tokens first) into the RAM tier, and the
    // first prefill_warm_vram of them on to VRAM, so generation does not start
    // on a cold cache. 0 disables.
    uint32_t  prefill_warm      = 48;
    uint32_t  prefill_warm_vram = 32;
    // Prompts of 2..prefill_decode_max tokens: run the dense core once for the
    // whole prompt and serve the union of its experts through the cache in
    // chunks, uploading each chunk's RAM-resident experts into a VRAM scratch
    // so the batched MoE runs on the GPU. Token by token was the alternative,
    // at ~50 ms per token whatever the prompt. Needs a GPU.
    bool      cache_batched     = true;
    uint32_t  cache_batch_chunk = 48;   // experts per cache fetch, at most 64
    // Layer-major prefill: each layer's experts are streamed ONCE per eval
    // batch of up to n_batch tokens, and the compute runs in chunks of this
    // many tokens (the graph arena bound); the attention layers chunk further
    // to ubatch_kv_product. Before, the sweep was per ubatch and the ubatch
    // shrank with the context: 131K would have taken ~500 sweeps.
    uint32_t  prefill_chunk     = 2048;
    // Override the GGUF's indexer top-k (2048 cells). Past ~150K of context
    // this quant loses its grounding at the stock budget (llama.cpp's
    // recommendation there is 4096); 0 keeps the checkpoint's value.
    uint32_t  indexer_top_k     = 0;
};

class engine {
public:
    ~engine();
    engine() = default;
    engine(const engine &) = delete;
    engine & operator=(const engine &) = delete;

    bool init(const model_index * hot, const model_index * cold,
              const engine_config & cfg, const std::string & backend_dir, std::string & err);

    // `hist` is the whole sequence including the n_new tokens being appended --
    // the PLE n-gram window needs the predecessors, so the engine cannot work
    // from the new tokens alone. Returns logits for the final position.
    const float * eval(const int32_t * hist, int32_t n_hist, int32_t n_new, std::string & err);

    void    reset();                       // clear state, rewind to position 0

    // Substitute externally computed embeddings (vision) for the tokens at
    // absolute positions [pos, pos + n). The engine still takes token ids --
    // the placeholder <|image_pad|> ids keep the sequence and the PLE n-gram
    // window well formed -- and only their EMBEDDINGS are replaced, which is
    // exactly what the reference does.
    void set_embeddings(int32_t pos, const float * emb, int32_t n);
    void clear_embeddings() { ov_pos_.clear(); ov_.clear(); }
    int32_t n_past() const { return n_past_; }
    int64_t n_vocab() const { return n_vocab_; }

    // The vision tower shares the language model's backend and buffer type.
    ggml_backend_t             backend() const { return w_.backend(); }
    ggml_backend_buffer_type_t buft()    const { return w_.buft(); }

    const expert_cache_stats & cache_stats() const { return ec_.stats(); }
    const io_engine &          cache_io()    const { return ec_.io(); }
    std::string memory_summary() const;

    double t_prefill = 0, t_decode = 0, t_io = 0, t_warm = 0;
    double t_pf_graphA = 0, t_pf_moe = 0, t_pf_read = 0;   // where a layer-major prefill's time goes
    double t_moe_gpu = 0, t_moe_cpu = 0, t_layerA = 0;   // where decode time goes
    // The GPU MoE runs asynchronously, overlapped with the expert I/O wait and
    // the CPU MoE. t_moe_gpu then counts launch cost plus whatever the final
    // sync still had to wait -- t_moe_gpu_sync is that wait alone, and ~0 means
    // the overlap hid the GPU MoE completely.
    double t_moe_gpu_sync = 0;
    double t_layerA_rec = 0, t_layerA_attn = 0;   // replayed vs rebuilt (decode only)
    uint64_t n_layerA_rec = 0, n_layerA_attn = 0;
    // Decode attention layers, split: host-side graph build + allocation,
    // and the synchronous GPU compute. Per-call input construction (mask,
    // QSA inputs, the device allocation for them) is t_inputs, once per token.
    double t_attn_build = 0, t_attn_compute = 0, t_inputs = 0;
    uint64_t n_exp_gpu = 0, n_exp_cpu = 0, n_exp_skipped = 0;
    uint64_t pred_hits = 0, pred_total = 0;   // predicted-vs-actual expert overlap
    uint64_t pred2_hits = 0, pred2_total = 0; // same, for the two-ahead prediction
    // Prediction quality by rank and by confidence margin -- the gate's
    // calibration data, collected on every decode run -- and per predicted layer.
    static constexpr int SPEC_MARGIN_BUCKETS = 12;
    uint64_t rank_hits[QWFN_SPEC_MAX] = {}, rank_total[QWFN_SPEC_MAX] = {};
    uint64_t margin_hits[SPEC_MARGIN_BUCKETS] = {}, margin_total[SPEC_MARGIN_BUCKETS] = {};
    std::vector<uint64_t> pred_hits_layer, pred_total_layer;
    uint64_t pf_gated = 0;   // predicted candidates the margin gate declined to read
    // MTP experiment: drafts scored against the next token (decode), and against
    // the prompt's own tokens when the prompt went through in one batch.
    uint64_t mtp_n = 0, mtp_acc = 0, mtp_top3 = 0, mtp_prompt_n = 0, mtp_prompt_acc = 0;
    double   t_mtp = 0;
    static int   margin_bucket(float m);
    static float margin_edge(int b);   // lower edge of bucket b
    uint64_t check_moe_gpu_calls = 0, check_moe_cpu_calls = 0;   // QWFN_CHECK_MOE bookkeeping
    int64_t n_prefill = 0, n_decode = 0;

private:
    bool eval_batch(const int32_t * hist, int32_t n_hist, int32_t n_new, std::string & err,
                    bool cache_batched = false);
    bool eval_prefill_big(const int32_t * hist, int32_t n_hist, int32_t T, std::string & err);
    // Per-chunk attention inputs for the mask-based path: the causal mask and
    // the QSA block tables for Tc queries starting at n_past_c.
    struct attn_inputs {
        ggml_context *        ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        ggml_tensor *         kq_mask = nullptr;
        qsa_inputs            qsa;
        uint32_t              ratio = 0;
        void release() { if (buf) ggml_backend_buffer_free(buf); if (ctx) ggml_free(ctx); buf = nullptr; ctx = nullptr; }
    };
    bool build_attn_inputs(int64_t n_past_c, int64_t Tc, attn_inputs & ai, std::string & err);
    int32_t max_ubatch(int32_t n_past) const;
    void build_qsa_inputs(int64_t n_kv, int64_t T, uint32_t ratio);
    void run_on(ggml_cgraph * gf, bool gpu);

    const model_index * mi_ = nullptr;
    engine_config       cfg_;
    hparams             hp_;

    weights        w_;    // dense core, GPU
    weights        wh_;   // PLE table + (for prefill) expert tensors, host mmap
    expert_cache   ec_;
    state          st_;
    prefill_streamer pf_;   // bulk expert reader, prefill only

    // A decode-time graph for a gated-DeltaNet layer references nothing that
    // changes with position: no n_past, no n_kv, no mask. Those 36 of 48 layers
    // can therefore be built once and replayed. Each keeps its own allocator so
    // its tensor addresses stay put between tokens, which is what lets ggml's
    // CUDA graph capture survive instead of being invalidated every layer.
    struct layer_graph {
        ggml_context * ctx = nullptr;
        ggml_cgraph  * gf  = nullptr;
        ggml_gallocr_t ga  = nullptr;
    };
    std::vector<layer_graph> gA_;
    std::vector<int64_t>     gA_bucket_;   // attention layers: the block bucket the graph was built for

    // Decode-time sparse attention state; see graph_builder::sparse_attn_decode.
    // Per attention layer a cache of pooled block keys; shared static tables,
    // the block bias and the five per-token inputs. Allocated before the VRAM
    // tier so it is accounted for.
    qsa_decode_inputs          qd_;
    std::vector<ggml_tensor *> pool_cache_;   // per layer; null for recurrent layers
    ggml_context *             qctx_ = nullptr;
    ggml_backend_buffer_t      qbuf_ = nullptr;
    uint32_t                   qsa_ratio_ = 0;
    bool                       pool_dirty_ = true;   // caches stale: after a prefill or a reset
    void qsa_decode_prepare(int32_t n_past);         // per-token inputs, bias window, bucket, rebuild

    ggml_gallocr_t galloc_gpu_ = nullptr;
    ggml_gallocr_t galloc_cpu_ = nullptr;
    // The decode GPU MoE graph gets its own allocator and outlives its launch:
    // it is submitted with ggml_backend_graph_compute_async and settled after
    // the I/O wait and the CPU MoE, so it must not share an arena with any
    // graph allocated in between.
    ggml_gallocr_t galloc_moe_ = nullptr;
    ggml_context * moe_ctx_   = nullptr;   // in-flight async MoE graph (legacy path), freed at settle
    bool           moe_inflight_ = false;  // a GPU MoE graph has been launched and not settled

    // Persistent per-layer GPU MoE graph: three ggml_mul_mat_id over the
    // layer's VRAM tier, selecting experts by slot through t_gids_ with the
    // gate weights in t_gw_. Always n_expert_used entries -- an expert that is
    // not in VRAM takes slot 0 with weight 0, an exact zero contribution -- so
    // the graph's shape never changes and ggml captures it as a CUDA graph.
    // The per-expert path built ~50 nodes and launched ~50 kernels per layer,
    // 23 us per expert of almost pure CPU time; this is a handful of nodes.
    struct moe_graph {
        ggml_context * ctx = nullptr;
        ggml_cgraph  * gf  = nullptr;
        ggml_gallocr_t ga  = nullptr;
    };
    std::vector<moe_graph> gM_;
    ggml_tensor * t_gids_ = nullptr, * t_gw_ = nullptr;
    // Pinned host staging for the per-layer id/weight uploads, so they can be
    // queued asynchronously behind the promotion copies instead of waiting
    // for them on the copy engine (measured 0.1 ms per layer, 1 s per 200
    // tokens, once the VRAM tier started promoting freely).
    ggml_context *        pctx_ = nullptr;
    ggml_backend_buffer_t pbuf_ = nullptr;
    ggml_tensor *         p_gids_ = nullptr, * p_gw_ = nullptr;
    // In-graph VRAM MoE: per-layer residency tables on the device (slot and
    // mask by expert id), their pinned staging, and the version each holds.
    bool                  moe_in_graph_ = false;
    ggml_context *        vctx_ = nullptr;   ggml_backend_buffer_t vbuf_ = nullptr;
    std::vector<ggml_tensor *> t_vslot_, t_vmask_;
    ggml_tensor *         p_vslot_ = nullptr, * p_vmask_ = nullptr;
    std::vector<uint64_t> vslot_ver_;
    int                   n_late_ = 0;   // slots of the late fold (experts promoted this token)
    // Learned routing predictor: per layer an F16 [n_embd, n_expert] head and an
    // F32 bias, used in place of that layer's router when predicting its routing.
    ggml_context *        predctx_ = nullptr; ggml_backend_buffer_t predbuf_ = nullptr;
    std::vector<ggml_tensor *> pred_w_, pred_b_;
    ggml_tensor * pred_w(uint32_t il) const { return il < pred_w_.size() ? pred_w_[il] : nullptr; }
    ggml_tensor * pred_b(uint32_t il) const { return il < pred_b_.size() ? pred_b_[il] : nullptr; }
    bool load_predictor(const std::string & path, std::string & err);
    // Training data for that head (QWFN_ROUTE_DUMP=dir, QWFN_ROUTE_STRIDE=n): during
    // a prefill, per layer L >= 1, the router input the predictor sees (hc_mix_ffn
    // of the residual entering L, before the PLE) and L's true routing, every
    // n-th token; plus each layer's router matrix once, for the initialisation.
    std::string           dump_dir_; int dump_stride_ = 4; bool routers_dumped_ = false;
    std::vector<FILE *>   dump_f_;
    ggml_tensor *         t_xdump_ = nullptr;
    bool dump_routers(const std::string & dir, std::string & err);
    void dump_layer(uint32_t il, int64_t T);
    // The same records from DECODE (QWFN_ROUTE_DUMP_DECODE=dir): the input the
    // decode graph actually hands the predictor (the residual missing the
    // CPU-served experts) and the next layer's true routing, every token. A
    // head trained on prefill tokens scored 72% at decode against the router's
    // 84%: the distributions differ, so the head must see decode data.
    std::string           dump_dec_dir_; bool routers_dec_dumped_ = false;
    std::vector<FILE *>   dump_dec_f_;
    ggml_tensor *         t_xdec_ = nullptr;             // [n_embd, n_layer]: column L = input predicting L+1
    ggml_tensor *         t_rscale_ = nullptr;           // skip_miss: [1] gate renormalisation of the routed sum
    bool                  deferred_wait_ = false;        // skip_miss: a fetch's reads are in flight, not yet waited for
public:
    // For a stall watchdog: what the expert cache is blocked on (racy reads).
    int    cache_wait_state() const { return ec_.wait_state(); }
    size_t cache_wait_count() const { return ec_.wait_count(); }
private:
    std::vector<int32_t>  tok_sel_; std::vector<float> tok_w_;   // this token's routing, all layers
    void dump_decode_token();
    bool build_moe_gpu_graph(uint32_t il);

    ggml_context *        wctx_ = nullptr;   ggml_backend_buffer_t wbuf_ = nullptr;
    // The work tensors above are sized for decode and the short-prompt batch.
    // A streamed prefill needs them at n_batch: that set lives only while one
    // runs (prefill_enter / prefill_leave), in the VRAM the expert tier's
    // dynamic buffer gives up for the duration. The decode pointers are saved
    // here while the prefill set is swapped in.
    struct work_set {
        ggml_tensor * res[2] = {nullptr, nullptr};
        ggml_tensor * cur = nullptr, * emb = nullptr, * sh = nullptr, * pg = nullptr, * pc = nullptr, * ple = nullptr;
        ggml_tensor * inject = nullptr, * sel = nullptr, * w = nullptr, * tok = nullptr, * pos = nullptr, * plei = nullptr;
        ggml_tensor * xdump = nullptr;
    };
    work_set              dec_ws_;
    ggml_context *        pwctx_ = nullptr;  ggml_backend_buffer_t pwbuf_ = nullptr;
    ggml_gallocr_t        galloc_pf_dyn_ = nullptr;
    bool                  in_prefill_ = false;
    uint64_t              tier_epoch_seen_ = 0;
    bool prefill_enter(std::string & err);
    void prefill_leave();
    void save_work_set(work_set & ws) const;
    void load_work_set(const work_set & ws);
    ggml_tensor *         h_tok_ = nullptr, * h_emb_ = nullptr;   // host: token ids, gathered embeddings
    ggml_context *        hctx_ = nullptr;   ggml_backend_buffer_t hbuf_ = nullptr;
    // Device twins of h_cur_/h_partial_, used when the prefill MoE runs on the
    // GPU. The host tensors stay: the surrounding prefill bookkeeping reads
    // them, and a graph must not mix buffers across devices.
    ggml_context *        dctx_ = nullptr;   ggml_backend_buffer_t dbuf_ = nullptr;
    ggml_tensor *         d_cur_ = nullptr, * d_partial_ = nullptr;
    ggml_gallocr_t        galloc_pf_ = nullptr;

    // VRAM scratch for the cache-served batched prefill: cache_batch_chunk
    // slots per part, sized for the largest layer, viewed per layer at that
    // layer's natural slice stride.
    ggml_backend_buffer_t scr_buf_  = nullptr;
    ggml_context *        scr_ctx_  = nullptr;
    ggml_tensor *         scr_xfer_ = nullptr;
    uint8_t *             scr_base_ = nullptr;
    size_t                scr_part_off_[EXPERT_NPARTS] = {0, 0, 0};
    uint32_t              scr_slots_ = 0;

    // Absolute position -> row in ov_ (n_embd floats each).
    std::vector<int32_t>  ov_pos_;
    std::vector<float>    ov_;

    // persistent, device side
    ggml_tensor * res_[2] = {nullptr, nullptr};
    ggml_tensor * t_cur_ = nullptr, * t_inject_ = nullptr, * t_sel_ = nullptr, * t_w_ = nullptr;
    ggml_tensor * t_emb_ = nullptr;   // token embeddings, before the hc repeat
    ggml_tensor * t_selnext_ = nullptr, * t_selnext2_ = nullptr;
    ggml_tensor * t_specscore_ = nullptr;   // logits of the predicted candidates, best first
    // Decode readback pack: one F32 tensor aliasing the first rows of t_cur_ that
    // holds, in order, this layer's FFN input (n_embd), the gate weights (U), the
    // expert ids as floats (U), the predicted ids as floats (K) and their logits
    // (K), so the host reads everything graph A produced in ONE device sync
    // instead of five. Exact: ids below 2^24 convert to F32 and back exactly.
    ggml_tensor *        t_pack_ = nullptr;
    int64_t              pack_n_ = 0;
    std::vector<float>   pack_host_;
    std::vector<int32_t> pred_next_;          // unpacked prediction, consumed by issue_prefetch
    std::vector<float>   scores_next_;
    std::vector<uint8_t> gA_pack_;            // per layer: the cached graph writes the pack
    ggml_tensor *        p_pc_ = nullptr;     // pinned staging for the CPU partial's async upload
    ggml_tensor *        t_hcmean_ = nullptr; // F32 [hc]: 1/hc each, for the fused stream mean (GPU graphs)
    // MTP draft head (experiment).
    model_index   mi_mtp_;
    weights       wm_;                        // the nextn block, experts included, resident
    state         st_mtp_;                    // its KV: one attention layer at n_ctx
    hparams       hpm_;                       // the MTP file's hparams, the block typed as attention
    bool          mtp_on_ = false, mtp_have_h_ = false, mtp_kv_valid_ = true;
    ggml_tensor * t_hlast_ = nullptr;         // F32 [n_embd, hc, Bd]: the wide residual after the last layer, per position of the last eval
    ggml_tensor * t_mtp_pos_ = nullptr;       // I32 [4*Bd]: the draft's positions
    int64_t       mtp_h_rows_ = 0;            // rows of t_hlast_ the last eval filled
    int32_t       mtp_draft_ = -1, mtp_draft_top_[3] = { -1, -1, -1 };
    // Run the head for n positions starting at `pos`, reading rows h_row.. of
    // t_hlast_ and e_row.. of t_emb_; `actual` (may be null) are the tokens at
    // positions pos+2.. for scoring, n_actual of them.
    bool mtp_draft(int64_t pos, int64_t n, int64_t h_row, int64_t e_row, const int32_t * actual, int64_t n_actual, std::string & err);
    ggml_tensor * t_sh_ = nullptr, * t_pg_ = nullptr, * t_pc_ = nullptr, * t_ple_ = nullptr;
    ggml_tensor * inp_tok_ = nullptr, * inp_pos_ = nullptr, * inp_ple_ = nullptr;
    // persistent, host side
    ggml_tensor * h_cur_ = nullptr, * h_partial_ = nullptr, * h_ple_ = nullptr, * h_ple_idx_ = nullptr;

    int32_t  n_past_  = 0;
    int64_t  n_vocab_ = 0;
    bool     have_expert_map_ = false;

    std::vector<float>   logits_, xfer_, zeros_;
    std::vector<int32_t> sel_, ids_;
    std::vector<float>   wgt_;
    std::vector<int32_t> pred_;   // last layer's prediction for this one
    std::vector<float>   spec_scores_, pred_margin_;   // their logits and margins to the cut-off
    std::vector<uint8_t> spec_block_mask_;             // by predicted layer
    // Two-ahead predictions in flight: pred2_a_ was made two layers back (and is
    // scored against this layer), pred2_b_ one layer back.
    std::vector<int32_t> pred2_a_, pred2_b_;
};

} // namespace qwfn
