#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace tierlm {

struct OlmoeHparams {
    int n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0, n_ff = 0;
    int n_expert = 0, n_expert_used = 0, n_vocab = 0, n_ctx = 0;
    float rope_base = 10000.0f, rms_eps = 1e-5f;
};

// OLMoE forward pass on the CPU. First version: every weight is loaded into RAM,
// and each call recomputes the whole sequence (no KV cache yet).
class OlmoeModel {
public:
    static std::unique_ptr<OlmoeModel> load(const std::string& path);   // throws on failure
    ~OlmoeModel();
    OlmoeModel(const OlmoeModel&) = delete;
    OlmoeModel& operator=(const OlmoeModel&) = delete;

    // Logits for the last position of `ids`.
    std::vector<float> forward(const std::vector<int32_t>& ids, int n_threads) const;
    const OlmoeHparams& hparams() const { return hp_; }

private:
    OlmoeModel() = default;
    struct Layer {
        ggml_tensor *attn_norm, *wq, *wk, *wv, *wo, *q_norm, *k_norm;
        ggml_tensor *ffn_norm, *router, *up, *gate, *down;
    };
    OlmoeHparams hp_;
    ggml_context* wctx_ = nullptr;
    ggml_tensor *tok_embd_ = nullptr, *out_norm_ = nullptr, *output_ = nullptr;
    std::vector<Layer> layers_;
};

}
