#pragma once
// Speculative decoding with a model's own MTP draft block (multi-token prediction).
//
// Each step: the small draft block guesses the next few tokens; the model then checks all of them in ONE pass,
// reading its weights once for several tokens instead of once per token. Guesses the model agrees with are free
// tokens; at the first disagreement the model's own choice is taken and the rest are undone (Model::accept).
// Every token that comes out is the one the model itself picks at that position, so with temperature 0 the reply
// is the same as without speculation (up to floating-point rounding), and with sampling it follows the same
// distribution.
#include "model.h"
#include <functional>
#include <vector>

namespace tierlm {

struct SpecStats {
    int steps = 0;         // checks run
    int drafted = 0;       // tokens guessed by the draft block
    int accepted = 0;      // guesses the model agreed with
};

class Speculator {
public:
    // n_draft: most tokens to guess per step. p_min: stop guessing once the draft block is less sure than this.
    explicit Speculator(Model& m, int n_draft = 3, float p_min = 0.0f);

    // Like Model::eval for a whole conversation: toks is the full token list, of which the first m.n_past() are
    // already in the model's cache (the caller truncates the cache to the part that still matches). Evaluates the
    // rest, keeping the hidden states the draft block needs, and brings the draft block up to date.
    // Returns the prediction after the last token. `routing` (MoE models) receives the experts the router picked
    // for every token evaluated, as Model::eval gives them.
    std::vector<float> prompt(const std::vector<int32_t>& toks, int n_threads, std::vector<int32_t>* routing = nullptr);

    // One step after token x, which was just chosen and is not yet in the cache. sample(row, accepted) picks the
    // model's token from a row of logits, given the tokens accepted so far in this step (after x). stop(t) is true
    // for a token that ends the reply. Returns the tokens that follow x: the accepted guesses plus one more. Like x,
    // the last one is not yet in the cache. `routing` (MoE models) receives the experts picked for the tokens that
    // were kept (x and the accepted guesses), n_kept x n_layer x n_expert_used ids.
    using Sampler = std::function<int32_t(const float* logits, const std::vector<int32_t>& accepted)>;
    std::vector<int32_t> step(int32_t x, int n_threads, const Sampler& sample, const std::function<bool(int32_t)>& stop,
                              std::vector<int32_t>* routing = nullptr);

    // Testing: guesses come from this function instead of the draft block (pos = position of x).
    using DraftSource = std::function<std::vector<int32_t>(int pos, int32_t x, int n)>;
    void set_draft_source(DraftSource f) { draft_src_ = std::move(f); }

    const SpecStats& stats() const { return st_; }
    void reset_stats() { st_ = SpecStats(); }

private:
    Model& m_;
    int n_draft_;
    float p_min_;
    std::vector<float> h_last_;   // the model's final hidden state at position h_pos_
    int h_pos_ = -1;
    int mtp_valid_ = 0;           // the draft block's cache is right for positions [0, mtp_valid_)
    SpecStats st_;
    DraftSource draft_src_;
};

}
