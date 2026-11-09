#pragma once
// Speculative decoding with a model's own MTP draft block (multi-token prediction), and prompt lookup.
//
// Each step: the small draft block guesses the next few tokens; the model then checks all of them in ONE pass,
// reading its weights once for several tokens instead of once per token. Guesses the model agrees with are free
// tokens; at the first disagreement the model's own choice is taken and the rest are undone (Model::accept).
// Every token that comes out is the one the model itself picks at that position, so with temperature 0 the reply
// is the same as without speculation (up to floating-point rounding), and with sampling it follows the same
// distribution.
//
// Prompt lookup: when the last few tokens also appear earlier in the conversation (a file being edited, code being
// quoted, a name being repeated), the tokens that followed them there are a good guess for what comes next. Those
// guesses are checked the same way, so they too only change how fast the reply comes, never what it says. It needs
// no draft block, so it works for every model; with a draft block, lookup guesses are used when there is a match
// and the draft block's otherwise. Lookup times its own checks and pauses while they are slower than plain steps.
#include "model.h"
#include <algorithm>
#include <chrono>
#include <functional>
#include <random>
#include <vector>

namespace tierlm {

struct SpecStats {
    int steps = 0;         // checks run
    int drafted = 0;       // tokens guessed by the draft block
    int accepted = 0;      // guesses the model agreed with
    int looked = 0;        // of the guesses, those from prompt lookup
    int look_ok = 0;       // prompt-lookup guesses the model agreed with
    double guess_s = 0, check_s = 0, catch_s = 0;   // time guessing, checking, and bringing the draft block up to date
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

    // Sampling (temperature > 0): dist(row, context) gives the sampler's probabilities for a row of logits (the
    // tokens it could pick, after penalties, temperature and top-k/top-p/min-p), where context is the tokens chosen
    // since x (x first). With it, the draft block's guesses are drawn from its own probabilities q and a guess d is
    // kept with probability min(1, p(d)/q(d)); after one isn't, the token is drawn from what p has left over q
    // (speculative sampling): the reply follows exactly the model's own distribution, and far more guesses are kept
    // than by checking the model's own draw against the draft block's single best guess. Unset: guesses are the
    // draft block's best, kept when the model's own pick (sample) agrees.
    using Dist = std::function<std::vector<std::pair<int32_t, float>>(const float* logits, const std::vector<int32_t>& context)>;
    void set_sampling(Dist dist, std::mt19937* rng) { dist_ = std::move(dist); rng_ = rng; }
    void clear_sampling() { dist_ = nullptr; rng_ = nullptr; }

    // Testing: guesses come from this function instead of the draft block (pos = position of x).
    using DraftSource = std::function<std::vector<int32_t>(int pos, int32_t x, int n)>;
    void set_draft_source(DraftSource f) { draft_src_ = std::move(f); }

    // Prompt lookup: guess up to n tokens (0 = off) from an earlier place in the conversation where the last
    // `min_match` or more tokens also appear.
    void set_lookup(int n, int min_match = 3) { lookup_max_ = std::max(0, n); lookup_min_ = std::max(1, min_match); look_len_ = lookup_max_; }
    int lookup() const { return lookup_max_; }
    // The guesses prompt lookup makes after context `ctx` (exposed for testing).
    std::vector<int32_t> lookup_guesses(const std::vector<int32_t>& ctx, int n) const;

    // The draft block's view of the conversation, saved and restored with the model's (Model::Snapshot).
    struct State { std::vector<float> h_last; int h_pos = -1, mtp_valid = 0; };
    State state() const { return {h_last_, h_pos_, mtp_valid_}; }
    void set_state(const State& s) { h_last_ = s.h_last; h_pos_ = s.h_pos; mtp_valid_ = s.mtp_valid; }
    // Remember the hidden state at the model's checkpoint (Model::checkpoint), so a prompt that goes back there
    // can carry on the draft block without recomputing from the start.
    void mark() { ck_ = state(); }

    const SpecStats& stats() const { return st_; }
    void reset_stats() { st_ = SpecStats(); }

private:
    Model& m_;
    int n_draft_;
    float p_min_;
    int32_t draw(const std::vector<std::pair<int32_t, float>>& w);
    static float prob_of(const std::vector<std::pair<int32_t, float>>& w, int32_t t);
    Dist dist_;
    std::mt19937* rng_ = nullptr;
    std::vector<float> h_last_;   // the model's final hidden state at position h_pos_
    int h_pos_ = -1;
    int mtp_valid_ = 0;           // the draft block's cache is right for positions [0, mtp_valid_)
    State ck_;                    // the state at the model's checkpoint (mark)
    SpecStats st_;
    DraftSource draft_src_;
    int lookup_max_ = 0, lookup_min_ = 3;
    int look_len_ = 0;            // guesses to make next time: shrinks after misses, grows back after hits
    std::vector<int32_t> hist_;   // the tokens in the model's cache, for prompt lookup
    // Prompt lookup pays for itself only when its checks give more tokens per second than plain steps: a guessed
    // token costs real time to check (on MoE models it reads its own experts), and a miss gets nothing for it. So
    // it times both and pauses when guessing is losing, trying again later (twice as late each time it loses).
    double t_plain_ = 0;          // seconds per step without guesses (running average)
    double look_s_ = 0;           // this round of lookup checks: seconds, tokens produced, checks
    int look_tok_ = 0, look_n_ = 0;
    int pause_ = 0, pause_len_ = 16;   // steps left without lookup; how long the next pause is
};

}
