#include "speculate.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tierlm {

Speculator::Speculator(Model& m, int n_draft, float p_min) : m_(m), n_draft_(std::max(0, n_draft)), p_min_(p_min) {}

std::vector<float> Speculator::prompt(const std::vector<int32_t>& toks, int n_threads, std::vector<int32_t>* routing) {
    if (routing) routing->clear();
    const int E = m_.hparams().hidden_width(), end = (int)toks.size();
    if (end == 0) return {};
    const int n0 = m_.n_past();                       // tokens [0, n0) are in the cache and unchanged
    if (m_.has_mtp() && n0 > 0 && h_pos_ != n0 - 1 && ck_.h_pos == n0 - 1 && !ck_.h_last.empty()) {
        h_last_ = ck_.h_last; h_pos_ = ck_.h_pos;    // the model went back to its checkpoint: so do we
    }
    // The draft block's entry at position p needs token p and the model's hidden state at p - 1. Entries before
    // c are still right. To redo the rest we need the hidden state at c - 1: kept from last time, or recomputed.
    const int c = std::min(mtp_valid_, n0);
    // (Without a draft block nothing needs the hidden states: the cache is reused as it is.)
    const int s_req = !m_.has_mtp() ? n0 : c == 0 ? 0 : (h_pos_ == c - 1 ? c : c - 1);
    if (n0 > s_req) m_.truncate(s_req);               // (hybrid models start over from 0 instead)
    int s = m_.n_past();
    if (s >= end) { m_.truncate(end - 1); s = m_.n_past(); }   // always evaluate at least one token

    std::vector<float> H, lg;                         // hidden states of positions [s, end)
    for (int off = s; off < end; off += m_.batch()) {
        const int k = std::min(m_.batch(), end - off);
        std::vector<int32_t> part;
        Model::EvalOut r = m_.eval_ex(toks.data() + off, k, n_threads, 1, true, 1, routing ? &part : nullptr);
        if (routing) routing->insert(routing->end(), part.begin(), part.end());
        H.insert(H.end(), r.hidden.begin(), r.hidden.end());
        lg = std::move(r.logits);
    }
    if (m_.has_mtp()) {
        const int c0 = c;                             // positions [c, end) need (re)doing in the draft block; the
                                                      // hidden state for c - 1 is in H (s <= c - 1) or kept (h_pos_)
        std::vector<float> hrows;
        for (int p0 = c0; p0 < end; p0 += m_.batch()) {
            const int k = std::min(m_.batch(), end - p0);
            hrows.assign((size_t)k * E, 0.0f);
            for (int i = 0; i < k; ++i) {
                const int q = p0 + i - 1;             // the hidden state paired with token p0 + i
                const float* src = q < 0 ? nullptr : q >= s ? H.data() + (size_t)(q - s) * E
                                                     : q == h_pos_ ? h_last_.data() : nullptr;
                if (q >= 0 && !src) throw std::runtime_error("speculative decoding: missing hidden state");
                if (src) std::copy(src, src + E, hrows.begin() + (size_t)i * E);
            }
            m_.mtp_eval(toks.data() + p0, hrows.data(), k, p0, n_threads);
        }
    }
    h_last_.assign(H.end() - E, H.end());
    h_pos_ = end - 1;
    mtp_valid_ = end;
    hist_ = toks;
    look_len_ = lookup_max_;                          // each reply starts with full-length guesses
    look_s_ = 0; look_tok_ = look_n_ = pause_ = 0; pause_len_ = 16;   // and lookup on (t_plain_ stays: same model)
    return lg;
}

// The most recent earlier place where the last `len` tokens of ctx appear (len from longest to lookup_min_), and the
// tokens that followed there.
std::vector<int32_t> Speculator::lookup_guesses(const std::vector<int32_t>& ctx, int n) const {
    std::vector<int32_t> out;
    const int N = (int)ctx.size();
    if (n <= 0 || N < lookup_min_ + 1) return out;
    const int longest = std::min(lookup_min_ + 3, N - 1);
    for (int len = longest; len >= lookup_min_ && out.empty(); --len) {
        const int32_t* tail = ctx.data() + N - len;
        // Candidate starts i: ctx[i, i+len) == tail, with at least one token after it, searched from the end.
        for (int i = N - len - 1; i >= 0; --i) {
            if (ctx[i] != tail[0] || !std::equal(tail, tail + len, ctx.data() + i)) continue;
            for (int j = i + len; j < N && (int)out.size() < n; ++j) out.push_back(ctx[j]);
            break;
        }
    }
    return out;
}

std::vector<int32_t> Speculator::step(int32_t x, int n_threads, const Sampler& sample, const std::function<bool(int32_t)>& stop,
                                      std::vector<int32_t>* routing) {
    const int P = m_.n_past(), E = m_.hparams().hidden_width(), V = m_.hparams().n_vocab;
    const int room = m_.max_ctx() - P - 1;            // positions left after x
    const int n_max = std::max(0, std::min(n_draft_, room));
    const auto t0 = std::chrono::steady_clock::now();

    // ---- 1. guess ----
    std::vector<int32_t> drafts;
    if (hist_.size() != (size_t)P) hist_.clear();     // out of step with the cache (it was changed elsewhere): no lookup
    // Prompt lookup's guesses for after x (none if the conversation has no match).
    std::vector<int32_t> look;
    const int n_look = std::max(0, std::min(std::min(look_len_, lookup_max_), room));
    const bool paused = pause_ > 0;
    if (paused) --pause_;
    if (!draft_src_ && n_look > 0 && !hist_.empty() && !paused) {
        hist_.push_back(x);
        look = lookup_guesses(hist_, n_look);
        hist_.pop_back();
        for (size_t i = 0; i < look.size(); ++i)
            if (stop(look[i])) { look.resize(i + 1); break; }
    }
    if (draft_src_) {
        drafts = draft_src_(P, x, n_max);
        if ((int)drafts.size() > n_max) drafts.resize(n_max);
    } else if (m_.has_mtp() && h_pos_ == P - 1 && n_max > 0) {
        int32_t tok = x;
        std::vector<float> h = h_last_, hn;
        for (int i = 0; i < n_max; ++i) {
            std::vector<float> lg = m_.mtp_eval(&tok, h.data(), 1, P + i, n_threads, &hn);
            const int32_t d = (int32_t)(std::max_element(lg.begin(), lg.end()) - lg.begin());
            if (p_min_ > 0.0f) {                      // how sure is the draft block? (softmax probability of d)
                double sum = 0.0;
                for (float v : lg) sum += std::exp((double)v - lg[d]);
                if (1.0 / sum < p_min_) break;
            }
            drafts.push_back(d);
            if (stop(d)) break;
            tok = d;
            h.swap(hn);
        }
    }
    // With the draft block's guesses, lookup only extends them: when lookup's guess starts with all of the draft
    // block's, two independent guessers agree and the longer guess is worth checking; when they differ, the draft block (usually the better
    // guesser) is trusted. Without a draft block, lookup's guesses are the only ones.
    int from_lookup = 0;
    if (!look.empty()) {
        const bool agree = drafts.empty() ? !m_.has_mtp() || h_pos_ != P - 1 || n_max == 0
                                          : look.size() > drafts.size() && std::equal(drafts.begin(), drafts.end(), look.begin());
        if (agree) { drafts = look; from_lookup = (int)look.size(); }
    }
    const int k = (int)drafts.size();
    using sclk = std::chrono::steady_clock;
    const auto t1 = sclk::now();

    // ---- 2. check all guesses in one pass ----
    std::vector<int32_t> batch(1, x);
    batch.insert(batch.end(), drafts.begin(), drafts.end());
    Model::EvalOut out = m_.eval_ex(batch.data(), k + 1, n_threads, k + 1, true, k > 0 ? k + 1 : 1, routing);

    // ---- 3. keep the guesses the model agrees with, plus the model's own next token ----
    std::vector<int32_t> produced, accepted;
    for (int i = 0; i <= k; ++i) {
        const int32_t t = sample(out.logits.data() + (size_t)i * V, accepted);
        produced.push_back(t);
        if (i < k && t == drafts[i] && !stop(t)) { accepted.push_back(t); continue; }
        break;
    }
    const int m = (int)accepted.size();
    m_.accept(1 + m);                                 // x and the m accepted guesses stay; the rest are undone
    if (routing && !routing->empty()) routing->resize(routing->size() / (size_t)(k + 1) * (size_t)(1 + m));

    const auto t2 = sclk::now();
    // ---- 4. bring the draft block up to date with what was kept ----
    if (m_.has_mtp()) {
        std::vector<float> hrows((size_t)(m + 1) * E);
        if (h_pos_ == P - 1) std::copy(h_last_.begin(), h_last_.end(), hrows.begin());
        std::copy(out.hidden.begin(), out.hidden.begin() + (size_t)m * E, hrows.begin() + E);
        m_.mtp_eval(batch.data(), hrows.data(), m + 1, P, n_threads);
    }
    h_last_.assign(out.hidden.begin() + (size_t)m * E, out.hidden.begin() + (size_t)(m + 1) * E);
    h_pos_ = P + m;
    mtp_valid_ = P + m + 1;
    const auto t3 = sclk::now();
    st_.guess_s += std::chrono::duration<double>(t1 - t0).count();
    st_.check_s += std::chrono::duration<double>(t2 - t1).count();
    st_.catch_s += std::chrono::duration<double>(t3 - t2).count();
    st_.steps++; st_.drafted += k; st_.accepted += m;
    if (from_lookup > 0) {
        st_.looked += k; st_.look_ok += m;
        // A match that led nowhere costs a wider check for nothing: guess just one token after a miss (cheap to
        // check), the number that were right plus one after a partial hit, and twice as many after a full one.
        look_len_ = m == k ? std::min(lookup_max_, std::max(1, k) * 2) : std::max(1, m + 1);
    }
    // ---- 5. is lookup paying for itself? ----
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (k == 0) t_plain_ = t_plain_ > 0 ? 0.9 * t_plain_ + 0.1 * dt : dt;
    else if (from_lookup > 0) {
        look_s_ += dt; look_tok_ += m + 1; look_n_++;
        if (look_n_ >= 4 && t_plain_ > 0) {           // a round of 4 checks: their tokens/s against plain steps'
            if (look_tok_ / look_s_ < 1.0 / t_plain_) { pause_ = pause_len_; pause_len_ = std::min(pause_len_ * 2, 1024); }
            else pause_len_ = 16;
            look_s_ = 0; look_tok_ = look_n_ = 0;
        }
    }
    if (!hist_.empty() || P == 0) { hist_.push_back(x); hist_.insert(hist_.end(), accepted.begin(), accepted.end()); }
    return produced;
}

}
