#include "speculate.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tierlm {

Speculator::Speculator(Model& m, int n_draft, float p_min) : m_(m), n_draft_(std::max(0, n_draft)), p_min_(p_min) {}

std::vector<float> Speculator::prompt(const std::vector<int32_t>& toks, int n_threads) {
    const int E = m_.hparams().n_embd, end = (int)toks.size();
    if (end == 0) return {};
    const int n0 = m_.n_past();                       // tokens [0, n0) are in the cache and unchanged
    // The draft block's entry at position p needs token p and the model's hidden state at p - 1. Entries before
    // c are still right. To redo the rest we need the hidden state at c - 1: kept from last time, or recomputed.
    const int c = std::min(mtp_valid_, n0);
    const int s_req = c == 0 ? 0 : (h_pos_ == c - 1 ? c : c - 1);
    if (n0 > s_req) m_.truncate(s_req);               // (hybrid models start over from 0 instead)
    int s = m_.n_past();
    if (s >= end) { m_.truncate(end - 1); s = m_.n_past(); }   // always evaluate at least one token

    std::vector<float> H, lg;                         // hidden states of positions [s, end)
    for (int off = s; off < end; off += m_.batch()) {
        const int k = std::min(m_.batch(), end - off);
        Model::EvalOut r = m_.eval_ex(toks.data() + off, k, n_threads, 1, true, 1);
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
    return lg;
}

std::vector<int32_t> Speculator::step(int32_t x, int n_threads, const Sampler& sample, const std::function<bool(int32_t)>& stop) {
    const int P = m_.n_past(), E = m_.hparams().n_embd, V = m_.hparams().n_vocab;
    const int room = m_.max_ctx() - P - 1;            // positions left after x
    const int n_max = std::max(0, std::min(n_draft_, room));

    // ---- 1. guess ----
    std::vector<int32_t> drafts;
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
    const int k = (int)drafts.size();

    // ---- 2. check all guesses in one pass ----
    std::vector<int32_t> batch(1, x);
    batch.insert(batch.end(), drafts.begin(), drafts.end());
    Model::EvalOut out = m_.eval_ex(batch.data(), k + 1, n_threads, k + 1, true, k > 0 ? k + 1 : 1);

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
    st_.steps++; st_.drafted += k; st_.accepted += m;
    return produced;
}

}
