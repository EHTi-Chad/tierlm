#include "speculate.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace tierlm {

Speculator::Speculator(Model& m, int n_draft, float p_min) : m_(m), n_draft_(std::max(0, n_draft)), p_min_(p_min) {
    // TIERLM_SPEC_TRACE=file (developers): a line per check (appended; every speculator of the process writes there)
    if (const char* e = std::getenv("TIERLM_SPEC_TRACE"); e && *e) trace_ = std::fopen(e, "a");
}
Speculator::~Speculator() { if (trace_) std::fclose(trace_); }

std::vector<float> Speculator::prompt(const std::vector<int32_t>& toks, int n_threads, std::vector<int32_t>* routing) {
    if (routing) routing->clear();
    const int E = m_.hparams().hidden_width(), end = (int)toks.size();
    if (end == 0) return {};
    const int n0 = m_.n_past();                       // tokens [0, n0) are in the cache and unchanged
    if (m_.has_mtp() && n0 > 0 && h_pos_ != n0 - 1)
        for (const State& ck : marks_)
            if (ck.h_pos == n0 - 1 && !ck.h_last.empty()) { h_last_ = ck.h_last; h_pos_ = ck.h_pos; break; }   // the model went back to a checkpoint: so do we
    // The draft block's entry at position p needs token p and the model's hidden state at p - 1. Entries before
    // c are still right. To redo the rest we need the hidden state at c - 1: kept from last time, or recomputed.
    const int c = std::min(mtp_valid_, n0);
    // (Without a draft block nothing needs the hidden states: the cache is reused as it is.)
    const int s_req = !m_.has_mtp() ? n0 : c == 0 ? 0 : (h_pos_ == c - 1 ? c : c - 1);
    if (n0 > s_req) m_.truncate(s_req);               // (hybrid models start over from 0 instead)
    int s = m_.n_past();
    if (s >= end) { m_.truncate(end - 1); s = m_.n_past(); }   // always evaluate at least one token

    std::vector<float> H, lg;                         // hidden states of positions [s, end)
    for (int off = s; off < end; ) {
        const int k = std::min(m_.batch_at(off), end - off);       // (fewer at a time far into a long context)
        std::vector<int32_t> part;
        Model::EvalOut r = m_.eval_ex(toks.data() + off, k, n_threads, 1, true, 1, routing ? &part : nullptr);
        if (routing) routing->insert(routing->end(), part.begin(), part.end());
        H.insert(H.end(), r.hidden.begin(), r.hidden.end());
        lg = std::move(r.logits);
        off += k;
    }
    if (m_.has_mtp()) {
        const int c0 = c;                             // positions [c, end) need (re)doing in the draft block; the
                                                      // hidden state for c - 1 is in H (s <= c - 1) or kept (h_pos_)
        std::vector<float> hrows;
        for (int p0 = c0, k = 0; p0 < end; p0 += k) {
            k = std::min(m_.batch_at(p0), end - p0);
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
            for (int j = i + len; j < N && (int)out.size() < n && !Model::is_image_id(ctx[j]); ++j) out.push_back(ctx[j]);   // (never an image's words)
            break;
        }
    }
    return out;
}

// A token drawn from (token, weight) pairs (weights needn't add up to 1).
int32_t Speculator::draw(const std::vector<std::pair<int32_t, float>>& w) {
    double total = 0.0;
    for (const auto& e : w) total += e.second;
    std::uniform_real_distribution<double> u(0.0, total);
    double r = u(*rng_);
    for (const auto& e : w) { r -= e.second; if (r <= 0.0) return e.first; }
    return w.back().first;
}

float Speculator::prob_of(const std::vector<std::pair<int32_t, float>>& w, int32_t t) {
    for (const auto& e : w) if (e.first == t) return e.second;
    return 0.0f;
}

bool Speculator::self_draft_ok() const {
    return (m_.has_split_experts() || m_.hparams().looped()) && m_.draft_ok();
}

// Guessing with the model itself: what's cut. Experts split between VRAM and RAM: only the VRAM ones (set_hot_only).
// A looped model: one pass of its loop instead of all (set_loop_draft). Both when both apply.
// TIERLM_LOOP_DRAFT=n (tests): n passes; TIERLM_LOOP_DRAFT_HOT=0: a looped model's guesses with all its experts.
static void self_cut(Model& m, bool on) {
    static const int passes = [] { const char* e = std::getenv("TIERLM_LOOP_DRAFT"); return e ? std::max(1, std::atoi(e)) : 1; }();
    static const bool loop_hot = [] { const char* e = std::getenv("TIERLM_LOOP_DRAFT_HOT"); return !e || std::atoi(e) != 0; }();
    const bool looped = m.hparams().looped();
    m.set_hot_only(on && m.has_split_experts() && (!looped || loop_hot));
    m.set_loop_draft(on && looped ? passes : 0);
}

std::vector<int32_t> Speculator::step(int32_t x, int n_threads, const Sampler& sample, const std::function<bool(int32_t)>& stop,
                                      std::vector<int32_t>* routing) {
    const int P = m_.n_past(), E = m_.hparams().hidden_width(), V = m_.hparams().n_vocab;
    const int room = m_.max_ctx() - P - 1;            // positions left after x
    // (adaptive: see set_adaptive; a guess needs room in the check the model's working memory holds)
    const bool adapt = adapt_ && m_.has_mtp() && !draft_src_ && n_draft_ > 0;
    // Two sizes of check, the set number of guesses or many: a check of another size than the last has to be laid
    // out again, which costs as much as one or two tokens (measured: 3 to 13 ms), so the size changes only when it
    // clearly pays. (Deciding every count freely was measured too: the size changed at more than half the checks, and
    // that took back what the longer guesses gained.) Until a few checks have been timed: the set number.
    const int n_many = std::max(n_draft_, std::min(adapt_most_, m_.check_width() - 1));   // (never fewer than were set)
    const int n_few = n_draft_;
    const int n_max = std::max(0, std::min(adapt && ad_n_ >= 8 ? n_many : n_draft_, room));
    // What one more token must bring in to be worth its place in the check: the reply's speed times the token's cost.
    double theta = 0.5;
    if (adapt && ad_ms_ > 0 && ad_n_ >= 8) {
        const double n = ad_n_, mw = ad_w_ / n, mt = ad_t_ / n, var = ad_ww_ / n - mw * mw, cov = ad_wt_ / n - mw * mt;
        double per = var > 0.3 ? cov / var : mt / (mw + 4.0);          // ms per token of a check (few widths seen: a guess at it)
        per = std::min(std::max(per, mt / (mw + 12.0)), mt / (mw + 0.5));
        theta = std::min(0.9, std::max(0.2, ad_tok_ / ad_ms_ * (per + ad_guess_ms_)));
        static const double fixed = [] { const char* e = std::getenv("TIERLM_DRAFT_THETA"); return e ? std::atof(e) : -1.0; }();   // (tests)
        if (fixed >= 0) theta = fixed;
    }
    // (how much of its sureness the draft block earns: of the guesses tested lately, kept over expected. Above 1 too:
    // Flash-Next's draft block on code is surer of less than it gets right)
    const double trust = ad_pred_ > 4.0 ? std::min(1.3, std::max(0.5, ad_kept_ / ad_pred_)) : 0.95;
    // (the last check had many guesses: this one keeps that size unless its guesses are far from paying)
    const double bar = adapt && last_many_ ? theta * 0.6 : theta;
    std::vector<float> conf;                          // how sure the draft block was of each guess made
    double chance = 1.0;                              // that every guess so far is kept
    const auto t0 = std::chrono::steady_clock::now();

    // ---- 1. guess ----
    trace_q_.clear();
    std::vector<int32_t> drafts;
    std::vector<std::vector<std::pair<int32_t, float>>> qs;   // sampling: the probabilities each guess was drawn from
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
    const bool self_used = self_ && !draft_src_ && !m_.has_mtp() && self_draft_ok();
    if (draft_src_) {
        drafts = draft_src_(P, x, n_max);
        if ((int)drafts.size() > n_max) drafts.resize(n_max);
    } else if (((m_.has_mtp() && h_pos_ == P - 1) || self_used) && n_max > 0) {
        int32_t tok = x;
        std::vector<float> h = h_last_, hn;
        if (self_used) self_cut(m_, true);                // (first: the mark copies only what the cut model writes)
        const bool marked = self_used && m_.draft_mark();
        if (self_used && !marked) self_cut(m_, false);
        const int lim = self_used && !marked ? 0 : n_max;
        for (int i = 0; i < lim; ++i) {
            std::vector<float> lg = self_used ? m_.eval_ex(&tok, 1, n_threads, 1, false, 1).logits
                                              : m_.mtp_eval(&tok, h.data(), 1, P + i, n_threads, &hn);
            int32_t d;
            float sure = 1.0f;                        // the draft block's probability for its best token
            if (dist_) {                              // sampling: draw the guess from the draft block's probabilities
                std::vector<int32_t> ctx(1, x);
                ctx.insert(ctx.end(), drafts.begin(), drafts.end());
                std::vector<std::pair<int32_t, float>> q = dist_(lg.data(), ctx);
                if (q.empty()) break;
                float qmax = 0.0f;
                for (const auto& e : q) qmax = std::max(qmax, e.second);
                if (p_min_ > 0.0f && qmax < p_min_) break;
                sure = qmax;
                if (adapt && i >= n_few && chance * trust * sure < bar * 0.5) break;   // (hopeless: see below)
                d = draw(q);
                qs.push_back(std::move(q));
            } else {
                d = (int32_t)(std::max_element(lg.begin(), lg.end()) - lg.begin());
                if (p_min_ > 0.0f || trace_ || adapt) {   // how sure is the draft block? (softmax probability of d)
                    const float top = lg[d];
                    double sum = 0.0;                 // (tokens e^-20 below the best add nothing that matters)
                    for (float v : lg) if (v > top - 20.0f) sum += std::exp((double)(v - top));
                    sure = (float)(1.0 / sum);
                    if (sure < p_min_) break;
                    if (adapt && i >= n_few && chance * trust * sure < bar * 0.5) break;   // (hopeless: see below)
                }
            }
            if (trace_) trace_q_.push_back(dist_ ? prob_of(qs.back(), d) : sure);
            conf.push_back(sure);
            chance *= trust * sure;
            drafts.push_back(d);
            if (stop(d)) break;
            tok = d;
            h.swap(hn);
        }
        if (marked) { m_.draft_rewind(); self_cut(m_, false); }      // back to x's place (the check writes it all again)
        // (adaptive, timed: the many when, together, the guesses past the few are expected to bring in more than
        // their place in the check costs, else the few. Each one on its own having to pay was too strict: a stretch
        // of sure guesses ending in a less sure one was cut to the few, and on Qwen3.8 Flash-Next a load could stay
        // with the few all along, 99.5 tok/s where the many made 126.)
        if (adapt && ad_n_ >= 8 && n_max == n_many && (int)drafts.size() > n_few &&
            !(std::find_if(drafts.begin(), drafts.end(), stop) != drafts.end())) {
            double c = 1.0, extra = 0.0;
            for (size_t i = 0; i < conf.size(); ++i) { c *= trust * conf[i]; if ((int)i >= n_few) extra += std::min(1.0, c); }
            if ((int)drafts.size() < n_many || extra < bar * (n_many - n_few)) {
                drafts.resize(n_few); conf.resize(n_few);
                if (dist_) qs.resize(n_few);
            }
        }
    }
    // With the draft block's guesses, lookup only extends them: when lookup's guess starts with all of the draft
    // block's, two independent guessers agree and the longer guess is worth checking; when they differ, the draft block (usually the better
    // guesser) is trusted. Without a draft block, lookup's guesses are the only ones.
    int from_lookup = 0;
    if (!look.empty()) {
        const bool agree = drafts.empty() ? (!m_.has_mtp() && !self_used) || (!self_used && h_pos_ != P - 1) || n_max == 0
                                          : look.size() > drafts.size() && std::equal(drafts.begin(), drafts.end(), look.begin());
        if (agree) { drafts = look; from_lookup = (int)look.size(); }
    }
    const int k = (int)drafts.size();
    if (adapt) last_many_ = k >= n_many && n_many > n_few;
    // (lookup's guesses, and any without a draw: certain, as far as the guesser goes)
    if (dist_) while ((int)qs.size() < k) qs.push_back({{drafts[qs.size()], 1.0f}});
    using sclk = std::chrono::steady_clock;
    const auto t1 = sclk::now();
    const EvalStats st0 = m_.stats();                 // (the trace: the check's parts)

    // ---- 2. check all guesses in one pass ----
    std::vector<int32_t> batch(1, x);
    batch.insert(batch.end(), drafts.begin(), drafts.end());
    Model::EvalOut out = m_.eval_ex(batch.data(), k + 1, n_threads, k + 1, true, k > 0 ? k + 1 : 1, routing);

    // ---- 3. keep the guesses the model agrees with, plus the model's own next token ----
    std::vector<int32_t> produced, accepted;
    for (int i = 0; i <= k; ++i) {
        const float* row = out.logits.data() + (size_t)i * V;
        if (dist_) {
            std::vector<int32_t> ctx(1, x);
            ctx.insert(ctx.end(), accepted.begin(), accepted.end());
            const std::vector<std::pair<int32_t, float>> p = dist_(row, ctx);
            if (i == k) { produced.push_back(draw(p)); break; }
            const int32_t d = drafts[i];
            const float pd = prob_of(p, d), qd = prob_of(qs[i], d);
            std::uniform_real_distribution<float> u(0.0f, 1.0f);
            if (qd > 0.0f && u(*rng_) * qd < pd) {    // kept with probability min(1, p(d) / q(d))
                produced.push_back(d);
                if (stop(d)) break;
                accepted.push_back(d);
                continue;
            }
            // not kept: draw from what p has over q, max(0, p - q), normalised (p itself if nothing is left)
            std::vector<std::pair<int32_t, float>> r;
            for (const auto& [t, pt] : p) {
                const float v = pt - prob_of(qs[i], t);
                if (v > 0.0f) r.push_back({t, v});
            }
            produced.push_back(draw(r.empty() ? p : r));
            break;
        }
        const int32_t t = sample(row, accepted);
        produced.push_back(t);
        if (i < k && t == drafts[i] && !stop(t)) { accepted.push_back(t); continue; }
        break;
    }
    const int m = (int)accepted.size();
    m_.accept(1 + m);                                 // x and the m accepted guesses stay; the rest are undone
    if (routing && !routing->empty()) routing->resize(routing->size() / (size_t)(k + 1) * (size_t)(1 + m));

    const auto t2 = sclk::now();
    const EvalStats st1 = m_.stats();
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
    if (adapt_) {                                     // what this check tells (recent checks count most)
        const double ms = std::chrono::duration<double, std::milli>(t3 - t0).count(), check_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        const double keep = 0.97, keepw = 0.99;
        ad_tok_ = ad_tok_ * keep + (m + 1); ad_ms_ = ad_ms_ * keep + ms;
        const double w = k + 1;
        ad_n_ = ad_n_ * keepw + 1; ad_w_ = ad_w_ * keepw + w; ad_t_ = ad_t_ * keepw + check_ms; ad_ww_ = ad_ww_ * keepw + w * w; ad_wt_ = ad_wt_ * keepw + w * check_ms;
        if (k > 0) {
            const double each = (std::chrono::duration<double, std::milli>(t1 - t0).count() + std::chrono::duration<double, std::milli>(t3 - t2).count()) / k;
            ad_guess_ms_ = 0.9 * ad_guess_ms_ + 0.1 * each;
        }
        if (from_lookup == 0) {                       // the draft block's guesses that were tested: the kept ones, and the first that wasn't
            ad_pred_ *= 0.995; ad_kept_ *= 0.995;
            for (int i = 0; i < (int)conf.size() && i < k && i <= m; ++i) { ad_pred_ += conf[i]; ad_kept_ += i < m ? 1.0 : 0.0; }
        }
    }
    if (trace_) {                                     // TIERLM_SPEC_TRACE: place, guesses, kept, from lookup, ms guessing / checking / catching up, the check's parts, then how sure each guess was
        std::fprintf(trace_, "%d %d %d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f", P, k, m, from_lookup,
                     std::chrono::duration<double, std::milli>(t1 - t0).count(), std::chrono::duration<double, std::milli>(t2 - t1).count(),
                     std::chrono::duration<double, std::milli>(t3 - t2).count(), st1.build_ms - st0.build_ms, st1.alloc_ms - st0.alloc_ms,
                     st1.input_ms - st0.input_ms, st1.compute_ms - st0.compute_ms, st1.output_ms - st0.output_ms);
        for (size_t i = 0; i < trace_q_.size() && (int)i < k; ++i) std::fprintf(trace_, " %.4f", trace_q_[i]);
        std::fprintf(trace_, "\n");
        std::fflush(trace_);
    }
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
        // A round of 4 checks: their tokens/s against plain steps'. Where a check with guesses costs several plain
        // steps (Inkling 2 Mini on an RTX 5090: 65 ms against 13, its step being thousands of small GPU steps that
        // only a plain step replays in one go), a round losing by 3 times or more is called after 2 checks, and a
        // lost round's pause is as many times longer as it lost by (up to 8): over a 500-token program that was 24
        // lost checks, now about 6.
        const double lost_by = t_plain_ > 0 && look_tok_ > 0 ? (look_s_ / look_tok_) / t_plain_ : 0.0;   // (1: the same speed)
        if (t_plain_ > 0 && (look_n_ >= 4 || (look_n_ >= 2 && lost_by >= 3.0))) {
            if (lost_by > 1.0) {
                pause_ = pause_len_ * std::clamp((int)lost_by, 1, 8);
                pause_len_ = std::min(pause_len_ * 2, 1024);
            } else pause_len_ = 16;
            look_s_ = 0; look_tok_ = look_n_ = 0;
        }
    }
    if (!hist_.empty() || P == 0) { hist_.push_back(x); hist_.insert(hist_.end(), accepted.begin(), accepted.end()); }
    return produced;
}

}
