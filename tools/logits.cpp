// Prints the last-position logits for a list of token ids. Used by the tests.
//   tierlm-logits model.gguf id [id ...]          whole sequence in one call
//   tierlm-logits model.gguf --step id [id ...]   first id alone, then one id per call (exercises the KV cache)
//   tierlm-logits model.gguf --rollback B KEEP id [id ...]
//       speculative undo: the last B+1 ids are a batch of B (evaluated as an undoable check) and one final id. Only
//       the first KEEP of the batch are accepted; the output must equal plain evaluation of the ids without the
//       other B-KEEP.
//   tierlm-logits model.gguf --mtp id [id ...]       the MTP draft block's prediction after the last id (fed with the
//                                                   model's hidden states; position 0 gets zeros)
//   tierlm-logits model.gguf --mtp-step id [id ...]  the same, running the draft block one position at a time
//   tierlm-logits model.gguf --gen N id [id ...]      N tokens of greedy generation after the ids (prints their ids)
//   tierlm-logits model.gguf --gen-spec N id [id ...] the same with speculative decoding (MTP draft block). With
//       TIERLM_DRAFT_SCRIPT=t1,t2,... the guesses come from that list (the expected continuation) instead, with every
//       TIERLM_DRAFT_CORRUPT-th one changed, so that both accepted and rejected guesses are exercised.
#include "model.h"
#include "speculate.h"
#include <sstream>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include "ggml-backend.h"
#include "backends.h"

using namespace tierlm;

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s model.gguf [--step] id [id ...]\n", argv[0]); return 1; }
    try {
        bool step = !std::strcmp(argv[2], "--step");
        const bool rollback = !std::strcmp(argv[2], "--rollback");
        const bool mtp = !std::strcmp(argv[2], "--mtp"), mtp_step = !std::strcmp(argv[2], "--mtp-step");
        const bool gen = !std::strcmp(argv[2], "--gen"), gen_spec = !std::strcmp(argv[2], "--gen-spec");
        // --parallel K: K conversations made from the ids (conversation s: the ids rotated by 3s, s fewer of them),
        // computed together in mixed steps (see below); prints each one's last logits, conversation 0 first.
        const int par = !std::strcmp(argv[2], "--parallel") ? std::atoi(argv[3]) : 0;
        const int n_gen = gen || gen_spec ? std::atoi(argv[3]) : 0;
        int nb = 0, keep = 0, first = step || mtp || mtp_step ? 3 : gen || gen_spec || par ? 4 : 2;
        if (rollback) { nb = std::atoi(argv[3]); keep = std::atoi(argv[4]); first = 5; }
        std::vector<int32_t> ids;
        for (int i = first; i < argc; ++i) ids.push_back(std::atoi(argv[i]));
        // Test hooks: TIERLM_RPC=host:port adds an RPC device (seen as a GPU); TIERLM_VRAM_MB limits the VRAM
        // the tier loader may use, forcing a model to be split between the device and RAM.
        load_backends();
        if (const char* ep = std::getenv("TIERLM_RPC"))
            if (!add_rpc_server(ep)) throw std::runtime_error("TIERLM_RPC needs a build with -DGGML_RPC=ON");
        LoadOptions o;
        o.max_ctx = std::max<int>(512, (int)ids.size() + 8);
        if (const char* v = std::getenv("TIERLM_VRAM_MB")) o.vram_budget_mb = std::atoi(v);
        if (const char* v = std::getenv("TIERLM_SPLIT")) o.split_mode = v;
        if (const char* v = std::getenv("TIERLM_DEVICE")) o.device = v;      // e.g. cpu, CUDA0, Vulkan0
        o.mtp = mtp || mtp_step || (gen_spec && !std::getenv("TIERLM_DRAFT_SCRIPT"));
        o.max_ctx = std::max<int>(o.max_ctx, (int)ids.size() + n_gen + 16);
        if (par) { o.parallel = par + 1; o.max_ctx = std::max<int>(o.max_ctx, (int)ids.size() * par + 64); }
        auto m = Model::load(argv[1], o);
        if (std::getenv("TIERLM_SHOW_PLACEMENT")) {
            const auto& p = m->placement();
            std::fprintf(stderr, "%s: %s, VRAM %.2f MiB, RAM %.2f MiB, KV %d/%d, attention %d/%d, ffn %d/%d layers in VRAM\n",
                         m->device_name().c_str(), p.mode.c_str(), p.gpu_weights / 1048576.0, p.cpu_weights / 1048576.0,
                         p.kv_gpu_layers, p.layers, p.attn_gpu, p.layers, p.ffn_gpu, p.layers);
            if (p.hot_experts > 0) {
                int split = 0;
                for (size_t l = 0; l < p.hot.size(); ++l) split += !p.hot[l].empty() && !p.ffn_vram[l];
                std::fprintf(stderr, "  hot experts in VRAM: %d, layers split between VRAM and RAM: %d\n", p.hot_experts, split);
            }
            if (p.gpus.size() > 1)
                for (const auto& gp : p.gpus)
                    std::fprintf(stderr, "  %s: %.2f MiB weights, %d layers' KV\n", gp.name.c_str(), gp.weights / 1048576.0, gp.layers);
        }
        const int nt = std::getenv("TIERLM_THREADS") ? std::atoi(std::getenv("TIERLM_THREADS")) : std::max(1, (int)std::thread::hardware_concurrency() / 2);

        std::vector<float> lg;
        if (gen || gen_spec) {
            auto argmax = [](const float* v, int n) { return (int32_t)(std::max_element(v, v + n) - v); };
            const int V = m->hparams().n_vocab;
            std::vector<int32_t> out;
            if (gen) {
                lg = m->eval(ids, nt);
                while ((int)out.size() < n_gen) { out.push_back(argmax(lg.data(), V)); lg = m->eval({out.back()}, nt); }
            } else {
                Speculator sp(*m, std::getenv("TIERLM_DRAFT_N") ? std::atoi(std::getenv("TIERLM_DRAFT_N")) : 3);
                if (const char* lk = std::getenv("TIERLM_LOOKUP")) sp.set_lookup(std::atoi(lk), 2);
                if (const char* sc = std::getenv("TIERLM_DRAFT_SCRIPT")) {
                    std::vector<int32_t> script;
                    std::stringstream ss(sc);
                    for (std::string t; std::getline(ss, t, ',');) script.push_back(std::atoi(t.c_str()));
                    const int every = std::getenv("TIERLM_DRAFT_CORRUPT") ? std::atoi(std::getenv("TIERLM_DRAFT_CORRUPT")) : 0;
                    const int np = (int)ids.size();
                    sp.set_draft_source([script, every, np, V](int pos, int32_t, int n) {
                        std::vector<int32_t> d;                           // x is script[pos - np]; guess what follows
                        for (int i = 1; i <= n && pos - np + i < (int)script.size(); ++i) {
                            const int j = pos - np + i;
                            d.push_back(every > 0 && j % every == 0 ? (script[j] + 1) % V : script[j]);
                        }
                        return d;
                    });
                }
                lg = sp.prompt(ids, nt);
                int32_t x = argmax(lg.data(), V);
                out.push_back(x);
                while ((int)out.size() < n_gen) {
                    auto prod = sp.step(x, nt, [&](const float* row, const std::vector<int32_t>&) { return argmax(row, V); },
                                        [](int32_t) { return false; });
                    for (int32_t t : prod) out.push_back(t);
                    x = prod.back();
                }
                out.resize(n_gen);
                const auto& st = sp.stats();
                std::fprintf(stderr, "steps %d, drafted %d, accepted %d, lookup %d/%d\n", st.steps, st.drafted, st.accepted, st.look_ok, st.looked);
            }
            for (int32_t t : out) std::printf("%d\n", t);
            return 0;
        } else if (par) {
            // 1. a throwaway conversation (the last one) takes some cells, so the others' cells are scattered later;
            // 2. conversation 0's first half through the one-conversation path (eval);
            // 3. the other conversations' first halves together; then the throwaway one is dropped (a hole);
            // 4. the rest in mixed steps: even conversations 1 token per step, odd ones up to 3.
            const int K = par, junk = par, n = (int)ids.size();
            std::vector<std::vector<int32_t>> seq(K);
            for (int s = 0; s < K; ++s) {
                const int len = std::max(1, n - s);
                for (int i = 0; i < len; ++i) seq[s].push_back(ids[(i + 3 * s) % n]);
            }
            m->eval_seqs({{junk, {1, 2, 3, 4, 5, 6, 7}}}, nt);
            std::vector<size_t> at(K, 0);
            std::vector<std::vector<float>> last(K);
            at[0] = std::max<size_t>(1, seq[0].size() / 2);
            last[0] = m->eval(std::vector<int32_t>(seq[0].begin(), seq[0].begin() + at[0]), nt);
            std::vector<Model::SeqInput> in;
            for (int s = 1; s < K; ++s) {
                at[s] = std::max<size_t>(1, seq[s].size() / 2);
                in.push_back({s, std::vector<int32_t>(seq[s].begin(), seq[s].begin() + at[s])});
            }
            auto run = [&](const std::vector<Model::SeqInput>& in) {
                auto r = m->eval_seqs(in, nt);
                for (size_t k = 0; k < in.size(); ++k) last[in[k].seq] = r[k];
            };
            if (!in.empty()) run(in);
            m->seq_reset(junk);
            for (;;) {
                in.clear();
                for (int s = 0; s < K; ++s) {
                    const size_t take = std::min<size_t>(seq[s].size() - at[s], s % 2 ? 3 : 1);
                    if (!take) continue;
                    in.push_back({s, std::vector<int32_t>(seq[s].begin() + at[s], seq[s].begin() + at[s] + take)});
                    at[s] += take;
                }
                if (in.empty()) break;
                run(in);
            }
            for (int s = 0; s < K; ++s) for (float v : last[s]) std::printf("%.6g\n", v);
            return 0;
        } else if (mtp || mtp_step) {
            const int n = (int)ids.size(), E = m->hparams().n_embd;
            std::vector<float> hid;
            for (int off = 0; off < n; off += m->batch()) {
                const int k = std::min(m->batch(), n - off);
                auto r = m->eval_ex(ids.data() + off, k, nt, 1, true, 1);
                hid.insert(hid.end(), r.hidden.begin(), r.hidden.end());
            }
            std::vector<float> hprev((size_t)n * E, 0.0f);            // h_{t-1}; zeros at position 0
            std::copy(hid.begin(), hid.end() - E, hprev.begin() + E);
            if (mtp) lg = m->mtp_eval(ids.data(), hprev.data(), n, 0, nt);
            else for (int t = 0; t < n; ++t) lg = m->mtp_eval(ids.data() + t, hprev.data() + (size_t)t * E, 1, t, nt);
        } else if (rollback) {
            const size_t np = ids.size() - nb - 1;
            m->eval(std::vector<int32_t>(ids.begin(), ids.begin() + np), nt);
            m->eval_ex(ids.data() + np, nb, nt, nb, true, nb);
            m->accept(keep);
            lg = m->eval({ids.back()}, nt);
        } else if (!step && std::getenv("TIERLM_SNAPSHOT")) {
            // Save the conversation half way, read a different one, bring the first back and finish it: the result
            // must be the same as reading it straight through.
            const size_t h = ids.size() / 2;
            m->eval(std::vector<int32_t>(ids.begin(), ids.begin() + h), nt);
            Model::Snapshot snap = m->save();
            m->reset();
            m->eval(std::vector<int32_t>(ids.rbegin(), ids.rend()), nt);
            m->restore(snap);
            lg = m->eval(std::vector<int32_t>(ids.begin() + h, ids.end()), nt);
        } else if (!step) lg = m->eval(ids, nt);
        else for (int32_t id : ids) lg = m->eval({id}, nt);
        if (m->live_hot() && std::getenv("TIERLM_SHOW_PLACEMENT")) {
            const auto& ls = m->live_stats();
            std::fprintf(stderr, "live experts: %lld picks, %lld served from VRAM, %lld swaps\n", ls.picks, ls.hits, ls.swaps);
        }
        for (float v : lg) std::printf("%.6g\n", v);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
