# tierlm's changes to ggml, and what we learned about it

tierlm runs on [ggml](https://github.com/ggml-org/ggml) (MIT license), the tensor library behind llama.cpp. We change
very little of it: **50 lines in 4 files**, listed here in full so anyone can review them, reuse them, or offer them
upstream. This page also records three things about ggml that took us a while to find and that matter to anyone
building on ggml directly rather than through llama.cpp.

- Base: ggml-org/ggml commit `da8e2a54` ("ci : upload API/ABI reports as artifacts (#1666)")
- The changes: [`ggml-tierlm.patch`](ggml-tierlm.patch) in this folder (apply with `git apply` in a ggml checkout of that commit),
  also on the `tierlm` branch of [EHTi-Chad/ggml](https://github.com/EHTi-Chad/ggml), which tierlm uses as its submodule
- License: the changes are MIT, like ggml itself

| # | Change | Files | Lines | Why |
|---|---|---|---|---|
| 1 | The CPU expert multiply skips negative expert ids | `src/ggml-cpu/ggml-cpu.c`, `src/ggml-cpu/repack.cpp` | +12 −2 | Lets a MoE layer be split between VRAM and RAM by expert, without the CPU doing work for experts that are on the GPU |
| 2 | Optional per-split timing in the scheduler | `src/ggml-backend.cpp`, `include/ggml-backend.h` | +38 | Shows where each token's time goes (GPU work, CPU work, copies between them) |

Nothing else in ggml is changed. tierlm also turns on one existing ggml build option (CUDA graphs, see Finding 1).

**Status.** These changes are made for tierlm and are not offered to ggml as they stand. Change 1 is small and
self-contained, but gives negative ids a meaning only on the CPU; a change for ggml itself would likely need every
backend to agree. Change 2 keeps one set of totals for the whole program (fine for tierlm, which runs one scheduler)
and is switched on by a tierlm-named variable; for ggml it would need per-scheduler counters and a neutral name. The
findings below need no changes to ggml at all.

---

## Change 1: the CPU expert multiply skips negative expert ids

**What.** `GGML_OP_MUL_MAT_ID` (the "multiply each token by the experts it picked" op of mixture-of-experts models)
takes an `ids` tensor with each token's chosen experts. On the CPU, an id below 0 now means *no expert here*: that
token's output row is set to zero and nothing is computed for it. Ids from 0 up work exactly as before.

```c
// src/ggml-cpu/ggml-cpu.c, ggml_compute_forward_mul_mat_id (the same 5 lines in repack.cpp's forward_mul_mat_id)
const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

// tierlm: a negative id means "no expert here" (a pick computed elsewhere): its output row is zero
if (i02 < 0) {
    memset((char *) dst->data + id*dst->nb[1] + iid1*dst->nb[2], 0, dst->ne[0]*sizeof(float));
    continue;
}
assert(i02 < n_as);
```

**Compatibility.** A negative id used to be invalid (an assert in debug builds, out-of-bounds reads in release
builds), so no working graph changes behaviour. The zeroing is done by thread 0 while it groups the rows, before the
barrier, so no other thread writes those rows. The tiled and repacked paths only ever see the grouped rows, so they
need no change. Only the CPU backend understands negative ids: the GPU backends still need valid ids (see Finding 2).

**Why.** tierlm can keep a model's most-used experts in VRAM and the rest in RAM ("hot experts"). Each such layer is
then two expert tensors: the experts in VRAM, computed by the GPU, and the experts in RAM, computed by the CPU. Every
token's picks are looked up in both: a pick that lives in the other part has to go *somewhere*. Before this change it
went to an all-zero stand-in expert with weight 0, which gives the right sum, but the CPU still multiplied the zero
expert out in full for every such pick. With 72% of the picks in VRAM, the CPU did as many expert multiplies as
without the split.

**Measured** (RTX 3070 laptop, Qwen3.6-35B-A3B UD-Q3_K_XL, CPU work per token with ggml's timing from Change 2):

| | CPU work per token |
|---|---|
| Experts in RAM, no split | 17.9 ms |
| Hot experts, picks of VRAM experts sent to the zero expert | about as much (the split was 18.2 against 23.3 tok/s) |
| Hot experts, picks of VRAM experts skipped (this change) | 13.6 ms |

**Exactness.** tierlm's tests compare every model type against a NumPy reference, with the model split over a
stand-in GPU and RAM at several VRAM sizes; split layers match to ~1e-6.

---

## Change 2: optional per-split timing in the scheduler

**What.** With the environment variable `TIERLM_PROFILE` set, `ggml_backend_sched_compute_splits` waits for each
split and adds up, per backend, the time spent copying the split's inputs in and the time spent computing it. A new
function reads (and optionally resets) the totals:

```c
// include/ggml-backend.h
// arrays have GGML_SCHED_MAX_BACKENDS (16) entries, indexed like the backends given to ggml_backend_sched_new
GGML_API void ggml_backend_sched_profile(double * copy_us, double * compute_us, int64_t * splits,
                                         int64_t * inputs, int64_t * graphs, bool reset);
```

When the variable is not set, the only cost is one `if` per split. When it is set, every split is waited for, so the
times add up correctly but everything runs a little slower than normal.

**How tierlm uses it.** Every 64 reply steps (graphs of up to 8 tokens) tierlm prints, for example:

```
profile: 64 graphs, 1.0 tokens each: 45.99 ms each = prepare 2.50 + compute 43.18 + read out 0.32
profile:   CUDA0       30.0 splits,   32.0 inputs: copy in   1.28 ms, work  22.57 ms per graph
profile:   CPU         30.0 splits,   58.0 inputs: copy in   1.46 ms, work  17.86 ms per graph
```

**What it found.** The GPU line above: 22.6 ms of GPU work per token for a part of the model that an RTX 3070 can
read in 2–3 ms. The GPU was spending its time launching thousands of small kernels, not computing. That led to
Finding 1, which made generation about 50% faster (21.7 → 33.3 tok/s on that laptop).

---

## Finding 1: CUDA graphs are off unless you turn them on, and they need a stable graph

ggml's `GGML_CUDA_GRAPHS` CMake option defaults to **OFF** (its description says "llama.cpp only"); llama.cpp turns it
on for itself. Without it, every op of every token is a separate kernel launch, and on Windows (WDDM) launches are
slow. With it, ggml records a split's kernels once and replays them with one launch.

Turning it on is not enough: ggml re-records (and, after a change, stops replaying until two identical calls in a
row) whenever a node's shape or data pointer changes. A plain decoding loop changes them every token. What tierlm
does so the replay holds (all outside ggml, in `src/model.cpp`):

1. **Write the KV cache by index, not into a view at the current position.** A `ggml_cpy` into
   `ggml_view(cache, offset = n_past)` has a new data pointer every token. `ggml_set_rows(cache, k, pos)`, with the
   positions as an input tensor, has the same graph every token.
2. **Let attention cover the cache in steps of 256 positions.** The K/V views and the mask are padded up to the
   next multiple of 256 (the extra positions are masked with -inf, so the result is exactly the same), so their
   shapes change once every 256 tokens instead of every token. The cache buffer is cleared at allocation, so the
   padded part never holds NaNs.
3. **Build each kind of graph in its own memory.** ggml-cuda keys its recorded graphs by the address of a split's
   first node. tierlm builds its graphs into a reused buffer; one-token steps, several-token steps (speculative
   decoding's check) and the MTP draft head now each get their own buffer, so they don't overwrite each other's
   recording.

**Measured** (same laptop and model, tierlm's "Find the fastest settings", tok/s):

| | before | after |
|---|---|---|
| Experts in RAM | 21.7 | 33.3 |
| Whole layers | 13.2 | 13.0–14.4 |
| Best setting (with speculative decoding) | 31.6 | 35.3 |

Results are identical: only how the work is launched changes.

## Finding 2: CUDA's expert multiply assumes a token never picks the same expert twice

`ggml_cuda_mul_mat_id` sorts the picks by expert (`mm_ids_helper` in `mmid.cu`) before the batched multiply (mmq).
Within a token, each lane that matches the current expert writes its slot to the same compacted index, so if one
token lists the same expert twice, one slot is lost; for gate/up the inverse map entry for it is then never written,
and the quantize step reads an uninitialised index and writes out of bounds. The result is
`CUDA error: an illegal memory access was encountered`, reported later at a synchronize or in `ggml_cuda_mul_mat_q`.

A router never picks an expert twice, so this does not happen in normal use. It does happen as soon as ids are
remapped, for example several picks pointing at one stand-in expert, as tierlm's hot-expert split first did. The CPU
backend handles repeats correctly, so a test on the CPU (or on ggml's RPC backend, which computes on the CPU) does
not show it. With `CUDA_LAUNCH_BLOCKING=1` the error is reported at the failing kernel.

tierlm's fix (outside ggml): before ids go to a GPU, a small CPU op moves any repeated id in a token to a distinct
unused expert of that part, whose weight is 0 there.

## Finding 3: ggml moves big batches of RAM weights to the GPU by itself

With `op_offload` on in `ggml_backend_sched_new` (tierlm turns it on when weights are split), ops with weights in RAM
and a batch of at least 32 tokens (`GGML_OP_OFFLOAD_MIN_BATCH`) run on the GPU: the weights are copied over for that
op. That is why reading a long prompt is fast even with experts in RAM. It also means a graph's RAM part is not always
computed by the CPU, so anything that only the CPU accepts (like Change 1's negative ids) must not reach those ops
for big batches. tierlm uses negative ids only for batches below that size, and distinct ids otherwise.

---

## Using these changes in your own project

```
git clone https://github.com/ggml-org/ggml && cd ggml
git checkout da8e2a54
git apply path/to/tierlm/docs/ggml-tierlm.patch
```

The patch is small and touches code that changes rarely, so it usually applies to newer ggml versions as well. If
you only want the CUDA graphs speed-up, you need no patch at all: build with `-DGGML_CUDA_GRAPHS=ON` and keep your
graphs stable as described in Finding 1.
