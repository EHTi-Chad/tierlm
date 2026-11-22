"""Image reading for Qwen3.8 Flash-Next (GGUF arch "qwen4exp" with its "mmproj" vision file), with no real model.

Builds tiny random Flash-Next models with transformers' own Qwen4ExpForConditionalGeneration (the vision encoder, the
same one as Qwen3.5 / 3.6, and the Flash-Next text model), writes the text model as tests/tiny_flashnext.py does and
the vision encoder as tests/tiny_vision.py does, and compares tierlm's logits after a chat that reads an image with
Hugging Face's. What's particular to Flash-Next here:
  - the per-layer n-gram embedding (PLE) reads the image's places as the image token (<|image_pad|>, the file's
    qwen4exp.ple.image_token_id), so the n-grams after an image are made with it;
  - the 3-part image positions (image, row, column) also go to the sparse attention's indexer (the qsa case, whose
    context goes far past its tiny budget).
Fed in one call, one token per call, and in calls that cut the image in the middle; one image and two.

The MTP draft block after images (guessing ahead in a conversation with pictures): the block reads what the model's
own layers read, the images' words where their tokens are and the layers' positions (3-part at an image, running
ahead of the places after it). Hugging Face's code has no draft block, so the reference is put together from its
layers as in tests/tiny_flashnext_mtp.py. Checked over every place at once, one at a time and seven at a time; and
the model's greedy reply after the images is the same with guessing as without.

usage: python tests/tiny_flashnext_vision.py build
"""
import os, subprocess, sys, tempfile
import numpy as np
import gguf
import torch
from PIL import Image
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny_flashnext as fn
import tiny_flashnext_mtp as fm

build = sys.argv[1] if len(sys.argv) > 1 else "build"
exe = os.path.join(build, "tierlm-vision")
tmp = tempfile.mkdtemp()
VSTART, VEND, IMG = 297, 298, 299                      # vision start / end / image pad (the vocabulary is 300)
VE, VH, VFF, VDEPTH, POS = 32, 2, 48, 2, 16
E = fn.BASE["hidden_size"]

def build_model(over, seed):
    torch.manual_seed(seed)
    text = {**fn.BASE, **over}
    text.pop("attn_implementation", None)
    cfg = Qwen4ExpConfig(text_config=text,
                         vision_config=dict(depth=VDEPTH, hidden_size=VE, num_heads=VH, intermediate_size=VFF, patch_size=16,
                                            spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=E,
                                            num_position_embeddings=POS, rope_parameters=dict(rope_type="axial", rope_theta=10000.0)),
                         image_token_id=IMG, vision_start_token_id=VSTART, vision_end_token_id=VEND, video_token_id=296,
                         attn_implementation="eager")
    m = Qwen4ExpForConditionalGeneration(cfg).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("A_log"): p.copy_(torch.log(torch.empty_like(p).uniform_(0.5, 4)))
            elif n.endswith("dt_bias"): p.copy_(0.5 + 0.1 * torch.randn_like(p))
            elif "visual" in n and p.dim() == 1 and ("norm" in n or "ln" in n): p.copy_(1 + 0.1 * torch.randn_like(p))
            elif "linear_attn.norm" in n: p.copy_(1 + 0.1 * torch.randn_like(p))
            elif "norm" in n: p.copy_(0.1 * torch.randn_like(p))
            else: p.copy_(0.2 * torch.randn_like(p))
    return cfg, m

class TextView:                                         # the text half, named as Qwen4ExpForCausalLM names it
    def __init__(self, m):
        self.model = m.model.language_model
        self.sd = {"model." + k: v for k, v in self.model.state_dict().items()}
        self.sd["lm_head.weight"] = m.lm_head.weight
    def state_dict(self): return self.sd

def write_mmproj(m, path):
    sd = {k: v.detach().numpy().astype(np.float32) for k, v in m.model.visual.state_dict().items()}
    T = {}
    w = sd["patch_embed.proj.weight"]
    T["v.patch_embd.weight"], T["v.patch_embd.weight.1"] = w[:, :, 0], w[:, :, 1]
    T["v.patch_embd.bias"] = sd["patch_embed.proj.bias"]
    T["v.position_embd.weight"] = sd["pos_embed.weight"]
    for i in range(VDEPTH):
        b, s = f"v.blk.{i}.", f"blocks.{i}."
        for ours, theirs in (("ln1", "norm1"), ("ln2", "norm2"), ("attn_qkv", "attn.qkv"), ("attn_out", "attn.proj"),
                             ("ffn_up", "mlp.linear_fc1"), ("ffn_down", "mlp.linear_fc2")):
            T[b + ours + ".weight"], T[b + ours + ".bias"] = sd[s + theirs + ".weight"], sd[s + theirs + ".bias"]
    for ours, theirs in (("v.post_ln", "merger.norm"), ("mm.0", "merger.linear_fc1"), ("mm.2", "merger.linear_fc2")):
        T[ours + ".weight"], T[ours + ".bias"] = sd[theirs + ".weight"], sd[theirs + ".bias"]
    g = gguf.GGUFWriter(path, "clip")
    g.add_string("clip.projector_type", "qwen3vl_merger")
    g.add_bool("clip.has_vision_encoder", True)
    g.add_uint32("clip.vision.embedding_length", VE); g.add_uint32("clip.vision.feed_forward_length", VFF)
    g.add_uint32("clip.vision.block_count", VDEPTH); g.add_uint32("clip.vision.attention.head_count", VH)
    g.add_uint32("clip.vision.patch_size", 16); g.add_uint32("clip.vision.image_size", 64)
    g.add_uint32("clip.vision.projection_dim", E); g.add_uint32("clip.vision.spatial_merge_size", 2)
    g.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
    g.add_array("clip.vision.image_mean", [0.5, 0.5, 0.5]); g.add_array("clip.vision.image_std", [0.5, 0.5, 0.5])
    g.add_bool("clip.use_gelu", True)
    for n, t in T.items(): g.add_tensor(n, np.ascontiguousarray(t, dtype=np.float32))
    g.write_header_to_file(); g.write_kv_data_to_file(); g.write_tensors_to_file(); g.close()

def image(w, h, seed):
    r_ = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w] / max(w, h)
    img = np.stack([np.sin(6 * x + 2 * y), np.cos(5 * y - 3 * x), np.sin(9 * x * y)], -1) * 100 + 128
    img += r_.normal(0, 12, img.shape)
    path = os.path.join(tmp, f"pic{seed}.png")
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)
    return path

def hf_patches(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    a = ((a - 0.5) / 0.5).transpose(2, 0, 1)
    gh, gw = a.shape[1] // 16, a.shape[2] // 16
    p = a.reshape(3, gh // 2, 2, 16, gw // 2, 2, 16).transpose(1, 4, 2, 5, 0, 3, 6)
    p = np.broadcast_to(p[:, :, :, :, :, None], (*p.shape[:5], 2, 16, 16)).reshape(gh * gw, 3 * 2 * 16 * 16)
    return p, gh, gw

def rel(a, b): return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-9))

CASES = {"flashnext": dict(), "two-full-layers": dict(num_hidden_layers=8, ple_layer_ids=[2]),
         "qsa-8": dict(indexer_budget=8)}
pics = [image(96, 64, 1), image(64, 128, 2)]
rng = np.random.default_rng(5)
fails = []
for ci, (name, over) in enumerate(CASES.items()):
    cfg, m = build_model(over, 40 + ci)
    textm, _ = fn.write(cfg.text_config, TextView(m), image_tok=IMG)
    mmproj = os.path.join(tmp, f"mmproj-{name}.gguf")
    write_mmproj(m, mmproj)
    after_len = 60 if name.startswith("qsa") else 6
    for which in ((0,), (1,), (0, 1)):
        segs = [[int(x) for x in rng.integers(2, 290, 4)] + [VSTART]]
        for k, w in enumerate(which):
            segs.append([VEND] + [int(x) for x in rng.integers(2, 290, after_len if k == len(which) - 1 else 3)]
                        + ([VSTART] if k < len(which) - 1 else []))
        pvs, grids, ids = [], [], list(segs[0])
        for k, w in enumerate(which):
            pv, gh, gw = hf_patches(pics[w]); pvs.append(pv); grids.append([1, gh, gw])
            ids += [IMG] * (gh * gw // 4) + segs[k + 1]
        with torch.no_grad():
            want = m(input_ids=torch.tensor([ids]), pixel_values=torch.tensor(np.concatenate(pvs)),
                     image_grid_thw=torch.tensor(grids), mm_token_type_ids=torch.tensor([[int(t == IMG) for t in ids]])
                     ).logits[0, -1].numpy()
        cmd_ids = []
        for k in range(len(which)): cmd_ids += segs[k] + [-1]
        cmd_ids += segs[-1]
        for mode in ("all", "step", "7"):
            out = os.path.join(tmp, "o.f32")
            args = [exe, mmproj, "--chat", textm, ",".join(pics[w] for w in which) if len(which) > 1 else pics[which[0]], out, mode, *map(str, cmd_ids)]
            r = subprocess.run(args, capture_output=True, text=True, env={**os.environ, "TIERLM_DEVICE": "cpu", "TIERLM_VISION_EXACT": "1", "TIERLM_KV_F32": "1"})
            if r.returncode: print(f"{name}: tierlm-vision failed: {r.stderr.strip()[-600:]}"); fails.append(name); continue
            got = np.fromfile(out, np.float32)
            d = rel(got, want)
            ok = d < 1e-3 and got.argmax() == want.argmax()
            print(f"{name:16s} images {str(which):7s} {mode:4s}: worst relative diff {d:.1e}, top {got.argmax()} vs {want.argmax()}  ->  {'ok' if ok else 'FAIL'}")
            if not ok: fails.append(f"{name} {which} {mode}")
    os.remove(textm)
# ---- the draft block after images ----
EXACT = {**os.environ, "TIERLM_DEVICE": "cpu", "TIERLM_VISION_EXACT": "1", "TIERLM_KV_F32": "1"}
for ci, (name, over) in enumerate({"draft": dict(), "draft-8layer": dict(num_hidden_layers=8, ple_layer_ids=[2])}.items()):
    torch.manual_seed(70 + ci)
    cfg, m = build_model(over, 70 + ci)
    mtp = fm.draft_block(cfg.text_config)()
    textm = fm.write(cfg.text_config, TextView(m), mtp, image_tok=IMG)
    mmproj = os.path.join(tmp, f"mmproj-{name}.gguf")
    write_mmproj(m, mmproj)
    for which in ((0,), (0, 1)):
        segs = [[int(x) for x in rng.integers(2, 290, 4)] + [VSTART]]
        for k, w in enumerate(which):
            segs.append([VEND] + [int(x) for x in rng.integers(2, 290, 9 if k == len(which) - 1 else 3)] + ([VSTART] if k < len(which) - 1 else []))
        pvs, grids, ids = [], [], list(segs[0])
        for k, w in enumerate(which):
            pv, gh, gw = hf_patches(pics[w]); pvs.append(pv); grids.append([1, gh, gw])
            ids += [IMG] * (gh * gw // 4) + segs[k + 1]
        run = lambda: m(input_ids=torch.tensor([ids]), pixel_values=torch.tensor(np.concatenate(pvs)), image_grid_thw=torch.tensor(grids),
                        mm_token_type_ids=torch.tensor([[int(t == IMG) for t in ids]]))
        want = fm.mtp_reference(cfg.text_config, m, mtp, ids, text=m.model.language_model, run=run)
        cmd_ids = []
        for k in range(len(which)): cmd_ids += segs[k] + [-1]
        cmd_ids += segs[-1]
        out = os.path.join(tmp, "o.f32")
        def tool(mode, env=EXACT):
            r = subprocess.run([exe, mmproj, "--chat", textm, ",".join(pics[w] for w in which), out, mode, *map(str, cmd_ids)], capture_output=True, text=True, env=env)
            if r.returncode: print(f"{name}: tierlm-vision {mode} failed: {r.stderr.strip()[-600:]}"); return None, ""
            return np.fromfile(out, np.float32), r.stdout.strip()
        for mode in ("mtp", "mtp-step", "mtp-7"):
            got, _ = tool(mode)
            d = rel(got, want) if got is not None else 1.0
            ok = got is not None and d < 1e-3 and got.argmax() == want.argmax()
            print(f"{name:16s} images {str(which):7s} {mode:8s}: worst relative diff {d:.1e}  ->  {'ok' if ok else 'FAIL'}")
            if not ok: fails.append(f"{name} {which} {mode}")
        for label, env in (("exact", EXACT), ("normal", {**os.environ})):
            plain, _ = tool("gen", env)
            spec, stats = tool("gen-spec", env)
            ok = plain is not None and spec is not None and np.array_equal(plain, spec)
            print(f"{name:16s} images {str(which):7s} {label:8s}: the reply with guessing is {'the same' if ok else 'DIFFERENT'} ({stats})  ->  {'ok' if ok else 'FAIL'}")
            if not ok: fails.append(f"{name} {which} gen {label}")
    os.remove(textm)
print("PASS" if not fails else "FAIL: " + ", ".join(fails))
sys.exit(1 if fails else 0)
