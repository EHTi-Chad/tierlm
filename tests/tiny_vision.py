"""Correctness test for image reading (Qwen3.5 / 3.6 with its vision "mmproj" file), with no real model.

Builds a tiny random Qwen3.5 with Hugging Face's own code (transformers' Qwen3_5ForConditionalGeneration: the vision
encoder and the hybrid text model), writes the text model and the vision encoder to GGUF the way the converter does,
and compares tierlm with Hugging Face, step by step:
  1. the prepared patches (an image already a multiple of 32, so no resizing): exact;
  2. the vision encoder's image words, from the same pixels: to float precision (big images, which use flash
     attention with 16-bit keys and values: close);
  3. a chat model reading an image between text: the logits after it, with the image's words given 3-part
     positions (image, row, column) and the text after it continuing from the image's largest position. Fed in
     one call, one token per call, and in calls that cut the image in the middle.
  4. an image that needs resizing: tierlm's bicubic resize against PIL's (close, not exact: Qwen's processors
     differ among themselves here too), and the image words' agreement.

usage: python tests/tiny_vision.py build
"""
import os, subprocess, sys, tempfile
import numpy as np
import gguf
import torch
from PIL import Image
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

build = sys.argv[1] if len(sys.argv) > 1 else "build"
exe = os.path.join(build, "tierlm-vision")
torch.manual_seed(7)
rng = np.random.default_rng(7)
tmp = tempfile.mkdtemp()

# ---- a tiny model, shaped like the real one where it matters ----
V, E, H, KV, HD, FF, L = 262, 64, 2, 1, 64, 48, 4           # text: 3 DeltaNet layers, then full attention
VE, VH, VFF, VDEPTH, POS = 32, 2, 48, 2, 16                   # vision: width, heads (head 16), MLP, blocks, 4x4 table
# A byte-level vocabulary (256 bytes, then the special tokens), so the files also work in tierlm-server
VSTART, VEND, IMG = 259, 260, 261                               # vision start / vision end / image pad token ids
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]
cfg = Qwen3_5Config(
    text_config=dict(vocab_size=V, hidden_size=E, num_hidden_layers=L, num_attention_heads=H, num_key_value_heads=KV,
                     head_dim=HD, intermediate_size=FF, linear_num_key_heads=2, linear_num_value_heads=4,
                     linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
                     layer_types=["linear_attention"] * 3 + ["full_attention"], max_position_embeddings=512,
                     rms_norm_eps=1e-6, tie_word_embeddings=False, mtp_num_hidden_layers=0,
                     rope_parameters=dict(rope_type="default", rope_theta=10000.0, partial_rotary_factor=0.25,
                                          mrope_section=[3, 3, 2], mrope_interleaved=True)),
    vision_config=dict(depth=VDEPTH, hidden_size=VE, num_heads=VH, intermediate_size=VFF, patch_size=16, spatial_merge_size=2,
                       temporal_patch_size=2, out_hidden_size=E, num_position_embeddings=POS,
                       rope_parameters=dict(rope_type="axial", rope_theta=10000.0)),
    image_token_id=IMG, vision_start_token_id=VSTART, vision_end_token_id=VEND, tie_word_embeddings=False)
model = Qwen3_5ForConditionalGeneration(cfg).float().eval()
with torch.no_grad():                                          # make every weight count (biases and norms too)
    for name, p in model.named_parameters():
        if "A_log" in name: p.copy_(torch.log(torch.empty_like(p).uniform_(0.5, 4)))
        elif "dt_bias" in name: p.copy_(0.5 + 0.1 * torch.randn_like(p))
        elif p.dim() == 1 and ("norm" in name or "ln" in name): p.copy_(0.1 * torch.randn_like(p) + (1.0 if "visual" in name or "linear_attn.norm" in name else 0.0))
        else: p.copy_(0.2 * torch.randn_like(p))
sd = {k: v.detach().numpy().astype(np.float32) for k, v in model.state_dict().items()}
pre = "model.language_model." if any(k.startswith("model.language_model.") for k in sd) else "model."
vis = "model.visual." if any(k.startswith("model.visual.") for k in sd) else "visual."

# ---- the vision file, named as the converter names it ----
def write_mmproj():
    T = {}
    w = sd[vis + "patch_embed.proj.weight"]                    # [VE, 3, 2, 16, 16]: Conv3D over 2 frames
    T["v.patch_embd.weight"], T["v.patch_embd.weight.1"] = w[:, :, 0], w[:, :, 1]
    T["v.patch_embd.bias"] = sd[vis + "patch_embed.proj.bias"]
    T["v.position_embd.weight"] = sd[vis + "pos_embed.weight"]
    for i in range(VDEPTH):
        b, s = f"v.blk.{i}.", f"{vis}blocks.{i}."
        for ours, theirs in (("ln1", "norm1"), ("ln2", "norm2"), ("attn_qkv", "attn.qkv"), ("attn_out", "attn.proj"),
                             ("ffn_up", "mlp.linear_fc1"), ("ffn_down", "mlp.linear_fc2")):
            T[b + ours + ".weight"] = sd[s + theirs + ".weight"]
            T[b + ours + ".bias"] = sd[s + theirs + ".bias"]
    for ours, theirs in (("v.post_ln", "merger.norm"), ("mm.0", "merger.linear_fc1"), ("mm.2", "merger.linear_fc2")):
        T[ours + ".weight"], T[ours + ".bias"] = sd[vis + theirs + ".weight"], sd[vis + theirs + ".bias"]
    path = os.path.join(tmp, "mmproj-tiny.gguf")
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
    return path

# ---- the text model, written the way the converter does (as in tests/tiny_qwen35.py) ----
HK, HV, S = 2, 4, 16
R = HV // HK
def tiled(x, axis, head):
    x = np.moveaxis(x, axis, 0); sh = x.shape
    x = x.reshape(HK, R, head, *sh[1:]).transpose(1, 0, 2, *range(3, len(sh) + 2)).reshape(sh)
    return np.ascontiguousarray(np.moveaxis(x, 0, axis))

def write_text():
    T = {"token_embd.weight": sd[pre + "embed_tokens.weight"], "output_norm.weight": 1 + sd[pre + "norm.weight"],
         "output.weight": sd["lm_head.weight"]}
    qk = 2 * HK * S
    for i in range(L):
        b, s = f"blk.{i}.", f"{pre}layers.{i}."
        T.update({b + "attn_norm.weight": 1 + sd[s + "input_layernorm.weight"],
                  b + "post_attention_norm.weight": 1 + sd[s + "post_attention_layernorm.weight"],
                  b + "ffn_gate.weight": sd[s + "mlp.gate_proj.weight"], b + "ffn_up.weight": sd[s + "mlp.up_proj.weight"],
                  b + "ffn_down.weight": sd[s + "mlp.down_proj.weight"]})
        if i < 3:
            a = s + "linear_attn."
            qkv, conv = sd[a + "in_proj_qkv.weight"], sd[a + "conv1d.weight"][:, 0, :]
            T.update({b + "attn_qkv.weight": np.concatenate([qkv[:qk], tiled(qkv[qk:], 0, S)]),
                      b + "attn_gate.weight": tiled(sd[a + "in_proj_z.weight"], 0, S),
                      b + "ssm_beta.weight": tiled(sd[a + "in_proj_b.weight"], 0, 1),
                      b + "ssm_alpha.weight": tiled(sd[a + "in_proj_a.weight"], 0, 1),
                      b + "ssm_conv1d.weight": np.concatenate([conv[:qk], tiled(conv[qk:], 0, S)]),
                      b + "ssm_dt.bias": tiled(sd[a + "dt_bias"], 0, 1), b + "ssm_a": tiled(-np.exp(sd[a + "A_log"]), 0, 1),
                      b + "ssm_norm.weight": sd[a + "norm.weight"], b + "ssm_out.weight": tiled(sd[a + "out_proj.weight"], 1, S)})
        else:
            a = s + "self_attn."
            T.update({b + "attn_q.weight": sd[a + "q_proj.weight"], b + "attn_k.weight": sd[a + "k_proj.weight"],
                      b + "attn_v.weight": sd[a + "v_proj.weight"], b + "attn_output.weight": sd[a + "o_proj.weight"],
                      b + "attn_q_norm.weight": 1 + sd[a + "q_norm.weight"], b + "attn_k_norm.weight": 1 + sd[a + "k_norm.weight"]})
    path = os.path.join(tmp, "tiny-qwen35-vl.gguf")
    g = gguf.GGUFWriter(path, "qwen35")
    g.add_block_count(L); g.add_embedding_length(E); g.add_head_count(H); g.add_head_count_kv(KV)
    g.add_key_length(HD); g.add_value_length(HD); g.add_rope_dimension_count(HD // 4); g.add_rope_freq_base(10000.0)
    g.add_layer_norm_rms_eps(1e-6); g.add_feed_forward_length(FF); g.add_context_length(512)
    g.add_ssm_conv_kernel(4); g.add_ssm_state_size(S); g.add_ssm_group_count(HK); g.add_ssm_time_step_rank(HV)
    g.add_ssm_inner_size(HV * S); g.add_full_attention_interval(4)
    g.add_array("qwen35.rope.dimension_sections", [3, 3, 2, 0])
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b_ in range(256):
        if b_ not in bs: bs.append(b_); cs.append(256 + n); n += 1
    byte_char = dict(zip(bs, map(chr, cs)))                    # GPT-2's bytes-as-characters
    g.add_string("general.name", "Tiny test Qwen3.5 VL")
    g.add_tokenizer_model("gpt2"); g.add_tokenizer_pre("qwen35")
    g.add_token_list([byte_char[b_] for b_ in range(256)] + SPECIAL)
    g.add_token_types([1] * 256 + [3] * len(SPECIAL))
    g.add_eos_token_id(258)
    g.add_chat_template("{% for m in messages %}<|im_start|>{{m.role}}\n{{m.content}}<|im_end|>\n{% endfor %}")
    for n, t in T.items(): g.add_tensor(n, np.ascontiguousarray(t, dtype=np.float32))
    g.write_header_to_file(); g.write_kv_data_to_file(); g.write_tensors_to_file(); g.close()
    return path

mmproj, textm = write_mmproj(), write_text()
if os.environ.get("TIERLM_VISION_SAVE"):                         # keep the pair, e.g. for trying tierlm-server
    import shutil
    d = os.environ["TIERLM_VISION_SAVE"]; os.makedirs(d, exist_ok=True)
    shutil.copy(textm, os.path.join(d, "Tiny-Qwen3.5-VL-F32.gguf")); shutil.copy(mmproj, os.path.join(d, "mmproj-Tiny-Qwen3.5-VL-F32.gguf"))
fails = []
def check(name, ok, detail):
    print(f"{name:34s}: {detail}  ->  {'ok' if ok else 'FAIL'}")
    if not ok: fails.append(name)
def rel(a, b): return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-9))
def run(*args):
    r = subprocess.run([exe, *map(str, args)], capture_output=True, text=True, env={**os.environ, "TIERLM_DEVICE": "cpu", "TIERLM_VISION_EXACT": "1"})
    if r.returncode: raise SystemExit(f"tierlm-vision failed: {r.stderr}")
    return r.stdout.split()

def image(w, h, seed):                                       # smooth shapes and some noise, like a photo
    r_ = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w] / max(w, h)
    img = np.stack([np.sin(6 * x + 2 * y), np.cos(5 * y - 3 * x), np.sin(9 * x * y)], -1) * 100 + 128
    img += r_.normal(0, 12, img.shape)
    path = os.path.join(tmp, f"pic{seed}.png")
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)
    return path

def hf_patches(path):                                           # Qwen's processor steps, by hand, on exact sizes
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    a = ((a - 0.5) / 0.5).transpose(2, 0, 1)                    # [3, H, W]
    gh, gw = a.shape[1] // 16, a.shape[2] // 16
    p = a.reshape(3, gh // 2, 2, 16, gw // 2, 2, 16).transpose(1, 4, 2, 5, 0, 3, 6)   # Qwen2VLImageProcessor.patchify
    p = np.broadcast_to(p[:, :, :, :, :, None], (*p.shape[:5], 2, 16, 16)).reshape(gh * gw, 3 * 2 * 16 * 16)
    return p, gh, gw

# 1-2. patches and image words, a 96 x 64 image (no resizing)
pic = image(96, 64, 1)
pv, gh, gw = hf_patches(pic)
out = os.path.join(tmp, "o.f32")
g2 = run(mmproj, "--patches", pic, out)
ours = np.fromfile(out, np.float32).reshape(gh * gw, 3, 256)
check("patches 96x64", [int(v) for v in g2] == [gh, gw] and rel(ours, pv.reshape(gh * gw, 3, 2, 256)[:, :, 0]) < 1e-6,
      f"grid {g2}, worst relative diff {rel(ours, pv.reshape(gh * gw, 3, 2, 256)[:, :, 0]):.1e}")
with torch.no_grad():
    ref = model.model.visual(torch.tensor(pv), grid_thw=torch.tensor([[1, gh, gw]])).pooler_output.numpy()
words = run(mmproj, pic, out)
ours = np.fromfile(out, np.float32).reshape(-1, E)
check("image words 96x64", ours.shape == ref.shape and rel(ours, ref) < 1e-4, f"{ours.shape[0]} words, worst relative diff {rel(ours, ref):.1e}")

# A taller image too (rows != columns), 64 x 128
pic2 = image(64, 128, 2)
pv2, gh2, gw2 = hf_patches(pic2)
with torch.no_grad():
    ref2 = model.model.visual(torch.tensor(pv2), grid_thw=torch.tensor([[1, gh2, gw2]])).pooler_output.numpy()
run(mmproj, pic2, out)
ours2 = np.fromfile(out, np.float32).reshape(-1, E)
check("image words 64x128", ours2.shape == ref2.shape and rel(ours2, ref2) < 1e-4, f"worst relative diff {rel(ours2, ref2):.1e}")

# A big image (more than 1,024 patches): tierlm uses flash attention there (16-bit keys and values): close
pic4 = image(512, 576, 4)
pv4, gh4, gw4 = hf_patches(pic4)
with torch.no_grad():
    ref4 = model.model.visual(torch.tensor(pv4), grid_thw=torch.tensor([[1, gh4, gw4]])).pooler_output.numpy()
r4 = subprocess.run([exe, mmproj, pic4, out], capture_output=True, text=True, env={**os.environ, "TIERLM_DEVICE": "cpu"})
ours4 = np.fromfile(out, np.float32).reshape(-1, E)
check("image words 512x576 (flash attn.)", r4.returncode == 0 and ours4.shape == ref4.shape and rel(ours4, ref4) < 1e-2,
      f"{gh4 * gw4} patches, worst relative diff {rel(ours4, ref4):.1e}")

# 3. a chat model reading the image between text, then more text
for picpath, pvx, ghx, gwx, tag in ((pic, pv, gh, gw, "96x64"), (pic2, pv2, gh2, gw2, "64x128")):
    n_img = ghx * gwx // 4
    before, after = [5, 17, 33, VSTART], [VEND, 44, 9, 71, 12, 3]
    ids = before + [IMG] * n_img + after
    with torch.no_grad():
        tt = torch.tensor([ids])
        mm = torch.tensor([[1 if t == IMG else 0 for t in ids]])
        logits = model(input_ids=tt, pixel_values=torch.tensor(pvx), image_grid_thw=torch.tensor([[1, ghx, gwx]]),
                       mm_token_type_ids=mm).logits[0, -1].numpy()
    for mode in ("all", "step", "7"):
        run(mmproj, "--chat", textm, picpath, out, mode, *before, -1, *after)
        got = np.fromfile(out, np.float32)
        d = rel(got, logits)
        check(f"chat with image {tag}, {mode}", d < 1e-3 and got.argmax() == logits.argmax(), f"worst relative diff {d:.1e}, top token {got.argmax()} vs {logits.argmax()}")

# Without an image, text alone is unchanged (positions are the plain ones)
ids = [5, 17, 33, 44, 9, 71]
with torch.no_grad():
    logits = model(input_ids=torch.tensor([ids])).logits[0, -1].numpy()
run(mmproj, "--chat", textm, pic, out, "all", *ids)
got = np.fromfile(out, np.float32)
check("text only (vision file loaded)", rel(got, logits) < 1e-3, f"worst relative diff {rel(got, logits):.1e}")

# 4. an image that must be resized: 100 x 70 -> 96 x 64. tierlm's bicubic against PIL's.
pic3 = image(100, 70, 3)
run(mmproj, "--patches", pic3, out)
ours3 = np.fromfile(out, np.float32).reshape(-1, 3, 16, 16)
pil = Image.open(pic3).convert("RGB").resize((96, 64), Image.BICUBIC)
pil.save(os.path.join(tmp, "pil.png"))
pv3, _, _ = hf_patches(os.path.join(tmp, "pil.png"))
pv3 = pv3.reshape(-1, 3, 2, 16, 16)[:, :, 0]
d = float(np.abs(ours3 - pv3).max())
check("resize 100x70 -> 96x64 (vs PIL)", d < 2.5 / 127.5 * 2, f"largest difference {d * 127.5:.2f} of 255 levels")

print("PASS" if not fails else "FAIL: " + ", ".join(fails))
sys.exit(1 if fails else 0)
