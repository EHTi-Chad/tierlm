# Third-party notices

tierlm includes or depends on the following open-source components. Each keeps its own licence; the full
licence text ships with the component at the path shown. If you redistribute tierlm (source or binaries),
keep these notices and licence files with it.

| Component | Used for | Licence | Location |
|---|---|---|---|
| [ggml](https://github.com/ggml-org/ggml) | Tensor math kernels, quantized formats, device backends (CPU, Vulkan, CUDA), GGUF file reader | MIT, Copyright (c) 2023-2026 The ggml authors | `third_party/ggml/LICENSE` (git submodule) |
| [cpp-httplib](https://github.com/yhirose/cpp-httplib) v0.63.0 | Embedded HTTP server in `tierlm-server` | MIT, Copyright (c) 2017 yhirose | `third_party/httplib/LICENSE` |
| [nlohmann/json](https://github.com/nlohmann/json) v3.12.0 | JSON parsing and output in `tierlm-server` | MIT, Copyright (c) 2013-2025 Niels Lohmann | `third_party/json/LICENSE.MIT` |

## Used for testing only (not shipped)

These are used by the scripts in `tests/` to check tierlm's results. They are not part of the engine or
its binaries.

| Component | Used for | Licence |
|---|---|---|
| [Hugging Face tokenizers](https://github.com/huggingface/tokenizers) | Reference token IDs for the tokenizer test | Apache 2.0 |
| [gguf-py](https://pypi.org/project/gguf/) | Writing tiny test models | MIT |
| [NumPy](https://numpy.org) | Reference implementation of the OLMoE math | BSD-3-Clause |

## Models

Model weights are not part of this repository. Each model has its own licence, set by its publisher. For
example, OLMoE-1B-7B-0924-Instruct is released by Ai2 under Apache 2.0. Check the licence of any model
before using it, especially commercially.
