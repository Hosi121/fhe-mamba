# Third-party sources

This repository's code and original coefficient bundles use the [MIT license](LICENSE).
External projects retain their own licenses and notices. The model weights,
FIDESlib/OpenFHE installations and datasets are downloaded separately.

| Component | Source and attribution | Upstream license |
| --- | --- | --- |
| Mamba-2 model and architecture | [State Spaces Mamba](https://github.com/state-spaces/mamba), Tri Dao and Albert Gu | Apache-2.0 for the implementation; see the checkpoint's own model card for weights |
| Converted Mamba-2-130M checkpoint | [AntonV/mamba2-130m-hf](https://huggingface.co/AntonV/mamba2-130m-hf), independent conversion of [state-spaces/mamba2-130m](https://huggingface.co/state-spaces/mamba2-130m) | MIT as declared by the conversion's model card |
| Mamba-3 SISO 187M | [state-spaces/mamba3-siso-187m](https://huggingface.co/state-spaces/mamba3-siso-187m), State Spaces | Apache-2.0 as declared by its model card; weights downloaded separately |
| Compatible Llama-3.1 tokenizer | [NVIDIA Llama-3.1 Nemotron](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF) | Upstream Llama-3.1 terms; tokenizer files downloaded separately |
| FIDESlib GPU backend | [CAPS-UMU/FIDESlib](https://github.com/CAPS-UMU/FIDESlib), Universidad de Murcia | [MIT](https://github.com/CAPS-UMU/FIDESlib/blob/cd171f20f510eeca04c71d7b0034ef073829f761/LICENSE.txt) |
| OpenFHE | [openfheorg/openfhe-development](https://github.com/openfheorg/openfhe-development) | [BSD-2-Clause](https://github.com/openfheorg/openfhe-development/blob/v1.4.2/LICENSE) |
| PyTorch, NumPy, Transformers | Installed through the Python dependency lockfile | Respective upstream licenses; not relicensed by this repository |

The native patch files modify the pinned FIDESlib/OpenFHE sources. Preserve
upstream notices when distributing those sources or compiled dependencies.
The checkpoint revision and file hashes used by the reproduction guide are
in [config/reproduction.json](config/reproduction.json).

The public coefficient bundles contain numerical approximation parameters,
not checkpoint weight tensors or private prompts. Their fitting/certification
code and the associated measured limitations are included in this repository.

Research background: Tri Dao and Albert Gu, *Transformers are SSMs:
Generalized Models and Efficient Algorithms Through Structured State Space
Duality*, ICML 2024. Additional algorithm sources are linked in the
[research notes](docs/research/2026-09-21-ssm-cryptographic-design.md).
