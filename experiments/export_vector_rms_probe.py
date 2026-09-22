#!/usr/bin/env python3
"""Freeze real checkpoint RMS inputs, learned gamma and public stress vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fhemamba._env import block_broken_torchvision

block_broken_torchvision()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fhemamba import __version__  # noqa: E402
from fhemamba.artifacts import current_git_commit  # noqa: E402
from fhemamba.ops import Exact  # noqa: E402
from fhemamba.reference import model_forward  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CaptureNormInputs(Exact):
    def __init__(self, layers, positions):
        self.layers, self.positions, self.inputs = layers, positions, {}

    def checkpoint(self, x, site):
        layer, name = site
        target = None
        if name == "residual":
            target = (layer, "rms_invsqrt")
        elif name == "y":
            target = (layer, "gated_rms_invsqrt")
        elif name == "layer_output" and layer == self.layers - 1:
            target = (self.layers, "rms_invsqrt")
        if target is not None:
            self.inputs[target] = x[0, self.positions].detach().double().cpu().numpy()
        return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--token-offset", type=int, default=180224)
    parser.add_argument("--recipes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("refusing to overwrite frozen vector fixtures")
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).float().eval().cuda()
    token_ids = torch.load(args.token_file, weights_only=True)
    ids = token_ids[:, args.token_offset : args.token_offset + 128].cuda()
    if ids.shape != (1, 128):
        parser.error("need a complete 128-token window")
    positions = np.linspace(0, 127, 8, dtype=int).tolist()
    capture = CaptureNormInputs(len(model.backbone.layers), positions)
    with torch.no_grad():
        model_forward(model, ids, ops=capture, scan="chunked", output_logits=False)
    recipes = json.loads((args.recipes / "manifest.json").read_text())
    records, files = [], {}
    for record in recipes["operators"]:
        layer, site = record["layer"], record["site"]
        norm = (
            model.backbone.norm_f
            if layer == len(model.backbone.layers)
            else model.backbone.layers[layer].mixer.norm
            if site == "gated_rms_invsqrt"
            else model.backbone.layers[layer].norm
        )
        gamma = norm.weight.detach().double().cpu().numpy()
        width = len(gamma)
        eps = float(norm.variance_epsilon)
        lo, hi = record["interval"]
        # Eight deterministic cases: zero, small-energy dense, sparse at each
        # endpoint, alternating signs and nonuniform dense components.
        target_variances = [eps, 2 * eps, np.sqrt(eps * hi), hi, hi, 2 * eps, hi, hi]
        stress = np.zeros((8, width))
        for i, variance in enumerate(target_variances):
            if i in (2, 3, 5):
                stress[i, int(np.argmax(np.abs(gamma)))] = -1 if i % 2 else 1
            elif i in (4, 6):
                stress[i] = np.where(np.arange(width) % 2, -1.0, 1.0)
            else:
                stress[i] = np.sin(np.arange(width) * 0.73 + i) + 0.25
            energy = float(np.mean(stress[i] ** 2))
            stress[i] *= np.sqrt(max(0, variance - eps) / energy) * (1 - 1e-12)
        vectors = np.concatenate((capture.inputs[(layer, site)], stress))
        variance = np.mean(vectors**2, axis=1) + eps
        if not np.isfinite(vectors).all() or np.any(variance < lo) or np.any(variance > hi):
            raise ValueError(f"fixture variance outside frozen domain at {layer}:{site}")
        name = Path(record["file"]).stem + ".vectors.txt"
        lines = [f"fhemamba-vector-rms-v1 {width} {len(vectors)} {eps:.17g}"]
        lines.append(" ".join(f"{x:.17g}" for x in gamma))
        lines.extend(" ".join(f"{x:.17g}" for x in row) for row in vectors)
        content = "\n".join(lines) + "\n"
        files[name] = content
        records.append(
            {
                "file": name,
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
                "recipe_file": record["file"],
                "recipe_sha256": record["sha256"],
                "layer": layer,
                "site": site,
                "width": width,
                "cases": len(vectors),
                "checkpoint_cases": 8,
                "synthetic_cases": 8,
                "variance": variance.tolist(),
                "gamma_max_abs": float(np.max(np.abs(gamma))),
                "epsilon": eps,
            }
        )
    args.output_dir.mkdir(parents=True)
    for name, content in files.items():
        (args.output_dir / name).write_text(content)
    report = {
        "stage": "vector-rms-probe-inputs",
        "version": __version__,
        "repo_commit": current_git_commit(),
        "source_sha256": sha(Path(__file__)),
        "dependency_source_sha256": {
            name: sha(Path(__file__).resolve().parents[1] / "src/fhemamba" / name)
            for name in ("reference.py", "ops.py")
        },
        "checkpoint_sha256": {
            str(p): sha(p)
            for p in sorted(args.checkpoint.iterdir())
            if p.suffix in (".safetensors", ".json")
        },
        "token_file_sha256": sha(args.token_file),
        "token_offset": args.token_offset,
        "tokens": 128,
        "selected_token_positions": positions,
        "input_recipe_manifest_sha256": sha(args.recipes / "manifest.json"),
        "operators": records,
        "measurement_scope": {
            "artifact_level_report": True,
            "full_model_correctness_claimed": False,
            "encrypted_execution": False,
            "coefficient_tuning_on_inputs": False,
            "claim": "Exact-checkpoint inputs and public stress cases; frozen polynomial recipes.",
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"exported {len(records)} sites with 16 vectors each")


if __name__ == "__main__":
    main()
