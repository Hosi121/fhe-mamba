"""Transformers Mamba-1 checkpoints using the existing stateful CPU reference."""

from __future__ import annotations

from fhemamba.checkpoints import hf_checkpoint_identity

from .contracts import capabilities


class Mamba1Adapter:
    architecture = "mamba1"
    fhe = None
    checkpoint_identity = staticmethod(hf_checkpoint_identity)

    def capabilities(self, config):
        activation = config.get("hidden_act", "silu")
        unsupported = None
        if activation not in ("silu", "swish"):
            unsupported = (
                f"Mamba-1 reference requires SiLU; hidden_act={activation!r} is unsupported"
            )
        return capabilities(self.architecture, "transformers", self.fhe, unsupported=unsupported)

    def load(self, checkpoint):
        from fhemamba._env import block_broken_torchvision

        block_broken_torchvision()
        try:
            from transformers import MambaForCausalLM
        except ImportError as exc:
            raise ValueError("Mamba-1 requires pip install 'fhemamba[experiments]'") from exc
        return (
            MambaForCausalLM.from_pretrained(checkpoint, local_files_only=True).float().cpu().eval()
        )

    def vocab_size(self, model):
        return model.get_input_embeddings().num_embeddings

    def generate_cpu(self, model, ids, length, backend, prepared, manifest):
        import torch

        from fhemamba.inference import GenerationResult
        from fhemamba.m1_payload import _trace_from_ids

        if backend != "exact":
            raise ValueError("Mamba-1 currently implements exact CPU generation only")
        trace = _trace_from_ids(
            model,
            torch.tensor([ids]),
            generate_tokens=length,
            record_layer_details=False,
        )
        checks = {
            "finite_hidden_states": True,  # The trace rejects non-finite hidden/logit values.
            "complete_generation": len(trace.generated_ids) == length,
        }
        passed = all(checks.values())
        return GenerationResult(
            ids,
            trace.generated_ids,
            backend,
            passed,
            False,
            "length" if passed else "validation_failed",
            {
                "architecture": self.architecture,
                "checks": checks,
                "selection": "greedy",
                "stopping": "length",
            },
        )


ADAPTER = Mamba1Adapter()
