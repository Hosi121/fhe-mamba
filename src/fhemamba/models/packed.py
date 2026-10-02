"""Shared packed CKKS execution with fixed classical-128 parameters and gates."""

from fhemamba.benchmarks.io import file_sha256

# Settings from the recorded public-state-reuse B300 candidate. This selects
# parameters, not a qualification claim for a newly prepared input or binary.
_CKKS_OPTIONS = {
    "security": "128-classic",
    "security_digits": 4,
    "tolerance": 0.001,
    "planned_refresh": True,
    "bootstrap_passes": 2,
    "batch_refresh": True,
    "inplace_ops": True,
    "gpu_plaintext_ntt": True,
    "profile_evaluation": True,
    "naf_rotations": True,
    "reuse_dead_inputs": True,
    "direct_plaintext_upload": True,
    "compact_weights": True,
    "move_plaintext_coefficients": True,
    "borrow_plaintext_upload": True,
    "bsgs_routing_stages": True,
    "cache_plaintexts": True,
    "frontier_refresh": True,
    "s2c_first": True,
    "gpu_plaintext_rns": True,
    "hoist_rotations": True,
    "share_chebyshev": True,
    "prefetch_plaintexts": True,
    "prefetch_workers": 2,
    "gpu_addend_rns": True,
    "plaintext_cache_capacity": 2048,
    "merge_refresh_correction": True,
    "gpu_plaintext_fft": True,
    "frontier_live_limit": 256,
    "reuse_public_ciphertexts": True,
}


def run_ckks(prepared, manifest, binary, output, timeout, *, profile="classical-128"):
    from fhemamba.benchmarks.generation import validate_generation
    from fhemamba.benchmarks.packed import run
    from fhemamba.inference import GenerationResult

    record = run(binary, prepared.path, output, timeout=timeout, **_CKKS_OPTIONS)
    native_path = output / "native.json"
    native = {}
    error = None
    try:
        # Native failure records can contain non-finite measurements. Do not
        # copy those into our strict JSON result, or invent reference tokens.
        import json

        if native_path.exists():
            value = json.loads(native_path.read_text())
            if not isinstance(value, dict):
                raise ValueError("native result must be a JSON object")
            native = value
        prepared.manifest()  # Recheck every consumed payload byte after native execution.
        validation = validate_generation(prepared.path, output, security="128-classic")
        if file_sha256(prepared.path / "manifest.json") != prepared.manifest_sha256:
            raise ValueError("prepared manifest changed during execution")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        validation = None
        error = str(exc)
    actual = native.get("generated_token_ids", [])
    if not isinstance(actual, list) or any(type(v) is not int or v < 0 for v in actual):
        actual = []
        error = "native output contains invalid generated token IDs"
        validation = None
    passed = record["passed"] is True and validation is not None
    result = GenerationResult(
        input_ids=list(manifest["prompt_ids"]),
        generated_ids=actual,
        backend="ckks",
        passed=passed,
        encrypted=native.get("encrypted") is True,
        stop_reason="length" if passed else "validation_failed",
        report={
            "architecture": manifest["architecture"],
            "profile": profile,
            "manifest_sha256": prepared.manifest_sha256,
            "run_path": str(output),
            "timed_out": record["timed_out"],
            "validation": validation,
            "error": error,
            "client_server_process_separated": False,
        },
    )
    return result
