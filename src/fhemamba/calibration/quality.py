"""Shared train-only calibration, closed-loop fitting and held-out PPL evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from fhemamba.ops import PolyOps, RangeRecorder, RecordingPolyOps, pool_by_name, union_ranges
from fhemamba.ppl import perplexity
from fhemamba.reference import model_forward


@dataclass
class QualityStudy:
    """One model and disjoint calibration/evaluation token streams.

    Every calibration window starts a fresh state, as does each PPL window.
    Recalibration widens the exact ranges once using the fitted polynomial.
    """

    model: object
    train_ids: object
    test_ids: object
    window: int = 1024
    cal_windows: int = 24
    max_windows: int = 40
    device: str = "cpu"

    def record(self, recorder=None):
        recorder = RangeRecorder() if recorder is None else recorder
        for index in range(self.cal_windows):
            chunk = self.train_ids[:, index * self.window : (index + 1) * self.window]
            model_forward(
                self.model, chunk.to(self.device), recorder, scan="chunked", output_logits=False
            )
        return recorder

    def evaluate(self, ops):
        return perplexity(
            lambda ids: model_forward(self.model, ids, ops, scan="chunked")["logits"],
            self.test_ids,
            window=self.window,
            max_windows=self.max_windows,
            device=self.device,
        )

    def fit(self, site_ranges, *, recalibrate=True, **options):
        ops = PolyOps.fit(
            ranges_by_name=pool_by_name(site_ranges), site_ranges=site_ranges, **options
        )
        if recalibrate:
            # Preserve the original recording circuit, including its default masks.
            probe = self.record(
                RecordingPolyOps(polys=ops.polys, enabled=ops.enabled, layer_polys=ops.layer_polys)
            )
            merged = union_ranges(site_ranges, probe.ranges)
            ops = PolyOps.fit(ranges_by_name=pool_by_name(merged), site_ranges=merged, **options)
        return ops
