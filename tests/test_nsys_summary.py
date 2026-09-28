import importlib
import sqlite3

import pytest


@pytest.fixture
def profiler(monkeypatch):
    return importlib.import_module("fhemamba.profiling.nsys")


def test_timeline_union_handles_nested_graphs_streams_and_empty_intervals(profiler):
    assert list(profiler.merged_intervals([])) == []
    assert list(profiler.merged_intervals([(1, 9), (2, 3), (4, 7), (9, 11), (15, 16)])) == [
        (1, 11),
        (15, 16),
    ]
    with pytest.raises(ValueError, match="negative"):
        list(profiler.merged_intervals([(9, 1)]))


def test_summary_counts_overlapping_streams_once_and_keeps_idle_bins(profiler):
    with sqlite3.connect(":memory:") as conn:
        conn.executescript("""
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            INSERT INTO StringIds VALUES (1,'kernel'),(2,'cudaDeviceSynchronize');
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL
              (start INTEGER,end INTEGER,deviceId INTEGER,demangledName INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
              (1000000000,4000000000,0,1),(2000000000,6000000000,0,1),
              (9000000000,11000000000,0,1);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,nameId INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (0,12000000000,2);
        """)
        result = profiler.summarize(conn, bin_seconds=5)
        assert result["gpu_activity_union_seconds"] == 7
        assert result["activities"]["kernel"]["summed_seconds"] == 9
        assert result["gpu_activity_coverage"] == pytest.approx(7 / 12)
        assert result["longest_gpu_gap_seconds"] == 3
        assert result["gpu_gaps_at_least_1ms"] == {"count": 3, "seconds": 5}
        assert [b["gpu_activity_coverage"] for b in result["time_bins"]] == [0.8, 0.4, 0.5]
        assert result["cuda_api_union_seconds"] == 12
        assert result["gpu_idle_inside_cuda_api_seconds"] == 5
        assert result["gpu_idle_outside_cuda_api_seconds"] == 0
        late = profiler.summarize(conn, bin_seconds=5, start_seconds=3)
        assert late["trace_window_seconds"] == 9
        assert late["gpu_activity_union_seconds"] == 5
        assert late["activities"]["kernel"]["summed_seconds"] == 6
        assert late["cuda_api"][0]["summed_seconds"] == 9
        assert [b["gpu_activity_coverage"] for b in late["time_bins"]] == [0.6, 0.5]
        with pytest.raises(ValueError, match="positive time span"):
            profiler.summarize(conn, start_seconds=12)
        conn.executescript("""
            DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME;
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
              (0,2000000000,2),(5000000000,8000000000,2),(10000000000,12000000000,2);
        """)
        split = profiler.summarize(conn)
        assert split["cuda_api_union_seconds"] == 7
        assert split["gpu_and_cuda_api_overlap_seconds"] == 3
        assert split["gpu_idle_inside_cuda_api_seconds"] == 4
        assert split["gpu_idle_outside_cuda_api_seconds"] == 1
        conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0,100,1,1)")
        with pytest.raises(ValueError, match="multiple devices"):
            profiler.summarize(conn)
        with pytest.raises(ValueError, match="isolated single-device"):
            profiler.summarize(conn, device=0)
