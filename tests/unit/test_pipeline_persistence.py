# tests/unit/test_pipeline_persistence.py

import pytest
from pathlib import Path

from ANIDSC.pipeline.pipeline import Pipeline
from ANIDSC.feature_extractor.frequency import FrequencyExtractor

@pytest.fixture
def freq_pipeline(tmp_path):
    # Build concrete extractor
    fe = FrequencyExtractor(time_window=5)
    # Mutate its internal state
    for ts in [0.0, 2.0, 4.5]:
        fe.update({"timestamp": ts})
    # Wrap in a Pipeline under the correct key
    p = Pipeline()
    p.components = {"feature_extractor": fe}
    fe.parent_pipeline = p
    return p

def test_frequency_extractor_persistence(tmp_path, freq_pipeline):
    pipeline = freq_pipeline
    outdir = tmp_path / "persist"

    # Persist everything
    pipeline.save_state(outdir)

    # On-disk check: folder for feature_extractor should exist
    fe_dir = outdir / "feature_extractor"
    assert fe_dir.is_dir()
    assert (fe_dir / "init_args.pkl").exists()
    assert (fe_dir / "state.pkl").exists()

    # Round-trip load
    loaded = Pipeline.load_state(outdir)
    fe2 = loaded.components["feature_extractor"]

    #  Check constructor args survived
    assert fe2.time_window == 5

    # Check the sliding-window deque survived
    assert list(fe2.state.sliding_window) == \
           list(pipeline.components["feature_extractor"].state.sliding_window)
