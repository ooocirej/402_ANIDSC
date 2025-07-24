import pickle
import torch
import pytest
from pathlib import Path

from ANIDSC.pipeline.pipeline import Pipeline
from ANIDSC.feature_extractor.frequency import FrequencyExtractor
from ANIDSC.model.torch_model.autoencoder import AE

@pytest.fixture(autouse=True)
def patch_mixins(tmp_path, monkeypatch):
    """
    Redirect all get_save_path() calls on the classes 
    to write under tmp_path, and ensure AE.custom_params exists.
    """
    # Make AE.state_dict() skip any custom_params logic
    monkeypatch.setattr(AE, "custom_params", [], raising=False)

    # Patch FrequencyExtractor.get_save_path on the class
    monkeypatch.setattr(
        FrequencyExtractor,
        "get_save_path",
        lambda self: str(tmp_path / "freq.pkl"),
        raising=False
    )

    # Patch AE.get_save_path on the class
    monkeypatch.setattr(
        AE,
        "get_save_path",
        lambda self: str(tmp_path / "ae.pth"),
        raising=False
    )

    return tmp_path

def test_frequency_save_load(tmp_path):
    freq = FrequencyExtractor(time_window=5)
    for t in range(3):
        freq.update({"timestamp": float(t)})

    # Save 
    freq.save()
    p = tmp_path / "freq.pkl"
    assert p.exists(), "Pickle mixin did not write freq.pkl"

    # Load 
    loaded = FrequencyExtractor.load(str(p))
    # Compare internal stats dict
    assert isinstance(loaded, FrequencyExtractor)
    assert loaded.state.__dict__ == freq.state.__dict__

def test_ae_save_load(tmp_path):
    ae = AE(input_dims=4, device="cpu")

    # Save
    ae.save()
    p = tmp_path / "ae.pth"
    assert p.exists(), "Torch mixin did not write ae.pth"

    # Load back
    loaded = AE.load(str(p))
    assert isinstance(loaded, AE)

    # Compare state_dicts
    orig = ae.state_dict()
    new  = loaded.state_dict()
    for k in orig:
        assert torch.equal(orig[k], new[k]), f"Mismatch at {k}"

def test_pipeline_save_load_state(tmp_path):
    # 1) Build a tiny pipeline by hand
    freq = FrequencyExtractor(time_window=10)
    ae   = AE(input_dims=1, latent_dim=4, device="cpu")
    pipeline = Pipeline(components={"freq": freq, "ae": ae})

    # 2) Seed some non-default state in the extractor
    for t in range(5):
        freq.update({"timestamp": float(t)})

    # 3) Save the entire pipeline under tmp_path/"cp"
    cp_dir = tmp_path / "cp"
    pipeline.save_state(cp_dir)

    # 4) Assert that the manifest and component folders exist
    assert (cp_dir / "pipeline_config.yaml").exists()
    assert (cp_dir / "freq"   / "state.pkl").exists()
    assert (cp_dir / "ae"     / "model.pth").exists()

    # 5) Load the pipeline back
    loaded = Pipeline.load_state(cp_dir)

    # 6) The wiring (to_dict) should match exactly
    assert loaded.to_dict() == pipeline.to_dict()

    # 7) The extractor’s running‐stats should match
    assert loaded.components["freq"].state.__dict__ == freq.state.__dict__

    # 8) The AE’s weights should match bit-for-bit
    orig_w = ae.state_dict()
    new_w  = loaded.components["ae"].state_dict()
    for k in orig_w:
        assert torch.equal(orig_w[k], new_w[k]), f"Weight mismatch at {k}"