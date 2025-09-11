from pathlib import Path
import torch

from ANIDSC.model.node_encoder.encoder import LinearNodeEncoder


def test_roundtrip_state_dict(tmp_path):
    enc = LinearNodeEncoder(n_features=8, node_latent_dim=4, device="cpu")
    enc.setup()

    with torch.no_grad():
        enc.linear.weight.add_(0.123)
        if enc.linear.bias is not None:
            enc.linear.bias.add_(0.123)

    out = tmp_path / "node_encoder"
    enc.save_state(out)  # writes full_model.pt

    enc2 = LinearNodeEncoder.load_state(out)

    # keys should match and tensors be equal (within tolerance)
    sd1, sd2 = enc.state_dict(), enc2.state_dict()
    assert sd1.keys() == sd2.keys()
    for k in sd1:
        a, b = sd1[k], sd2[k]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        else:
            assert a == b

    assert (out / "full_model.pt").exists()

