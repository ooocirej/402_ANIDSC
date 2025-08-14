from behave import given, when, then
from pathlib import Path
import importlib, tempfile
import torch

@given("a torch model {model_name} with input_dims {ndim:d} on cpu")
def step_make_model(ctx, model_name, ndim):
    submod, cls_name = model_name.rsplit(".", 1)
    module = importlib.import_module(f"ANIDSC.model.{submod}")
    cls = getattr(module, cls_name)

    # Construct on CPU for test stability
    ctx.model = cls(input_dims=ndim, device="cpu")
    ctx.model.eval()
    ctx.ndim = ndim


@given("a fixed random input batch of size 5 and width {ndim:d}")
def step_fixed_input(ctx, ndim):
    torch.manual_seed(0)
    ctx.X = torch.randn(5, int(ndim))


@when("I save the model to a temporary directory")
def step_save(ctx):
    ctx.tmpdir = Path(tempfile.mkdtemp(prefix="behave_torch_models_"))
    (ctx.tmpdir).mkdir(parents=True, exist_ok=True)
    ctx.ckpt = ctx.tmpdir / "full_model.pt"
    torch.save(ctx.model, str(ctx.ckpt))
    assert ctx.ckpt.exists() and ctx.ckpt.stat().st_size > 0


@when("I load the model back from that directory")
def step_load(ctx):
    ctx.model_loaded = torch.load(str(ctx.ckpt), map_location="cpu")
    ctx.model_loaded.eval()


@then("the loaded model has the same parameters as the original")
def step_compare_state(ctx):
    sd1, sd2 = ctx.model.state_dict(), ctx.model_loaded.state_dict()
    assert sd1.keys() == sd2.keys(), f"keys differ: {sd1.keys() ^ sd2.keys()}"
    for k in sd1:
        a, b = sd1[k], sd2[k]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        else:
            assert a == b, f"mismatch at {k}: {a} != {b}"


@then("the loaded model produces the same outputs as the original on the fixed input")
def step_compare_outputs(ctx):

    with torch.no_grad():
        out1 = ctx.model(ctx.X)
        out2 = ctx.model_loaded(ctx.X)

    if isinstance(out1, torch.Tensor) and isinstance(out2, torch.Tensor):
        torch.testing.assert_close(out1, out2, rtol=1e-5, atol=1e-6)
    elif isinstance(out1, (list, tuple)) and isinstance(out2, (list, tuple)):
        assert len(out1) == len(out2)
        for a, b in zip(out1, out2):
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
            else:
                assert a == b
    else:
        assert out1 == out2
