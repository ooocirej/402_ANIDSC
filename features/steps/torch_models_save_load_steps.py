from behave import given, when, then
from pathlib import Path
from ANIDSC.model.model import BaseOnlineODModel
import importlib, tempfile
import torch

@given("a detection model component {model_name} with input_dims {ndim:d} on cpu")
def step_make_model(context, model_name, ndim):
    submod, cls_name = model_name.rsplit(".", 1)
    cls = getattr(importlib.import_module(f"ANIDSC.model.{submod}"), cls_name)

    # use CPU
    model = cls(input_dims=ndim, device="cpu").eval()

    comp = BaseOnlineODModel(model_name=model_name)
    comp.model = model
    comp.save_attr=["model_name"] # save for saving and loading (since its only the component by itself)

    context.model_name = model_name
    context.ndim = ndim
    context.comp = comp
    


@given("a fixed random input batch of size 5 and width {ndim:d}")
def step_fixed_input(context, ndim):
    torch.manual_seed(0)
    context.X = torch.randn(5, int(ndim))

@given("I briefly train the component for 5 steps")
def step_brief_train(context):
    context.comp.model.train()
    torch.manual_seed(0)
    for _ in range(int(5)):
        ret = context.comp.model.train_step(context.X)
        if isinstance(ret, tuple):
            _, loss = ret
        else:
            loss = ret
    context.comp.model.eval()


@when("I save the component to a temporary directory")
def step_save(context):
    context.tmpdir = Path(tempfile.mkdtemp(prefix="behave_comp_"))
    context.comp.save_state(context.tmpdir)


@when("I load the component back from that directory")
def step_load(context):
    context.comp_loaded = BaseOnlineODModel.load_state(context.tmpdir, model_name=context.model_name)
    context.comp.model.eval()
    context.comp_loaded.model.eval()


@then("the loaded component has the same parameters as the original")
def step_compare_state(context):
    sd1, sd2 = context.comp.model.state_dict(), context.comp_loaded.model.state_dict()
    assert sd1.keys() == sd2.keys(), f"keys differ: {sd1.keys() ^ sd2.keys()}"
    for k in sd1:
        a, b = sd1[k], sd2[k]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        else:
            assert a == b, f"mismatch at {k}: {a} != {b}"


# @then("the loaded component produces the same outputs as the original on the fixed input")
# def step_compare_outputs(context):

#     with torch.no_grad():
#         sig = inspect.signature(context.model.forward)
#         if "inference" in sig.parameters:
#             out1 = context.model.forward(context.X, inference=True)
#             out2 = context.model_loaded.forward(context.X, inference=True)
#         else:
#             torch.manual_seed(1234); out1 = context.model(context.X)
#             torch.manual_seed(1234); out2 = context.model_loaded(context.X)

#     if isinstance(out1, torch.Tensor) and isinstance(out2, torch.Tensor):
#         torch.testing.assert_close(out1, out2, rtol=1e-5, atol=1e-6)
#     elif isinstance(out1, (list, tuple)) and isinstance(out2, (list, tuple)):
#         assert len(out1) == len(out2)
#         for a, b in zip(out1, out2):
#             if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
#                 torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
#             else:
#                 assert a == b
#     else:
#         assert out1 == out2
