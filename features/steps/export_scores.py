from behave import given, then
import math
import numpy as np
import pandas as pd
from pathlib import Path
from ANIDSC.evaluator.evaluator import BaseEvaluator 

def _tap_install(context):
    if context.__dict__.get("_tap_installed", False):
        return
    context._tap_installed = True
    context._pred_rows = []
    context._orig_eval_process = BaseEvaluator.process

    def tapped_process(self, results):
        out = context._orig_eval_process(self, results) 

        scores = results.get("score")
        thr    = results.get("threshold")
        bnum   = results.get("batch_num")

        if scores is not None and np.size(scores) > 0:
            s = np.asarray(scores, dtype=float).reshape(-1)
            sc_med  = float(np.nanmedian(s))
            sc_mean = float(np.nanmean(s))
            sc_max  = float(np.nanmax(s))
        else:
            sc_med = sc_mean = sc_max = None

        y_true = results.get("y_true") or results.get("label")
        y_pred = None
        if thr is not None and sc_med is not None and math.isfinite(sc_med):
            try:
                y_pred = int(sc_med >= float(thr))
            except Exception:
                y_pred = None

        context._pred_rows.append({
            "batch_num": bnum,
            "score_median": sc_med,
            "score_mean": sc_mean,
            "score_max": sc_max,
            "threshold": float(thr) if thr is not None else None,
            "y_true": int(y_true) if isinstance(y_true, (bool, int, np.integer)) else None,
            "y_pred": y_pred,
        })
        return out

    BaseEvaluator.process = tapped_process

def _tap_uninstall(context):
    if context.__dict__.get("_tap_installed", False):
        BaseEvaluator.process = context._orig_eval_process
        context._tap_installed = False

@given("enable anomaly score tap")
def step_enable_tap(context):
    _tap_install(context)

@then('export anomaly scores to "{out_path}"')
def step_export_scores(context, out_path):
    # expand placeholders so each Example writes its own CSV
    fe  = context.pipeline.request_attr("data_source", "fe_name")
    ds  = context.pipeline.request_attr("data_source", "dataset_name")
    fn  = context.pipeline.request_attr("data_source", "file_name")
    out_path = out_path.format(fe_name=fe, dataset=ds, file=fn, pipeline=str(context.pipeline))

    print("yes")
    _tap_uninstall(context)
    rows = context.__dict__.get("_pred_rows", [])
    df = pd.DataFrame(rows)
    p = Path(out_path); p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
