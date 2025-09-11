from pathlib import Path
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go

# ---- CONFIG: where your CSVs are ----
SCORES_ROOT = Path("tmp/scores")   # change if needed (can also be absolute)

# ---- helpers ----
def find_csvs(root: Path):
    files = sorted(root.rglob("*.csv"))
    opts = []
    for p in files:
        # label: fe_name/file  (fallback to filename)
        parts = p.relative_to(root).parts
        label = "/".join(parts[-2:]) if len(parts) >= 2 else p.name
        opts.append({"label": label, "value": str(p)})
    return opts

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # coerce numeric cols
    for c in ["batch_num", "score_median", "score_mean", "score_max", "threshold", "y_true", "y_pred"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None = None):
    n = len(y_pred)
    if y_true is None or y_true.size != n or np.isnan(y_true).all():
        # unsupervised: show counts only
        return {"n": n, "predicted_anomalies": int(np.nansum(y_pred))}
    y = y_true.astype(int)
    yp = y_pred.astype(int)
    tp = int(np.sum((y==1)&(yp==1)))
    tn = int(np.sum((y==0)&(yp==0)))
    fp = int(np.sum((y==0)&(yp==1)))
    fn = int(np.sum((y==1)&(yp==0)))
    acc = (tp+tn)/n if n else np.nan
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0
    out = {"n": n, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
           "tp": tp, "tn": tn, "fp": fp, "fn": fn}
    return out

def plot_timeseries(df: pd.DataFrame, score_col: str, use_override: bool, override_thr: float | None):
    x = df["batch_num"] if "batch_num" in df else np.arange(len(df))
    s = df[score_col]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=s, mode="lines", name=score_col))
    # threshold: series from CSV or horizontal override
    if use_override and override_thr is not None:
        fig.add_hline(y=float(override_thr), line_dash="dot", annotation_text=f"threshold={override_thr:.3f}")
        y_pred = (s >= override_thr).astype(int)
    else:
        thr_series = df["threshold"] if "threshold" in df.columns else pd.Series([np.nan]*len(df))
        fig.add_trace(go.Scatter(x=x, y=thr_series, mode="lines", name="threshold", line=dict(dash="dot")))
        # row-wise decision using CSV threshold (handles evolving threshold)
        y_pred = (s >= thr_series).astype(int) if "threshold" in df.columns else pd.Series([0]*len(df))
    # mark predicted anomalies
    idx = y_pred.fillna(0).astype(int).to_numpy().nonzero()[0]
    if idx.size:
        fig.add_trace(go.Scatter(x=x.iloc[idx] if hasattr(x, "iloc") else x[idx],
                                 y=s.iloc[idx] if hasattr(s, "iloc") else s[idx],
                                 mode="markers", name="predicted anomaly"))
    fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=360)
    return fig, y_pred

# ---- app ----
app = Dash(__name__)
app.title = "ANIDSC – CSV Visualizer"

csv_options = find_csvs(SCORES_ROOT)

app.layout = html.Div([
    html.H2("ANIDSC – Anomaly Score Visualizer (CSV)"),
    html.Div([
        html.Div([
            html.H4("Run A"),
            dcc.Dropdown(id="file-a", options=csv_options, placeholder="Pick CSV for Run A…"),
            dcc.RadioItems(
                id="score-col-a",
                options=[{"label": "score_median", "value": "score_median"},
                         {"label": "score_mean", "value": "score_mean"},
                         {"label": "score_max", "value": "score_max"}],
                value="score_median",
                inline=True
            ),
            html.Div(id="metrics-a")
        ], style={"flex":"1","padding":"8px","border":"1px solid #222","borderRadius":"8px","marginRight":"8px"}),
        html.Div([
            html.H4("Run B"),
            dcc.Dropdown(id="file-b", options=csv_options, placeholder="Pick CSV for Run B…"),
            dcc.RadioItems(
                id="score-col-b",
                options=[{"label": "score_median", "value": "score_median"},
                         {"label": "score_mean", "value": "score_mean"},
                         {"label": "score_max", "value": "score_max"}],
                value="score_median",
                inline=True
            ),
            html.Div(id="metrics-b")
        ], style={"flex":"1","padding":"8px","border":"1px solid #222","borderRadius":"8px"})
    ], style={"display":"flex","gap":"8px","marginBottom":"8px"}),

    html.Div([
        dcc.Checklist(
            id="override-toggle",
            options=[{"label": "Override threshold", "value": "ovr"}],
            value=[]
        ),
        dcc.Input(id="override-input", type="number", placeholder="e.g. 0.95", debounce=True, style={"marginLeft":"8px"}),
        html.Span(" (leave empty to use CSV thresholds)"),
    ], style={"margin":"6px 0 10px"}),

    html.Div([
        html.Div([dcc.Graph(id="fig-a")], style={"flex":"1","padding":"8px"}),
        html.Div([dcc.Graph(id="fig-b")], style={"flex":"1","padding":"8px"})
    ], style={"display":"flex","gap":"8px"}),

    html.H4("Comparison"),
    dash_table.DataTable(
        id="compare-table",
        columns=[{"name": c, "id": c} for c in ["metric","run_a","run_b","delta_b_minus_a"]],
        style_cell={"padding":"6px","textAlign":"left"},
        style_header={"fontWeight":"bold"})
], style={"fontFamily":"system-ui, sans-serif", "padding":"10px 14px"})

def _metrics_block(m: dict | None):
    if not m: return html.Div("No labels; showing anomaly count only." , style={"opacity":0.7})
    items = []
    for k in ["n","predicted_anomalies","accuracy","precision","recall","f1","tp","tn","fp","fn"]:
        if k in m and m[k] is not None:
            v = m[k]
            if isinstance(v, float): v = f"{v:.3f}"
            items.append(html.Div(f"{k}: {v}"))
    return html.Div(items, style={"fontSize":"14px","marginTop":"6px"})

@app.callback(
    Output("fig-a","figure"), Output("metrics-a","children"),
    Input("file-a","value"), Input("score-col-a","value"),
    Input("override-toggle","value"), Input("override-input","value")
)
def update_a(path, score_col, ovr_toggle, ovr_value):
    if not path: return go.Figure(), html.Div("Select a CSV.", style={"opacity":0.6})
    df = load_csv(path)
    use_override = ("ovr" in (ovr_toggle or [])) and (ovr_value is not None)
    fig, y_pred = plot_timeseries(df, score_col, use_override, ovr_value)
    y_true = df["y_true"].to_numpy() if "y_true" in df.columns else None
    m = compute_metrics(y_true, y_pred.to_numpy() if hasattr(y_pred, "to_numpy") else np.asarray(y_pred))
    return fig, _metrics_block(m)

@app.callback(
    Output("fig-b","figure"), Output("metrics-b","children"),
    Input("file-b","value"), Input("score-col-b","value"),
    Input("override-toggle","value"), Input("override-input","value")
)
def update_b(path, score_col, ovr_toggle, ovr_value):
    if not path: return go.Figure(), html.Div("Select a CSV.", style={"opacity":0.6})
    df = load_csv(path)
    use_override = ("ovr" in (ovr_toggle or [])) and (ovr_value is not None)
    fig, y_pred = plot_timeseries(df, score_col, use_override, ovr_value)
    y_true = df["y_true"].to_numpy() if "y_true" in df.columns else None
    m = compute_metrics(y_true, y_pred.to_numpy() if hasattr(y_pred, "to_numpy") else np.asarray(y_pred))
    return fig, _metrics_block(m)

@app.callback(
    Output("compare-table","data"),
    Input("metrics-a","children"), Input("metrics-b","children"),
    State("file-a","value"), State("file-b","value")
)
def compare(_, __, pa, pb):
    # Recompute metrics here to populate the table consistently
    rows=[]
    def extract(path):
        if not path: return None
        df = load_csv(path)
        y_true = df["y_true"].to_numpy() if "y_true" in df.columns else None
        # prefer CSV y_pred if present; otherwise recompute against CSV threshold
        if "y_pred" in df.columns:
            y_pred = df["y_pred"].to_numpy()
        else:
            thr = df["threshold"] if "threshold" in df.columns else np.nan
            y_pred = (df["score_median"] >= thr).astype(int).to_numpy()
        return compute_metrics(y_true, y_pred)
    ma, mb = extract(pa), extract(pb)
    def add(name, a, b):
        if a is None and b is None: return
        fa = None if a is None else (f"{a:.3f}" if isinstance(a,(int,float)) else a)
        fb = None if b is None else (f"{b:.3f}" if isinstance(b,(int,float)) else b)
        fd = None
        if isinstance(a,(int,float)) and isinstance(b,(int,float)):
            fd = f"{(b-a):+.3f}"
        rows.append({"metric": name, "run_a": fa, "run_b": fb, "delta_b_minus_a": fd})
    keys = ["n","predicted_anomalies","accuracy","precision","recall","f1","tp","tn","fp","fn"]
    for k in keys:
        add(k, ma.get(k) if ma else None, mb.get(k) if mb else None)
    return rows

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
