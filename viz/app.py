from pathlib import Path
import json
import numpy as np
import pandas as pd
import time
from dash import Dash, dcc, html, Input, Output, State, no_update, MATCH, ctx
try:
    from dash import dash_table
except Exception:
    import dash_table
import plotly.graph_objects as go
import plotly.io as pio
from dash.dependencies import ALL

# Set Plotly theme
pio.templates.default = "plotly_white"

SCORES_ROOT = Path("tests/unit/AfterImage")
SCORES_ROOT = SCORES_ROOT if SCORES_ROOT.exists() else Path("tmp/scores")
SAMPLE_PERIOD_S = 10.0
CLICK_ZOOM_WINDOW_S = 600
CLICK_ZOOM_WINDOW_N = 1000
CLICK_Y_PADDING_FRAC = 0.05
MAX_POINTS = 10_000
RESULTS_ROOT = Path("tests/unit/AfterImage/results")

# Enhanced color scheme
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'score': 'rgba(102, 126, 234, 0.8)',
    'threshold': 'rgba(234, 102, 102, 0.8)',
    'moving_avg': 'rgba(102, 234, 126, 0.8)',
    'anomaly': '#ff4444',
    'background': '#f8f9fa',
    'card_bg': 'white',
    'text': '#495057',
    'text_light': '#6c757d'
}

# ---------- STYLES ----------
app_styles = {
    'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'minHeight': '100vh',
    'padding': '20px'
}

container_styles = {
    'maxWidth': '1400px',
    'margin': '0 auto',
    'background': 'rgba(255, 255, 255, 0.98)',
    'borderRadius': '20px',
    'boxShadow': '0 20px 60px rgba(0,0,0,0.3)',
    'overflow': 'hidden'
}

header_styles = {
    'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'color': 'white',
    'padding': '30px',
    'textAlign': 'center'
}

control_panel_styles = {
    'padding': '30px',
    'background': '#f8f9fa',
    'borderBottom': '1px solid #dee2e6'
}

stat_card_styles = {
    'background': 'white',
    'padding': '20px',
    'borderRadius': '12px',
    'boxShadow': '0 2px 10px rgba(0,0,0,0.08)',
    'transition': 'all 0.3s',
    'cursor': 'default'
}

button_styles = {
    'padding': '10px 20px',
    'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'color': 'white',
    'border': 'none',
    'borderRadius': '8px',
    'fontWeight': '600',
    'cursor': 'pointer',
    'boxShadow': '0 4px 6px rgba(102, 126, 234, 0.2)',
    'marginLeft': '10px'
}

dropdown_styles = {
    'borderRadius': '8px',
    'border': '2px solid #dee2e6'
}

graph_card_styles = {
    'background': 'white',
    'padding': '20px',
    'borderRadius': '12px',
    'boxShadow': '0 2px 10px rgba(0,0,0,0.08)',
    'marginBottom': '20px'
}

# ---------- HELPERS ----------
def find_csvs(root: Path):
    """
    Expect paths like:
      results/<dataset>/<feature>/<file>/<pipeline>/<csv_file>.csv
    Label becomes: "<dataset>/<feature>/<file> — <pipeline>"
    Falls back to a reasonable label if structure differs.
    """
    files = sorted(root.rglob("*.csv"))
    opts = []

    for p in files:
        label = p.name
        try:
            rel = p.relative_to(root)
            parts = rel.parts  # includes the filename at the end

            # Try to parse the last 5 parts: ds/feat/file/pipeline/file.csv
            if len(parts) >= 5:
                dataset, feature, file_name, pipeline = parts[-5], parts[-4], parts[-3], parts[-2]
                label = f"{dataset}/{feature}/{file_name} — {pipeline}"
            elif len(parts) >= 3:
                # Fallback: <parent>/<parent>/<file.csv>
                label = "/".join(parts[-3:-1])  # two dirs
            else:
                label = parts[-1]  # just the filename
        except Exception:
            # Final fallback if relative_to fails
            label = p.name

        opts.append({"label": label, "value": str(p)})

    return opts

def build_results_tree(root: Path) -> dict:
    """
    Directories => dict of {child_name: child_node}
    CSV files  => string full path (leaf)
    Supports arbitrary nesting.
    """
    tree: dict = {}

    # include empty dirs so user can browse them
    for d in root.rglob("*"):
        if d.is_dir():
            rel = d.relative_to(root).parts
            if not rel:
                continue
            cur = tree
            for part in rel:
                cur = cur.setdefault(part, {})  # ensure dict nodes

    # add CSV leaves
    for csv_path in root.rglob("*.csv"):
        rel = csv_path.relative_to(root).parts
        cur = tree
        for part in rel[:-1]:
            cur = cur.setdefault(part, {})
        cur[rel[-1]] = str(csv_path)  # leaf is a string
    return tree

TREE = build_results_tree(RESULTS_ROOT)

def node_from_parts(tree: dict, parts: list[str]):
    cur = tree
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur

def children_options(node) -> list[dict]:
    """ Show folders first, then CSVs (alphabetically). """
    if not isinstance(node, dict):
        return []  # leaf
    folders = sorted([k for k, v in node.items() if isinstance(v, dict)])
    csvs    = sorted([k for k, v in node.items() if isinstance(v, str)])
    names = folders + csvs
    return [{"label": n, "value": n} for n in names]

def is_leaf(node) -> bool:
    return isinstance(node, str)

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_cols = [
        "time","batch_num","median_score","median_threshold",
        "lower_quartile_score","upper_quartile_score",
        "soft_min_score","soft_max_score",
        "detection_rate","y_true","y_pred","threshold"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in df.columns:
        if c.startswith("score_"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for th in ("median_threshold","threshold"):
        if th in df.columns:
            df[th] = df[th].replace([np.inf, -np.inf], np.nan)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def list_score_cols(df: pd.DataFrame) -> list:
    """Return only the main score columns for selection"""
    # Only allow selection of median_score and detection_rate
    selectable = ["median_score", "detection_rate"]
    scores = [c for c in selectable if c in df.columns]
    
    # If neither exists, try to find any score column as fallback
    if not scores:
        scores = [c for c in df.columns if c.startswith("score_")][:1]
    if not scores:
        scores = [c for c in df.columns if c.lower() in ("score", "scores")][:1]
    
    return scores if scores else ["median_score"]

def score_options_for(path: str, default: str | None = None):
    try:
        head = pd.read_csv(path, nrows=1)
        cols = list_score_cols(head)
    except Exception:
        cols = []
    if not cols:
        cols = ["median_score"]
    opts = [{"label": c, "value": c} for c in cols]
    value = default if (default and default in cols) else cols[0]
    return opts, value

def get_threshold_series(df: pd.DataFrame) -> pd.Series:
    if "median_threshold" in df.columns:
        return pd.to_numeric(df["median_threshold"], errors="coerce")
    if "threshold" in df.columns:
        return pd.to_numeric(df["threshold"], errors="coerce")
    return pd.Series([np.nan] * len(df), index=df.index)

def x_vals(df: pd.DataFrame):
    return df["batch_num"] if "batch_num" in df.columns else pd.Series(range(len(df)))

def _to_datetime_from_seconds(series):
    return pd.to_datetime(pd.to_numeric(series, errors="coerce"), unit="s", utc=True)

def time_to_datetime_series(df: pd.DataFrame):
    if "time" in df.columns:
        t = df["time"]
        if np.issubdtype(t.dtype, np.number):
            tt = pd.to_numeric(t, errors="coerce")
            finite = tt.dropna()
            if finite.size:
                mx = float(finite.max())
                if mx > 1e12:
                    return pd.to_datetime(tt, unit="ms", utc=True), True
                if mx > 1e9:
                    return pd.to_datetime(tt, unit="s",  utc=True), True
                elapsed = tt.fillna(0).clip(lower=0)
                cum_s = elapsed.cumsum()
                return _to_datetime_from_seconds(cum_s), True
        dt = pd.to_datetime(t, utc=True, errors="coerce")
        if dt.notna().any():
            return dt, True
    if "batch_num" in df.columns:
        secs = pd.to_numeric(df["batch_num"], errors="coerce") * float(SAMPLE_PERIOD_S)
        return _to_datetime_from_seconds(secs), True
    return pd.Series(range(len(df))), False

def compute_metrics(y_true: np.ndarray | None, y_pred: np.ndarray):
    n = len(y_pred)
    if y_true is None or y_true.size != n or np.isnan(y_true).all():
        return {"n": n, "predicted_anomalies": int(np.nansum(y_pred))}
    y = y_true.astype(int); yp = y_pred.astype(int)
    tp = int(np.sum((y==1)&(yp==1))); tn = int(np.sum((y==0)&(yp==0)))
    fp = int(np.sum((y==0)&(yp==1))); fn = int(np.sum((y==1)&(yp==0)))
    acc = (tp+tn)/n if n else np.nan
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec  = tp/(tp+fn) if (tp+fn) else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0
    return {"n":n,"accuracy":acc,"precision":prec,"recall":rec,"f1":f1,"tp":tp,"tn":tn,"fp":fp,"fn":fn}

def _downsample_indices_len_safe(n: int, keep_idxs, max_points: int = MAX_POINTS) -> np.ndarray:
    """Return positional indices (<='max_points'). Always keep `keep_idxs` (positional),
    then fill the rest uniformly. Works even if n <= max_points."""
    if n <= max_points:
        return np.arange(n, dtype=int)

    # Ensure keep_idxs are valid positional indices
    keep = np.unique(np.clip(np.asarray(list(keep_idxs), dtype=int), 0, n - 1))
    
    # If we already have too many important points, we need to prioritize them
    if keep.size >= max_points:
        # Sort the keep indices and take evenly spaced ones
        # This ensures we at least keep a representative sample of the important points
        step = max(1, keep.size // max_points)
        selected = keep[::step]
        # Always include first and last of the important points
        if len(selected) < max_points and len(keep) > 0:
            selected = np.unique(np.concatenate([
                selected,
                [keep[0], keep[-1]]
            ]))
        return selected[:max_points]

    # Calculate how many additional points we can add
    remaining_budget = max_points - keep.size
    
    # Create a mask to exclude already selected points
    mask = np.ones(n, dtype=bool)
    mask[keep] = False
    available_indices = np.where(mask)[0]
    
    if len(available_indices) == 0:
        return keep
    
    # Calculate optimal number of additional points to sample
    if len(available_indices) <= remaining_budget:
        # We can include all available points
        additional = available_indices
    else:
        # Sample uniformly from available indices
        # Use linspace to get evenly distributed indices
        sample_indices = np.linspace(0, len(available_indices) - 1, remaining_budget, dtype=int)
        additional = available_indices[sample_indices]
    
    # Combine and sort
    idx = np.unique(np.concatenate([keep, additional]))
    
    # Final safety check - should not happen with corrected logic
    if idx.size > max_points:
        # This time, we preserve the original keep points as much as possible
        # by ensuring they appear in the final selection
        keep_set = set(keep[:min(len(keep), max_points // 2)])  # Reserve half for important points
        remaining_budget = max_points - len(keep_set)
        
        # Get other indices not in keep_set
        other_indices = np.array([i for i in idx if i not in keep_set])
        if len(other_indices) > remaining_budget:
            step = len(other_indices) / remaining_budget
            selected_others = other_indices[np.floor(np.arange(remaining_budget) * step).astype(int)]
            idx = np.unique(np.concatenate([list(keep_set), selected_others]))
        
    return np.sort(idx)[:max_points]

def summarize_run(df: pd.DataFrame, score_col: str):
    if score_col not in df.columns:
        scores = list_score_cols(df)
        s = pd.Series([np.nan]*len(df)) if not scores else pd.to_numeric(df[scores[0]], errors="coerce")
        score_col = scores[0] if scores else score_col
    else:
        s = pd.to_numeric(df[score_col], errors="coerce")
    thr_used = get_threshold_series(df)
    y_pred = (s >= thr_used).astype(int) if thr_used.notna().any() else pd.Series([0]*len(df))
    n = int(s.notna().sum())
    n_anom = int(y_pred.fillna(0).sum())
    frac_anom = (n_anom/n) if n else 0.0
    first_anom = int(df.loc[y_pred.fillna(0).astype(bool), "batch_num"].iloc[0]) if "batch_num" in df and (y_pred.fillna(0)>0).any() else None
    peak_idx = int(s.idxmax()) if n else None
    peak_batch = int(df.loc[peak_idx, "batch_num"]) if peak_idx is not None and "batch_num" in df else None
    peak_score = float(s.max()) if n else None
    thr_median = float(np.nanmedian(thr_used)) if len(thr_used) else None
    segments=[]
    if n:
        mask = y_pred.fillna(0).astype(int).to_numpy()
        xv = (df["batch_num"].to_numpy() if "batch_num" in df.columns else np.arange(len(df)))
        in_seg=False; start=None
        for i, flag in enumerate(mask):
            if flag and not in_seg:
                in_seg=True; start=i
            if in_seg and (i==len(mask)-1 or not mask[i+1]):
                end=i
                segments.append({"start_batch": int(xv[start]), "end_batch": int(xv[end]), "length": int(end-start+1)})
                in_seg=False
    top_segs = sorted(segments, key=lambda d: d["length"], reverse=True)[:3]
    y_true = pd.to_numeric(df.get("y_true"), errors="coerce") if "y_true" in df.columns else None
    label_metrics = None
    if y_true is not None and y_true.notna().any():
        m = compute_metrics(y_true.fillna(0).to_numpy(), y_pred.fillna(0).to_numpy())
        label_metrics = m
    return {
        "n": n,
        "score_mean": float(s.mean()) if n else None,
        "score_median": float(s.median()) if n else None,
        "score_std": float(s.std()) if n else None,
        "n_anomalies": n_anom,
        "frac_anomalies": frac_anom,
        "first_anomaly_batch": first_anom,
        "peak_score": peak_score,
        "peak_batch": peak_batch,
        "threshold_median": thr_median,
        "top_segments": top_segs,
        "label_metrics": label_metrics,
    }, y_pred

def _metrics_block(m: dict | None):
    if not m:
        return html.Div("No labels; showing anomaly count only.", style={"opacity":0.7, "color": COLORS['text_light']})
    items=[]
    for k in ["n","predicted_anomalies","accuracy","precision","recall","f1","tp","tn","fp","fn"]:
        if k in m and m[k] is not None:
            v = m[k]; v = f"{v:.3f}" if isinstance(v, float) else v
            items.append(html.Div(f"{k}: {v}"))
    return html.Div(items, style={"fontSize":"14px","marginTop":"6px", "color": COLORS['text']})

def format_summary(s: dict):
    def f(x): return "" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
    
    # Create stat cards with only desired metrics
    stats_grid = html.Div([
        create_stat_card("Mean Score", f(s["score_mean"])),
        create_stat_card("Median Score", f(s["score_median"])),
        create_stat_card("Anomaly Rate", f"{s['frac_anomalies']*100:.2f}%" if s['frac_anomalies'] else "0%"),
    ], style={
        'display': 'grid',
        'gridTemplateColumns': 'repeat(auto-fit, minmax(150px, 1fr))',
        'gap': '15px',
        'marginBottom': '20px'
    })
    
    blocks = [stats_grid]
    
    if s.get("label_metrics"):
        lm = s["label_metrics"]
        label_grid = html.Div([
            html.H4("Label Metrics", style={'color': COLORS['primary'], 'marginBottom': '15px'}),
            html.Div([
                create_stat_card("Accuracy", f"{lm['accuracy']:.3f}"),
                create_stat_card("Precision", f"{lm['precision']:.3f}"),
                create_stat_card("Recall", f"{lm['recall']:.3f}"),
                create_stat_card("F1 Score", f"{lm['f1']:.3f}"),
            ], style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(auto-fit, minmax(150px, 1fr))',
                'gap': '15px'
            })
        ])
        blocks.append(label_grid)
    
    return html.Div(blocks)

def create_stat_card(label, value):
    return html.Div([
        html.Div(label, style={
            'fontSize': '0.85em',
            'color': COLORS['text_light'],
            'marginBottom': '5px',
            'textTransform': 'uppercase',
            'letterSpacing': '0.5px'
        }),
        html.Div(value, style={
            'fontSize': '1.5em',
            'fontWeight': 'bold',
            'color': COLORS['text']
        })
    ], style={**stat_card_styles, 'textAlign': 'center'})

def plot_timeseries(df: pd.DataFrame, score_col: str, axis_mode="time", render_mode="gl"):
    if score_col not in df.columns:
        scores = list_score_cols(df)
        if not scores:
            return go.Figure(), pd.Series([0]*len(df))
        score_col = scores[0]

    # full-res series
    s = pd.to_numeric(df[score_col], errors="coerce")
    x, is_dt = get_x_series(df, axis_mode)
    thr = get_threshold_series(df)

    # predicted anomalies on full-res (positional)
    if thr.notna().any():
        y_pred_full = (s >= thr).astype(int)
        anom_pos = y_pred_full.fillna(0).to_numpy().nonzero()[0]
    else:
        y_pred_full = pd.Series([0]*len(df), index=s.index)
        anom_pos = np.array([], dtype=int)

    # preserve anomalies + first/last + argmin/argmax by POSITION (not labels)
    n = len(s)
    preserve = set(anom_pos.tolist())
    if n:
        preserve.update([0, n - 1])
        svals = s.to_numpy(copy=False)
        try:
            if np.isfinite(np.nanmax(svals)):
                preserve.add(int(np.nanargmax(svals)))
            if np.isfinite(np.nanmin(svals)):
                preserve.add(int(np.nanargmin(svals)))
        except Exception:
            pass

    # pick indices only for plotting
    idx = _downsample_indices_len_safe(n, preserve, max_points=MAX_POINTS)

    # strict positional slicing to avoid label/pos mismatches
    def take_pos(series_like):
        if isinstance(series_like, pd.Series) or isinstance(series_like, pd.Index):
            return series_like.iloc[idx]
        if isinstance(series_like, np.ndarray):
            return series_like[idx]
        # fallback: wrap so we can .iloc
        return pd.Series(series_like).iloc[idx]

    sx   = take_pos(s)
    xx   = take_pos(x)
    thrx = take_pos(thr) if thr.notna().any() else thr

    # recompute anomalies for the downsampled series (for markers only)
    if thr.notna().any():
        yp_plot = (sx >= thrx).astype(int)
        anom_idx_plot = yp_plot.fillna(0).to_numpy().nonzero()[0]
    else:
        anom_idx_plot = np.array([], dtype=int)

    TraceType = go.Scattergl if render_mode == "gl" else go.Scatter
    fig = go.Figure()

    # quartile band (downsampled safely if present)
    if "upper_quartile_score" in df.columns and "lower_quartile_score" in df.columns:
        uq = pd.to_numeric(df["upper_quartile_score"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        lq = pd.to_numeric(df["lower_quartile_score"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        fig.add_trace(TraceType(x=xx, y=uq, mode="lines", name="Upper Quartile",
                                line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(TraceType(x=xx, y=lq, mode="lines", name="Quartile Range",
                                line=dict(width=0), fill="tonexty",
                                fillcolor="rgba(102, 126, 234, 0.1)", hoverinfo="skip"))

    # main score
    fig.add_trace(TraceType(
        x=xx, y=sx, mode="lines", name=score_col,
        line=dict(color=COLORS['score'], width=2),
        hovertemplate='<b>Score:</b> %{y:.4f}<extra></extra>'
    ))

    # threshold
    if thr.notna().any():
        fig.add_trace(TraceType(
            x=xx, y=thrx, mode="lines", name="Threshold",
            line=dict(color=COLORS['threshold'], width=2.5, dash="dash"),
            hovertemplate='<b>Threshold:</b> %{y:.4f}<extra></extra>'
        ))

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=500,
        plot_bgcolor='rgba(248, 249, 250, 0.8)',
        paper_bgcolor='white',
        font=dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'),
        hovermode='x unified',
        hoverlabel=dict(bgcolor="white", font_size=12),
        xaxis=dict(
            showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)',
            title=dict(text="Batch Number" if axis_mode == "batch" else "Time", font=dict(size=14))
        ),
        yaxis=dict(
            showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)',
            title=dict(text="Score Value", font=dict(size=14))
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.8)", bordercolor="rgba(0,0,0,0.1)", borderwidth=1
        )
    )

    apply_xaxis_mode(fig, is_dt, axis_mode)
    return fig, y_pred_full

def apply_xaxis_mode(fig, is_dt: bool, axis_mode: str):
    if axis_mode == "time" and is_dt:
        fig.update_xaxes(
            type="date",
            tickformat="%H:%M:%S",
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=3, label="3h", step="hour", stepmode="backward"),
                    dict(count=6, label="6h", step="hour", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor="white",
                activecolor=COLORS['primary'],
                font=dict(size=11)
            ),
            rangeslider=dict(visible=False)
        )
    else:
        fig.update_xaxes(rangeslider=dict(visible=False))

def apply_click_zoom(fig, clickData, df: pd.DataFrame, axis_mode: str,
                     time_window_s: int = CLICK_ZOOM_WINDOW_S,
                     batch_window_n: int = CLICK_ZOOM_WINDOW_N):
    if not clickData or "points" not in clickData or not clickData["points"]:
        return
    x_clicked = clickData["points"][0].get("x")
    if x_clicked is None:
        return

    if axis_mode == "time":
        xc = pd.to_datetime(x_clicked, utc=True, errors="coerce")
        if pd.isna(xc):
            return
        half = pd.Timedelta(seconds=time_window_s/2)
        x0, x1 = xc - half, xc + half
        fig.update_xaxes(autorange=False, range=[x0, x1])
        x_series, is_dt = time_to_datetime_series(df)
        if not is_dt:
            return
        mask = (x_series >= x0) & (x_series <= x1)
    else:
        try:
            xc = float(x_clicked)
        except Exception:
            return
        x0, x1 = xc - batch_window_n/2, xc + batch_window_n/2
        fig.update_xaxes(autorange=False, range=[x0, x1])
        if "batch_num" in df.columns and df["batch_num"].notna().any():
            mask = (df["batch_num"] >= x0) & (df["batch_num"] <= x1)
        else:
            idx = pd.Series(range(len(df)))
            mask = (idx >= x0) & (idx <= x1)

    try:
        ys = None
        for tr in fig.data:
            if getattr(tr, "mode", "").startswith("lines"):
                ys = pd.to_numeric(pd.Series(tr.y), errors="coerce")
                break
        if ys is not None:
            sub = ys[mask.fillna(False)]
            if sub.notna().any():
                ymin, ymax = float(sub.min()), float(sub.max())
                if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
                    pad = CLICK_Y_PADDING_FRAC * max(1e-12, ymax - ymin)
                    fig.update_yaxes(range=[ymin - pad, ymax + pad], autorange=False)
    except Exception:
        pass

def filter_visible(df: pd.DataFrame, relayout: dict | None, axis_mode: str) -> pd.DataFrame:
    if not relayout: 
        return df
    if relayout.get("xaxis.autorange") is True or relayout.get("yaxis.autorange") is True: 
        return df
    x0 = relayout.get("xaxis.range[0]"); x1 = relayout.get("xaxis.range[1]")
    if x0 is None or x1 is None: return df
    if axis_mode == "time":
        try:
            x0v = pd.to_datetime(x0, utc=True); x1v = pd.to_datetime(x1, utc=True)
            xt, is_dt = time_to_datetime_series(df)
            if is_dt:
                mask = (xt >= x0v) & (xt <= x1v)
                return df[mask.fillna(False)]
        except Exception:
            return df
        return df
    if "batch_num" in df.columns and df["batch_num"].notna().any():
        return df[(df["batch_num"] >= float(x0)) & (df["batch_num"] <= float(x1))]
    i0, i1 = int(float(x0)), int(float(x1))
    i0, i1 = max(i0, 0), min(i1, len(df)-1)
    return df.iloc[i0:i1+1]

def get_x_series(df: pd.DataFrame, axis_mode: str):
    if axis_mode == "time":
        return time_to_datetime_series(df)
    return x_vals(df), False

def make_run_card(i, _csv_options_unused):
    return html.Div([
        html.H4(f"Run {i}", style={'color': COLORS['primary'], 'marginBottom': '20px'}),

        html.Div(id={"type": "picker", "index": i}),
        dcc.Store(id={"type": "selection", "index": i}, data=[], storage_type="memory"),
        dcc.Store(id={"type": "resolved-path", "index": i}, data=None, storage_type="memory"),

        # score column picker
        html.Div([
            html.Label("Score Column:", style={'fontWeight': '600', 'color': COLORS['text'], 'marginRight': '10px'}),
            dcc.RadioItems(
                id={"type":"score-col","index":i},
                options=[{"label":"median_score","value":"median_score"}],
                value="median_score",
                inline=True,
                style={'marginBottom': '15px'},
                persistence=True,
                persistence_type="memory",
                labelStyle={'marginRight': '15px', 'color': COLORS['text']}
            ),
        ]),

        dcc.Loading(
            id={"type":"loading-metrics","index":i},
            type="circle",
            color=COLORS['primary'],
            children=[
                html.Div(id={"type":"metrics","index":i}),
                html.Div(id={"type":"summary","index":i}, style={"marginTop":"20px"}),
            ]
        ),
        dcc.Store(id={"type":"zoom","index":i}, data={"tick": 0}, storage_type="memory"),
        dcc.Loading(
            id={"type":"loading-graph","index":i},
            type="circle",
            color=COLORS['primary'],
            children=[
                dcc.Graph(
                    id={"type":"fig","index":i},
                    config={
                        "scrollZoom": True,
                        "displaylogo": False,
                        "modeBarButtonsToAdd": ["select2","lasso2"],
                        "doubleClick": "reset+autosize",
                        "toImageButtonOptions": {
                            "format": "png",
                            "filename": "anidsc_plot",
                            "height": 600,
                            "width": 1200,
                            "scale": 2
                        }
                    },
                    style={"marginTop":"20px"}
                ),
            ]
        ),
    ], style={**graph_card_styles, 'marginBottom': '20px'}, key=f"card-{i}")

# ---------- APP ----------
app = Dash(__name__)
app.title = "ANIDSC Visualizer"
csv_options = find_csvs(SCORES_ROOT)

app.layout = html.Div([
    # Outer container with gradient background
    html.Div([
        # Main container
        html.Div([
            # Header
            html.Div([
                html.H1(" ANIDSC Results Visualizer", 
                       style={'fontSize': '2.5em', 'marginBottom': '10px', 'textShadow': '2px 2px 4px rgba(0,0,0,0.2)'}),
                html.Div("Interactive Anomaly Detection Analysis Dashboard", 
                        style={'opacity': 0.95, 'fontSize': '1.1em'})
            ], style=header_styles),
            
            # Control Panel
            html.Div([
                html.Div([
                    html.Div([
                        html.Label("X-Axis Mode", style={'fontWeight': '600', 'color': COLORS['text'], 'marginBottom': '5px'}),
                        dcc.RadioItems(
                            id="x-axis-mode",
                            options=[
                                {"label": " Over Time", "value": "time"},
                                {"label": " Per Batch", "value": "batch"}
                            ],
                            value="batch",
                            inline=True,
                            labelStyle={'marginRight': '20px', 'cursor': 'pointer'}
                        )
                    ], style={'marginRight': '30px', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.Label("Render Mode", style={'fontWeight': '600', 'color': COLORS['text'], 'marginBottom': '5px'}),
                        dcc.RadioItems(
                            id="render-mode",
                            options=[
                                {"label": " GPU (WebGL)", "value": "gl"},
                                {"label": " CPU (SVG)", "value": "svg"}
                            ],
                            value="gl",
                            inline=True,
                            labelStyle={'marginRight': '20px', 'cursor': 'pointer'}
                        )
                    ], style={'marginRight': '30px', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.Button("➕ Add Graph", id="add-graph", n_clicks=0, style=button_styles),
                        html.Button("➖ Remove Last", id="remove-graph", n_clicks=0, style=button_styles),
                    ], style={'display': 'inline-block', 'float': 'right'})
                ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'})
            ], style=control_panel_styles),
            
            # Store for graph sequence
            dcc.Store(id="graph-id-seq", data=1),
            
            # Graphs container
            dcc.Loading(
                id="graphs-loading",
                type="circle",
                color=COLORS['primary'],
                children=html.Div(id="graphs-container", style={'padding': '30px'})
            )
            
        ], style=container_styles)
    ], style=app_styles)
])

# ---------- CALLBACKS ----------

def _level_id(i, lvl):
    return {"type": "level", "index": i, "lvl": lvl}

@app.callback(
    Output({"type": "picker", "index": MATCH}, "children"),
    Input({"type": "selection", "index": MATCH}, "data"),
    prevent_initial_call=False
)
def render_picker(selection):
    selection = selection or []
    i = ctx.triggered_id["index"]
    children = []

    cur = TREE
    depth = 0

    # render already chosen levels
    for chosen in selection:
        opts = children_options(cur)
        children.append(
            dcc.Dropdown(
                id=_level_id(i, depth),
                options=opts,
                value=chosen,
                placeholder=f"Level {depth+1}...",
                style={**dropdown_styles, 'marginBottom': '10px'},
                persistence=True, persistence_type="memory"
            )
        )
        cur = (cur or {}).get(chosen)
        if cur is None:
            break
        depth += 1

    # render the next level if current node is a folder
    if cur is not None and not is_leaf(cur):
        children.append(
            dcc.Dropdown(
                id=_level_id(i, depth),
                options=children_options(cur),
                value=None,
                placeholder=f"Level {depth+1}...",
                style={**dropdown_styles, 'marginBottom': '10px'},
                persistence=True, persistence_type="memory"
            )
        )
    return children

@app.callback(
    Output({"type": "selection", "index": MATCH}, "data"),
    Output({"type": "resolved-path", "index": MATCH}, "data"),
    Input({"type": "level", "index": MATCH, "lvl": ALL}, "value"),
    State({"type": "level", "index": MATCH, "lvl": ALL}, "id"),
    prevent_initial_call=False
)
def on_level_change(values, ids):
    if not ids:
        return [], None
    pairs = sorted(zip(ids, values), key=lambda x: x[0]["lvl"])
    chosen = []
    for _, v in pairs:
        if v:
            chosen.append(v)
        else:
            break
    node = node_from_parts(TREE, chosen) if chosen else TREE
    resolved = node if is_leaf(node) else None  # leaf node is the CSV path string
    return chosen, resolved

@app.callback(
    Output("graphs-container","children"),
    Input("add-graph","n_clicks"),
    Input("remove-graph","n_clicks"),
    State("graphs-container","children"),
    State("graph-id-seq","data"),
    prevent_initial_call=False
)
def manage_cards_directly(n_add, n_remove, current_children, seq):
    trigger = ctx.triggered_id
    
    if current_children is None:
        return [make_run_card(0, csv_options)]
    
    current_children = list(current_children or [])
    
    if trigger == "add-graph":
        if len(current_children) < 5:
            new_id = seq if seq else len(current_children)
            new_card = make_run_card(new_id, csv_options)
            current_children.append(new_card)
    
    elif trigger == "remove-graph" and len(current_children) > 1:
        current_children.pop()
    
    return current_children

@app.callback(
    Output("graph-id-seq","data"),
    Input("add-graph","n_clicks"),
    State("graph-id-seq","data"),
    prevent_initial_call=True
)
def increment_seq(n_add, seq):
    return (seq or 1) + 1

@app.callback(
    Output({"type":"score-col","index":MATCH}, "options"),
    Output({"type":"score-col","index":MATCH}, "value"),
    Input({"type":"resolved-path","index":MATCH}, "data")
)
def refresh_scores_dynamic(resolved_csv):
    if not resolved_csv:
        return [{"label":"median_score","value":"median_score"}], "median_score"
    return score_options_for(resolved_csv, default="median_score")

@app.callback(
    Output({"type":"fig","index":MATCH}, "figure"),
    Output({"type":"metrics","index":MATCH}, "children"),
    Output({"type":"summary","index":MATCH}, "children"),
    Output({"type":"zoom","index":MATCH}, "data"),
    Input({"type":"resolved-path","index":MATCH}, "data"),   # <-- CHANGED
    Input({"type":"score-col","index":MATCH}, "value"),
    Input({"type":"fig","index":MATCH}, "relayoutData"),
    Input({"type":"fig","index":MATCH}, "clickData"),
    Input("render-mode","value"),
    Input("x-axis-mode","value"),
    State({"type":"zoom","index":MATCH}, "data")
)
def update_card(path, score_col, relayout, clickData, render_mode, axis_mode, zoom_state):
    if not path:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[dict(
                text="Select dataset → feature → file → pipeline",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color=COLORS['text_light'])
            )],
            height=400,
            plot_bgcolor='rgba(248, 249, 250, 0.5)',
            paper_bgcolor='white'
        )
        return empty_fig, html.Div("No data loaded", style={"opacity":0.6, "color": COLORS['text_light']}), html.Div(), (zoom_state or {"tick":0,"last_click_ts":0})

    df = load_csv(path)
    
    if not isinstance(zoom_state, dict) or "tick" not in zoom_state or "last_click_ts" not in zoom_state:
        zoom_state = {"tick": 0, "last_click_ts": 0}

    # Use enhanced plotting function with render_mode
    fig, _ = plot_timeseries(df, score_col, axis_mode=axis_mode, render_mode=render_mode)

    prop_id = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    is_click_event = prop_id.endswith(".clickData")
    is_relayout_event = prop_id.endswith(".relayoutData")

    new_state = zoom_state.copy()

    is_reset = bool(
        is_relayout_event and relayout and (
            relayout.get("autosize") is True or
            relayout.get("xaxis.autorange") is True or
            relayout.get("yaxis.autorange") is True
        )
    )

    new_state = zoom_state.copy() if isinstance(zoom_state, dict) else {"tick": 0, "last_click_ts": 0}

    if is_reset:
        new_state["tick"] = 0
    elif is_click_event and clickData and clickData.get("points"):
        apply_click_zoom(fig, clickData, df, axis_mode, time_window_s=CLICK_ZOOM_WINDOW_S, batch_window_n=CLICK_ZOOM_WINDOW_N)
        new_state["tick"] = new_state.get("tick", 0) + 1
    elif new_state["tick"] > 0 and is_relayout_event and relayout:
        if "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
            fig.update_xaxes(range=[relayout["xaxis.range[0]"], relayout["xaxis.range[1]"]], autorange=False)
        if "yaxis.range[0]" in relayout and "yaxis.range[1]" in relayout:
            fig.update_yaxes(range=[relayout["yaxis.range[0]"], relayout["yaxis.range[1]"]], autorange=False)

    fig.update_layout(uirevision=f"{axis_mode}-{new_state['tick']}")

    relayout_for_metrics = relayout if is_relayout_event else None
    dfv = filter_visible(df, relayout_for_metrics, axis_mode)
    summary, y_pred = summarize_run(dfv, score_col)
    y_true = dfv.get("y_true").to_numpy() if "y_true" in dfv.columns else None
    m = compute_metrics(y_true, y_pred.to_numpy() if hasattr(y_pred,"to_numpy") else np.asarray(y_pred))

    return fig, _metrics_block(m), format_summary(summary), new_state

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)