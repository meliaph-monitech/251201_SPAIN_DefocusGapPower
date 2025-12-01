import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os

import plotly.graph_objects as go
from scipy.signal import savgol_filter, butter, filtfilt


st.set_page_config(layout="wide", page_title="Weld Bead Signal Explorer")

# ============================================================
#                   BEAD SEGMENTATION
# ============================================================

def segment_beads(df, column, threshold):
    """Return list of (start_idx, end_idx) for beads in column > threshold."""
    start_indices, end_indices = [], []
    signal = pd.to_numeric(df[column], errors="coerce").to_numpy()
    i = 0
    n = len(signal)
    while i < n:
        val = signal[i]
        if np.isfinite(val) and val > threshold:
            start = i
            i += 1
            while i < n and np.isfinite(signal[i]) and signal[i] > threshold:
                i += 1
            end = i - 1
            start_indices.append(start)
            end_indices.append(end)
        else:
            i += 1
    return list(zip(start_indices, end_indices))


def parse_levels_from_path(path):
    """Expect: Level1/Level2/Level3/file.csv"""
    norm = path.strip("/")
    parts = norm.split("/")
    filename = parts[-1]
    if len(parts) >= 4:
        return parts[0], parts[1], parts[2], filename
    return None, None, None, filename


def get_file_prefix(filename):
    """Return substring of name before 2nd underscore."""
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return base


# ============================================================
#                   SESSION STATE
# ============================================================

if "segmented_records" not in st.session_state:
    st.session_state.segmented_records = None

if "files_df" not in st.session_state:
    st.session_state.files_df = None

if "signal_col" not in st.session_state:
    st.session_state.signal_col = None

if "sampling_rate" not in st.session_state:
    st.session_state.sampling_rate = 4000.0

if "all_columns" not in st.session_state:
    st.session_state.all_columns = None

if "level3_colors" not in st.session_state:
    st.session_state.level3_colors = {}


# ============================================================
#                   UPLOAD & SEGMENT
# ============================================================

st.sidebar.header("Step 1: Upload ZIP & Global Bead Segmentation")
uploaded_zip = st.sidebar.file_uploader("Upload ZIP of CSV folders", type="zip")

sample_columns = None

if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, "r") as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if csvs:
            with zf.open(csvs[0]) as f:
                df = pd.read_csv(f)
            sample_columns = df.columns.tolist()
            st.session_state.all_columns = sample_columns

if uploaded_zip and sample_columns:
    seg_col = st.sidebar.selectbox(
        "Column for Bead Segmentation",
        sample_columns,
        index=min(2, len(sample_columns)-1)
    )
    seg_thresh = st.sidebar.number_input("Threshold", value=1.0)

    if st.sidebar.button("Run Global Bead Segmentation"):
        records = []
        files_info = {}

        with zipfile.ZipFile(uploaded_zip, "r") as zf:
            for path in zf.namelist():
                if not path.lower().endswith(".csv"):
                    continue

                L1, L2, L3, fname = parse_levels_from_path(path)
                if L1 is None or L2 is None or L3 is None:
                    continue

                with zf.open(path) as f:
                    df = pd.read_csv(f)

                if seg_col not in df.columns:
                    continue

                bead_ranges = segment_beads(df, seg_col, seg_thresh)
                bead_count = len(bead_ranges)

                # Save each bead
                for bead_num, (s, e) in enumerate(bead_ranges, start=1):
                    bead_df = df.iloc[s:e+1].reset_index(drop=True)
                    if not bead_df.empty:
                        records.append({
                            "RelPath": path,
                            "Level1": L1,
                            "Level2": L2,
                            "Level3": L3,
                            "FileName": fname,
                            "BeadNumber": bead_num,
                            "df": bead_df
                        })

                # File-level info
                if path not in files_info:
                    files_info[path] = {
                        "RelPath": path,
                        "Level1": L1,
                        "Level2": L2,
                        "Level3": L3,
                        "FileName": fname,
                        "BeadCount": bead_count
                    }

        st.session_state.segmented_records = records
        st.session_state.files_df = pd.DataFrame(files_info.values())

        # Assign colors for Level3
        unique_L3 = sorted({r["Level3"] for r in records})
        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            "#bcbd22", "#17becf"
        ]
        st.session_state.level3_colors = {
            L3: palette[i % len(palette)] for i, L3 in enumerate(unique_L3)
        }

        st.success("Segmentation complete.")


# ============================================================
#                   MAIN VIEW
# ============================================================

st.title("Weld Bead Signal Explorer (Concatenated Beads)")

records = st.session_state.segmented_records
files_df = st.session_state.files_df

if records:

    # ---------------------
    # Show hierarchy
    # ---------------------
    st.subheader("Step 2: Folder Hierarchy")
    st.dataframe(files_df, use_container_width=True)

    # ---------------------
    # Global settings
    # ---------------------
    st.subheader("Step 3: Visualization Settings")

    cols = st.session_state.all_columns
    default_col = st.session_state.signal_col or cols[0]

    signal_col = st.selectbox(
        "Signal Column",
        cols,
        index=cols.index(default_col) if default_col in cols else 0
    )
    st.session_state.signal_col = signal_col

    sr = st.number_input("Sampling Rate (Hz)", value=st.session_state.sampling_rate)
    st.session_state.sampling_rate = sr

    colors = st.session_state.level3_colors

    # ---------------------
    # Organize data
    # ---------------------
    data_by_L2 = {}

    for r in records:
        L2 = r["Level2"]
        L3 = r["Level3"]
        fpath = r["RelPath"]
        fname = r["FileName"]
        df_bead = r["df"]

        if signal_col not in df_bead.columns:
            continue

        y = pd.to_numeric(df_bead[signal_col], errors="coerce").dropna().reset_index(drop=True)
        if y.empty:
            continue

        entry = data_by_L2.setdefault(L2, {}).setdefault(
            fpath,
            {
                "Level3": L3,
                "FileName": fname,
                "FilePrefix": get_file_prefix(fname),
                "beads": []
            }
        )
        entry["beads"].append(y)

    if not data_by_L2:
        st.info("No valid data.")
        st.stop()

    # Utility: concatenate beads
    def concat_beads(beads, transform=None):
        xs, ys, ids = [], [], []
        boundaries = []
        offset = 0

        for b_i, bead in enumerate(beads, start=1):
            y = np.asarray(bead, float)
            if transform:
                y = transform(y)

            n = len(y)
            x = np.arange(n) + offset

            xs.append(x)
            ys.append(y)
            ids.append(np.full(n, b_i, int))

            if b_i > 1:
                boundaries.append(offset)

            offset += n

        if not xs:
            return np.array([]), np.array([]), [], np.array([])

        return (
            np.concatenate(xs),
            np.concatenate(ys),
            boundaries,
            np.concatenate(ids)
        )

    # ============================================================
    #                     PLOTTING TABS
    # ============================================================

    tabs = st.tabs([
        "Raw Signal",
        "Smoothed",
        "Low-pass",
        "Curve Fit",
        "FFT Band",
        "Intensity (dB)"
    ])

    # ================== RAW =====================
    with tabs[0]:
        st.subheader("Raw Signal (Step Line)")

        for L2 in sorted(data_by_L2.keys()):
            with st.expander(f"Level2: {L2}", expanded=True):
                fig = go.Figure()
                seen = set()
                all_bounds = set()

                for path, entry in data_by_L2[L2].items():
                    L3 = entry["Level3"]
                    prefix = entry["FilePrefix"]
                    beads = entry["beads"]

                    x, y, bounds, bead_ids = concat_beads(beads)
                    if x.size == 0:
                        continue

                    all_bounds.update(bounds)
                    color = colors[L3]
                    showleg = L3 not in seen
                    seen.add(L3)

                    custom = np.stack([bead_ids], axis=-1)

                    fig.add_trace(go.Scatter(
                        x=x, y=y,
                        mode="lines",
                        line_shape="hv",                 # <-- TRUE STEP LINE
                        line=dict(color=color, width=1), # thin lines
                        name=L3,
                        legendgroup=L3,
                        showlegend=showleg,
                        meta=prefix,
                        customdata=custom,
                        hovertemplate=(
                            "L3: %{name}<br>"
                            "File: %{meta}<br>"
                            "Bead: %{customdata[0]}<br>"
                            "Index: %{x}<br>"
                            "Value: %{y}<extra></extra>"
                        )
                    ))

                # bead dividers
                for b in sorted(all_bounds):
                    fig.add_vline(x=b, line_width=1, line_dash="dot", line_color="black")

                fig.update_layout(
                    legend=dict(groupclick="togglegroup"),
                    xaxis_title="Index (concatenated)",
                    yaxis_title=signal_col,
                )
                st.plotly_chart(fig, use_container_width=True)

    # ================== SMOOTHED =====================
    with tabs[1]:
        st.subheader("Smoothed (Savitzky-Golay Step Line)")

        w = st.slider("Window", 5, 101, 51, step=2)
        if w % 2 == 0:
            w += 1
        poly = st.slider("Poly Order", 2, 5, 2)

        def sg(y):
            if len(y) < w:
                return y
            return savgol_filter(y, w, poly)

        for L2 in sorted(data_by_L2.keys()):
            with st.expander(f"Level2: {L2}", expanded=False):
                fig = go.Figure()
                seen = set()
                all_bounds = set()

                for path, entry in data_by_L2[L2].items():
                    L3 = entry["Level3"]
                    prefix = entry["FilePrefix"]
                    x, y, bounds, bead_ids = concat_beads(entry["beads"], sg)
                    if x.size == 0:
                        continue

                    all_bounds.update(bounds)
                    color = colors[L3]
                    showleg = L3 not in seen
                    seen.add(L3)
                    custom = np.stack([bead_ids], axis=-1)

                    fig.add_trace(go.Scatter(
                        x=x, y=y,
                        mode="lines",
                        line_shape="hv",
                        line=dict(color=color, width=1),
                        name=L3,
                        legendgroup=L3,
                        showlegend=showleg,
                        meta=prefix,
                        customdata=custom,
                        hovertemplate=(
                            "L3: %{name}<br>File: %{meta}<br>"
                            "Bead: %{customdata[0]}<br>"
                            "Index: %{x}<br>Value: %{y}<extra></extra>"
                        )
                    ))

                for b in sorted(all_bounds):
                    fig.add_vline(x=b, line_width=1, line_dash="dot")

                fig.update_layout(legend=dict(groupclick="togglegroup"))
                st.plotly_chart(fig, use_container_width=True)

    # ================== LOW-PASS =====================
    with tabs[2]:
        st.subheader("Low-pass Filter (Step Line)")

        cutoff = st.slider("Cutoff (0–0.5)", 0.01, 0.5, 0.1)
        order = st.slider("Order", 1, 5, 2)

        def lp(y):
            if len(y) < order * 3:
                return y
            b, a = butter(order, cutoff)
            try:
                return filtfilt(b, a, y)
            except:
                return y

        for L2 in sorted(data_by_L2.keys()):
            with st.expander(f"Level2: {L2}", expanded=False):
                fig = go.Figure()
                seen = set()
                all_bounds = set()

                for path, entry in data_by_L2[L2].items():
                    L3 = entry["Level3"]
                    prefix = entry["FilePrefix"]
                    x, y, bounds, bead_ids = concat_beads(entry["beads"], lp)
                    if x.size == 0:
                        continue

                    all_bounds.update(bounds)
                    color = colors[L3]
                    showleg = L3 not in seen
                    seen.add(L3)
                    custom = np.stack([bead_ids], axis=-1)

                    fig.add_trace(go.Scatter(
                        x=x, y=y,
                        mode="lines",
                        line_shape="hv",
                        line=dict(color=color, width=1),
                        name=L3,
                        legendgroup=L3,
                        showlegend=showleg,
                        meta=prefix,
                        customdata=custom,
                        hovertemplate=(
                            "L3: %{name}<br>File: %{meta}<br>"
                            "Bead: %{customdata[0]}<br>"
                            "Index: %{x}<br>Value: %{y}<extra></extra>"
                        )
                    ))

                for b in sorted(all_bounds):
                    fig.add_vline(x=b, line_width=1, line_dash="dot")

                fig.update_layout(legend=dict(groupclick="togglegroup"))
                st.plotly_chart(fig, use_container_width=True)

    # ================== CURVE FIT =====================
    with tabs[3]:
        st.subheader("Curve Fit (Step Line)")

        deg = st.slider("Polynomial Degree", 1, 50, 10)

        def polyfit_transform(y):
            x0 = np.arange(len(y))
            if len(y) <= deg:
                return y
            try:
                c = np.polyfit(x0, y, deg)
                return np.polyval(c, x0)
            except:
                return y

        for L2 in sorted(data_by_L2.keys()):
            with st.expander(f"Level2: {L2}", expanded=False):
                fig = go.Figure()
                seen = set()
                all_bounds = set()

                for path, entry in data_by_L2[L2].items():
                    L3 = entry["Level3"]
                    prefix = entry["FilePrefix"]
                    x, y, bounds, bead_ids = concat_beads(entry["beads"], polyfit_transform)
                    if x.size == 0:
                        continue

                    all_bounds.update(bounds)
                    color = colors[L3]
                    showleg = L3 not in seen
                    seen.add(L3)
                    custom = np.stack([bead_ids], axis=-1)

                    fig.add_trace(go.Scatter(
                        x=x, y=y,
                        mode="lines",
                        line_shape="hv",
                        line=dict(color=color, width=1),
                        name=L3,
                        legendgroup=L3,
                        showlegend=showleg,
                        meta=prefix,
                        customdata=custom,
                        hovertemplate=(
                            "L3: %{name}<br>File: %{meta}<br>"
                            "Bead: %{customdata[0]}<br>"
                            "Index: %{x}<br>Value: %{y}<extra></extra>"
                        )
                    ))

                for b in sorted(all_bounds):
                    fig.add_vline(x=b, line_width=1, line_dash="dot")

                fig.update_layout(legend=dict(groupclick="togglegroup"))
                st.plotly_chart(fig, use_container_width=True)

    # ================== FFT (No Step Line) =====================
    with tabs[4]:
        st.subheader("FFT Spectrum (Smooth)")

        low, high = st.slider("Band (Hz)", 0, 2000, (100, 500))

        for L2 in sorted(data_by_L2.keys()):
            with st.expander(f"Level2: {L2}", expanded=False):
                fig = go.Figure()
                seen = set()

                for path, entry in data_by_L2[L2].items():
                    L3 = entry["Level3"]
                    prefix = entry["FilePrefix"]
                    beads = entry["beads"]

                    x, y, _, _ = concat_beads(beads)
                    if y.size < 4:
                        continue

                    freqs = np.fft.rfftfreq(len(y), 1/sr)
                    mag = np.abs(np.fft.rfft(y))
                    db = 20 * np.log10(mag + 1e-12)

                    mask = (freqs >= low) & (freqs <= high)
                    if not mask.any():
                        continue

                    color = colors[L3]
                    showleg = L3 not in seen
                    seen.add(L3)

                    fig.add_trace(go.Scatter(
                        x=freqs[mask],
                        y=db[mask],
                        mode="lines",
                        line=dict(color=color, width=1),
                        name=L3,
                        legendgroup=L3,
                        showlegend=showleg,
                        meta=prefix,
                        hovertemplate=(
                            "L3: %{name}<br>File: %{meta}<br>"
                            "Freq: %{x} Hz<br>Int: %{y} dB<extra></extra>"
                        )
                    ))

                fig.update_layout(
                    yaxis=dict(autorange="reversed"),
                    legend=dict(groupclick="togglegroup")
                )

                st.plotly_chart(fig, use_container_width=True)

    # ================== INTENSITY (Step Line) =====================
    with tabs[5]:
        st.subheader("Signal Intensity (dB)")

        exact = st.checkbox("Exact Frequency")
        if exact:
            f0 = st.number_input("Frequency (Hz)", value=150.0)
        else:
            b_low, b_high = st.slider("Band", 0, 2000, (130, 170))

        for L2 in sorted(data_by_L2.keys()):
            with st.expander(f"Level2: {L2}", expanded=False):
                fig = go.Figure()
                seen = set()

                for path, entry in data_by_L2[L2].items():
                    L3 = entry["Level3"]
                    prefix = entry["FilePrefix"]
                    x, y, _, _ = concat_beads(entry["beads"])

                    if len(y) < 32:
                        continue

                    # sliding FFT
                    nper = min(1024, len(y)//4)
                    nover = int(0.9*nper)
                    step = max(1, nper-nover)
                    nfft = 2048

                    ts, vals = [], []

                    for s in range(0, len(y)-nper, step):
                        seg = y[s:s+nper]
                        fft = np.fft.rfft(seg, nfft)
                        freqs = np.fft.rfftfreq(nfft, 1/sr)

                        if exact:
                            idx = np.abs(freqs - f0).argmin()
                            mag = np.abs(fft[idx])
                        else:
                            m = (freqs >= b_low) & (freqs <= b_high)
                            mag = np.sum(np.abs(fft[m]))

                        db = 20*np.log10(mag+1e-12)
                        ts.append(s/sr)
                        vals.append(db)

                    if not ts:
                        continue

                    color = colors[L3]
                    showleg = L3 not in seen
                    seen.add(L3)

                    fig.add_trace(go.Scatter(
                        x=ts,
                        y=vals,
                        mode="lines",
                        line_shape="hv",
                        line=dict(color=color, width=1),
                        name=L3,
                        legendgroup=L3,
                        showlegend=showleg,
                        meta=prefix,
                        hovertemplate=(
                            "L3: %{name}<br>File: %{meta}<br>"
                            "t: %{x}s<br>Int: %{y} dB<extra></extra>"
                        )
                    ))

                fig.update_layout(legend=dict(groupclick="togglegroup"))
                st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload a ZIP and run segmentation.")
