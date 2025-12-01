import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.signal import savgol_filter, butter, filtfilt

st.set_page_config(layout="wide", page_title="Weld Bead Signal Explorer")

# ---------- Bead Segmentation ----------

def segment_beads(df, column, threshold):
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

def aggregate_for_step(x, y, interval):
    """Aggregate x and y into buckets of given interval for step plotting."""
    if interval <= 1 or len(x) <= interval:
        return x, y
    agg_x = x[::interval]
    agg_y = [np.mean(y[i:i+interval]) for i in range(0, len(y), interval)]
    return agg_x[:len(agg_y)], agg_y

def parse_levels_from_path(path):
    """
    Expect: Level1/Level2/Level3/file.csv
    Returns (level1, level2, level3, filename).
    """
    norm = path.strip("/")
    parts = norm.split("/")
    filename = parts[-1]
    if len(parts) >= 4:
        level1 = parts[0]
        level2 = parts[1]
        level3 = parts[2]
    else:
        level1 = level2 = level3 = None
    return level1, level2, level3, filename

# ---------- Session State ----------

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

# ---------- Sidebar: ZIP Upload & Global Bead Segmentation ----------

st.sidebar.header("Step 1: Upload ZIP & Global Bead Segmentation")
uploaded_zip = st.sidebar.file_uploader("Upload ZIP of CSV folders", type="zip")

sample_columns = None
seg_col = None

if uploaded_zip:
    # Peek first CSV to get columns
    with zipfile.ZipFile(uploaded_zip, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if csv_names:
            first_csv = csv_names[0]
            with zf.open(first_csv) as f:
                sample_df = pd.read_csv(f)
            sample_columns = sample_df.columns.tolist()
            st.session_state.all_columns = sample_columns

if uploaded_zip and sample_columns:
    seg_col = st.sidebar.selectbox(
        "Column for Bead Segmentation (applied to all CSVs)",
        sample_columns,
        index=min(2, len(sample_columns) - 1),  # often 3rd col
        key="seg_col_select",
    )
    seg_thresh = st.sidebar.number_input(
        "Segmentation Threshold",
        value=1.0,
        step=0.1,
        format="%.3f",
        key="seg_thresh",
    )

    if st.sidebar.button("Run Global Bead Segmentation"):
        records = []
        file_info = {}

        with zipfile.ZipFile(uploaded_zip, "r") as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".csv"):
                    continue

                level1, level2, level3, filename = parse_levels_from_path(name)
                # Only CSVs under Level3 folder
                if level1 is None or level2 is None or level3 is None:
                    continue

                with zf.open(name) as f:
                    df = pd.read_csv(f)

                if seg_col not in df.columns:
                    continue

                bead_ranges = segment_beads(df, seg_col, seg_thresh)
                bead_count = len(bead_ranges)

                for bead_num, (start, end) in enumerate(bead_ranges, start=1):
                    bead_df = df.iloc[start:end+1].reset_index(drop=True)
                    if bead_df.empty:
                        continue
                    records.append({
                        "RelPath": name,
                        "Level1": level1,
                        "Level2": level2,
                        "Level3": level3,
                        "FileName": filename,
                        "BeadNumber": bead_num,
                        "df": bead_df,
                    })

                # file-level info
                if name not in file_info:
                    file_info[name] = {
                        "RelPath": name,
                        "Level1": level1,
                        "Level2": level2,
                        "Level3": level3,
                        "FileName": filename,
                        "BeadCount": bead_count,
                    }

        if not records:
            st.warning("No beads found with the selected segmentation column/threshold.")
        else:
            st.session_state.segmented_records = records
            st.session_state.files_df = pd.DataFrame(list(file_info.values()))

            # assign colors per Level3
            level3_values = sorted({r["Level3"] for r in records})
            palette = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                "#bcbd22", "#17becf",
            ]
            level3_colors = {}
            for i, lvl3 in enumerate(level3_values):
                level3_colors[lvl3] = palette[i % len(palette)]
            st.session_state.level3_colors = level3_colors

            st.success("✅ Global bead segmentation complete for all CSVs.")

# ---------- Main Layout ----------

st.title("Weld Bead Signal Explorer (by Category & Bead)")

segmented_records = st.session_state.segmented_records
files_df = st.session_state.files_df

if segmented_records is not None and files_df is not None and len(segmented_records) > 0:
    # Step 2: Build Hierarchy & Load Signals
    st.subheader("Step 2: Build Hierarchy & Load Signals")
    st.markdown(
        "Below is the extracted folder hierarchy (Level1/Level2/Level3) "
        "and number of beads per CSV file."
    )
    st.dataframe(files_df, use_container_width=True)

    # ---- Signal column & sampling rate selection ----
    st.subheader("Step 3: Global Visualization Settings")

    all_cols = st.session_state.all_columns or list(segmented_records[0]["df"].columns)
    if st.session_state.signal_col and st.session_state.signal_col in all_cols:
        default_idx = all_cols.index(st.session_state.signal_col)
    else:
        default_idx = 0

    signal_col = st.selectbox(
        "Signal column to visualize (applies to all beads, all plots)",
        all_cols,
        index=default_idx,
        key="signal_col_global",
    )
    st.session_state.signal_col = signal_col

    sampling_rate = st.number_input(
        "Sampling Rate (Hz, used for FFT & Intensity)",
        value=float(st.session_state.get("sampling_rate", 4000.0)),
        min_value=1.0,
        step=1.0,
        key="sampling_rate_global",
    )
    st.session_state.sampling_rate = sampling_rate

    # ---- Build nested structure: Level2 -> BeadNumber -> list of traces ----
    data_by_level2 = {}
    level3_colors = st.session_state.level3_colors

    for rec in segmented_records:
        lvl2 = rec["Level2"]
        lvl3 = rec["Level3"]
        bead_num = rec["BeadNumber"]
        fname = rec["FileName"]
        df_bead = rec["df"]

        if signal_col not in df_bead.columns:
            continue

        sig = pd.to_numeric(df_bead[signal_col], errors="coerce").dropna().reset_index(drop=True)
        if sig.empty:
            continue

        trace = {
            "Level3": lvl3,
            "FileName": fname,
            "signal": sig,
        }
        data_by_level2.setdefault(lvl2, {}).setdefault(bead_num, []).append(trace)

    if not data_by_level2:
        st.info("No valid bead signals found for the selected signal column.")
    else:
        # Tabs: same transformations as before, but per bead, per Level2
        tabs = st.tabs([
            "Raw Signal",
            "Smoothed",
            "Low-pass Filter",
            "Curve Fit",
            "FFT Band Intensity",
            "Signal Intensity (dB)",
        ])

        # common helper: create subplot grid per Level2
        def create_subplots_for_level2(bead_dict, title_prefix):
            bead_numbers = sorted(bead_dict.keys())
            if not bead_numbers:
                return None

            max_cols = 8
            n_beads = len(bead_numbers)
            ncols = min(n_beads, max_cols)
            nrows = math.ceil(n_beads / ncols)

            fig = make_subplots(
                rows=nrows,
                cols=ncols,
                subplot_titles=[f"Bead {b}" for b in bead_numbers],
            )
            return fig, bead_numbers, nrows, ncols

        # ---------- Tab 0: Raw Signal ----------
        with tabs[0]:
            st.subheader("Raw Signal (Step Line, per Level2 category & Bead)")

            for lvl2 in sorted(data_by_level2.keys()):
                bead_dict = data_by_level2[lvl2]
                with st.expander(f"Level2: {lvl2}", expanded=True):
                    fig_data = create_subplots_for_level2(bead_dict, "Raw Signal")
                    if fig_data is None:
                        st.write("No beads.")
                        continue
                    fig, bead_numbers, nrows, ncols = fig_data

                    seen_lvl3 = set()

                    for idx, bead_num in enumerate(bead_numbers):
                        row = idx // ncols + 1
                        col = idx % ncols + 1

                        for trace in bead_dict[bead_num]:
                            y = trace["signal"].to_numpy()
                            x = np.arange(len(y))
                            lvl3 = trace["Level3"]
                            color = level3_colors.get(lvl3, "#333333")

                            showlegend = lvl3 not in seen_lvl3
                            if showlegend:
                                seen_lvl3.add(lvl3)

                            fig.add_trace(
                                go.Scatter(
                                    x=x,
                                    y=y,
                                    mode="lines",
                                    name=lvl3,
                                    line=dict(color=color, shape="hv"),
                                    showlegend=showlegend,
                                ),
                                row=row,
                                col=col,
                            )

                    fig.update_layout(
                        title=f"Raw Signal (Step Line) - Level2: {lvl2}",
                        height=300 * math.ceil(len(bead_numbers) / ncols),
                    )
                    fig.update_xaxes(title_text="Index")
                    fig.update_yaxes(title_text=signal_col)
                    st.plotly_chart(fig, use_container_width=True)

        # ---------- Tab 1: Smoothed ----------
        with tabs[1]:
            st.subheader("Savitzky-Golay Smoothed (Step Line, per Level2 & Bead)")

            window = st.slider("Savitzky-Golay Window Length", 5, 101, 51, step=2, key="sg_window")
            if window % 2 == 0:
                window += 1
            poly = st.slider("Polynomial Order", 2, 5, 2, key="sg_poly")
            step_interval = st.slider(
                "Step Interval (points) - Smoothed (downsampling for step view)",
                1, 500, 50,
                key="sg_step_interval",
            )

            for lvl2 in sorted(data_by_level2.keys()):
                bead_dict = data_by_level2[lvl2]
                with st.expander(f"Level2: {lvl2}", expanded=False):
                    fig_data = create_subplots_for_level2(bead_dict, "Smoothed")
                    if fig_data is None:
                        st.write("No beads.")
                        continue
                    fig, bead_numbers, nrows, ncols = fig_data

                    seen_lvl3 = set()

                    for idx, bead_num in enumerate(bead_numbers):
                        row = idx // ncols + 1
                        col = idx % ncols + 1

                        for trace in bead_dict[bead_num]:
                            data = trace["signal"].to_numpy()
                            if len(data) < window:
                                continue
                            smoothed = savgol_filter(data, window, poly)
                            x_vals = np.arange(len(smoothed))
                            x_vals, smoothed = aggregate_for_step(x_vals, smoothed, step_interval)

                            lvl3 = trace["Level3"]
                            color = level3_colors.get(lvl3, "#333333")
                            showlegend = lvl3 not in seen_lvl3
                            if showlegend:
                                seen_lvl3.add(lvl3)

                            fig.add_trace(
                                go.Scatter(
                                    x=x_vals,
                                    y=smoothed,
                                    mode="lines",
                                    name=lvl3,
                                    line=dict(color=color, shape="hv"),
                                    showlegend=showlegend,
                                ),
                                row=row,
                                col=col,
                            )

                    fig.update_layout(
                        title=f"Smoothed Signal (Step Line) - Level2: {lvl2}",
                        height=300 * math.ceil(len(bead_numbers) / ncols),
                    )
                    fig.update_xaxes(title_text="Index")
                    fig.update_yaxes(title_text=signal_col)
                    st.plotly_chart(fig, use_container_width=True)

        # ---------- Tab 2: Low-pass Filter ----------
        with tabs[2]:
            st.subheader("Low-pass Filtered (Step Line, per Level2 & Bead)")

            cutoff = st.slider(
                "Low-pass Cutoff Frequency (normalized 0–0.5)",
                0.01, 0.5, 0.1,
                key="lp_cutoff",
            )
            order = st.slider("Filter Order", 1, 5, 2, key="lp_order")
            step_interval = st.slider(
                "Step Interval (points) - Low-pass",
                1, 500, 50,
                key="lp_step_interval",
            )

            for lvl2 in sorted(data_by_level2.keys()):
                bead_dict = data_by_level2[lvl2]
                with st.expander(f"Level2: {lvl2}", expanded=False):
                    fig_data = create_subplots_for_level2(bead_dict, "Low-pass")
                    if fig_data is None:
                        st.write("No beads.")
                        continue
                    fig, bead_numbers, nrows, ncols = fig_data

                    seen_lvl3 = set()

                    for idx, bead_num in enumerate(bead_numbers):
                        row = idx // ncols + 1
                        col = idx % ncols + 1

                        for trace in bead_dict[bead_num]:
                            data = trace["signal"].to_numpy()
                            if len(data) < order * 3:
                                continue
                            b, a = butter(order, cutoff, btype="low", analog=False)
                            try:
                                filtered = filtfilt(b, a, data)
                            except Exception:
                                continue

                            x_vals = np.arange(len(filtered))
                            x_vals, filtered = aggregate_for_step(x_vals, filtered, step_interval)

                            lvl3 = trace["Level3"]
                            color = level3_colors.get(lvl3, "#333333")
                            showlegend = lvl3 not in seen_lvl3
                            if showlegend:
                                seen_lvl3.add(lvl3)

                            fig.add_trace(
                                go.Scatter(
                                    x=x_vals,
                                    y=filtered,
                                    mode="lines",
                                    name=lvl3,
                                    line=dict(color=color, shape="hv"),
                                    showlegend=showlegend,
                                ),
                                row=row,
                                col=col,
                            )

                    fig.update_layout(
                        title=f"Low-pass Filtered (Step Line) - Level2: {lvl2}",
                        height=300 * math.ceil(len(bead_numbers) / ncols),
                    )
                    fig.update_xaxes(title_text="Index")
                    fig.update_yaxes(title_text=signal_col)
                    st.plotly_chart(fig, use_container_width=True)

        # ---------- Tab 3: Curve Fit ----------
        with tabs[3]:
            st.subheader("Polynomial Curve Fit (Step Line, per Level2 & Bead)")

            deg = st.slider("Curve Fit Polynomial Degree", 1, 50, 10, key="cf_deg")
            step_interval = st.slider(
                "Step Interval (points) - Curve Fit",
                1, 500, 50,
                key="cf_step_interval",
            )

            for lvl2 in sorted(data_by_level2.keys()):
                bead_dict = data_by_level2[lvl2]
                with st.expander(f"Level2: {lvl2}", expanded=False):
                    fig_data = create_subplots_for_level2(bead_dict, "Curve Fit")
                    if fig_data is None:
                        st.write("No beads.")
                        continue
                    fig, bead_numbers, nrows, ncols = fig_data

                    seen_lvl3 = set()

                    for idx, bead_num in enumerate(bead_numbers):
                        row = idx // ncols + 1
                        col = idx % ncols + 1

                        for trace in bead_dict[bead_num]:
                            data = trace["signal"].to_numpy()
                            x = np.arange(len(data))
                            if len(x) <= deg:
                                continue
                            try:
                                coeffs = np.polyfit(x, data, deg)
                                fitted = np.polyval(coeffs, x)
                            except Exception:
                                continue

                            x_vals = x
                            x_vals, fitted = aggregate_for_step(x_vals, fitted, step_interval)

                            lvl3 = trace["Level3"]
                            color = level3_colors.get(lvl3, "#333333")
                            showlegend = lvl3 not in seen_lvl3
                            if showlegend:
                                seen_lvl3.add(lvl3)

                            fig.add_trace(
                                go.Scatter(
                                    x=x_vals,
                                    y=fitted,
                                    mode="lines",
                                    name=lvl3,
                                    line=dict(color=color, shape="hv"),
                                    showlegend=showlegend,
                                ),
                                row=row,
                                col=col,
                            )

                    fig.update_layout(
                        title=f"Curve Fit (Step Line) - Level2: {lvl2}",
                        height=300 * math.ceil(len(bead_numbers) / ncols),
                    )
                    fig.update_xaxes(title_text="Index")
                    fig.update_yaxes(title_text=signal_col)
                    st.plotly_chart(fig, use_container_width=True)

        # ---------- Tab 4: FFT Band Intensity ----------
        with tabs[4]:
            st.subheader("FFT Spectrum (Zoomed Band, in dB, per Level2 & Bead)")

            sr_fft = st.session_state.sampling_rate
            band_low, band_high = st.slider(
                "Frequency Band (Hz) - Spectrum",
                min_value=0,
                max_value=2000,
                value=(100, 500),
                step=10,
                key="fft_band",
            )

            for lvl2 in sorted(data_by_level2.keys()):
                bead_dict = data_by_level2[lvl2]
                with st.expander(f"Level2: {lvl2}", expanded=False):
                    fig_data = create_subplots_for_level2(bead_dict, "FFT")
                    if fig_data is None:
                        st.write("No beads.")
                        continue
                    fig, bead_numbers, nrows, ncols = fig_data

                    seen_lvl3 = set()

                    for idx, bead_num in enumerate(bead_numbers):
                        row = idx // ncols + 1
                        col = idx % ncols + 1

                        for trace in bead_dict[bead_num]:
                            data = trace["signal"].to_numpy()
                            if len(data) < 4:
                                continue

                            fft_vals = np.fft.rfft(data)
                            freqs = np.fft.rfftfreq(len(data), d=1.0 / sr_fft)
                            magnitude = np.abs(fft_vals)
                            magnitude_db = 20 * np.log10(magnitude + 1e-12)

                            mask = (freqs >= band_low) & (freqs <= band_high)
                            if not np.any(mask):
                                continue
                            freqs_zoom = freqs[mask]
                            mag_zoom = magnitude_db[mask]

                            lvl3 = trace["Level3"]
                            color = level3_colors.get(lvl3, "#333333")
                            showlegend = lvl3 not in seen_lvl3
                            if showlegend:
                                seen_lvl3.add(lvl3)

                            fig.add_trace(
                                go.Scatter(
                                    x=freqs_zoom,
                                    y=mag_zoom,
                                    mode="lines",
                                    name=lvl3,
                                    line=dict(color=color, shape="hv"),
                                    showlegend=showlegend,
                                    fill="tozeroy",
                                    fillcolor="rgba(0,0,0,0.03)",
                                ),
                                row=row,
                                col=col,
                            )

                    fig.update_layout(
                        title=f"FFT Spectrum (Zoomed {band_low}-{band_high} Hz, dB) - Level2: {lvl2}",
                        height=300 * math.ceil(len(bead_numbers) / ncols),
                    )
                    fig.update_xaxes(title_text="Frequency (Hz)")
                    fig.update_yaxes(title_text="Intensity (dB)", autorange="reversed")
                    st.plotly_chart(fig, use_container_width=True)

        # ---------- Tab 5: Signal Intensity (dB) ----------
        with tabs[5]:
            st.subheader("Signal Intensity (dB) Over Time (per Level2 & Bead)")

            sr_int = st.session_state.sampling_rate

            use_exact_freq = st.checkbox(
                "Use Exact Frequency (Hz) Instead of Band",
                key="int_exact",
            )

            if use_exact_freq:
                exact_freq = st.number_input(
                    "Exact Frequency (Hz)",
                    value=150.0,
                    min_value=0.0,
                    max_value=2000.0,
                    step=10.0,
                    key="int_exact_freq",
                )
            else:
                band_low_int, band_high_int = st.slider(
                    "Frequency Band (Hz) - Intensity",
                    0, 2000, (130, 170),
                    step=10,
                    key="int_band",
                )

            for lvl2 in sorted(data_by_level2.keys()):
                bead_dict = data_by_level2[lvl2]
                with st.expander(f"Level2: {lvl2}", expanded=False):
                    fig_data = create_subplots_for_level2(bead_dict, "Intensity")
                    if fig_data is None:
                        st.write("No beads.")
                        continue
                    fig, bead_numbers, nrows, ncols = fig_data

                    seen_lvl3 = set()

                    for idx, bead_num in enumerate(bead_numbers):
                        row = idx // ncols + 1
                        col = idx % ncols + 1

                        for trace in bead_dict[bead_num]:
                            data = trace["signal"].to_numpy()
                            n = len(data)
                            if n < 16:
                                continue

                            nperseg = min(1024, n // 4)
                            if nperseg < 4:
                                continue
                            noverlap = int(0.9 * nperseg)
                            step_points = max(1, nperseg - noverlap)
                            nfft = min(2048, 4 ** int(np.ceil(np.log2(nperseg * 2))))

                            times = []
                            intensities = []

                            for start in range(0, n - nperseg, step_points):
                                segment = data[start:start + nperseg]
                                fft_vals = np.fft.rfft(segment, n=nfft)
                                freqs = np.fft.rfftfreq(nfft, d=1.0 / sr_int)

                                if use_exact_freq:
                                    idx_freq = (np.abs(freqs - exact_freq)).argmin()
                                    band_mag = np.abs(fft_vals[idx_freq])
                                else:
                                    mask = (freqs >= band_low_int) & (freqs <= band_high_int)
                                    if not np.any(mask):
                                        continue
                                    band_mag = np.sum(np.abs(fft_vals[mask]))

                                intensity_db = 20 * np.log10(band_mag + 1e-12)
                                times.append(start / sr_int)
                                intensities.append(intensity_db)

                            if not times:
                                continue

                            lvl3 = trace["Level3"]
                            color = level3_colors.get(lvl3, "#333333")
                            showlegend = lvl3 not in seen_lvl3
                            if showlegend:
                                seen_lvl3.add(lvl3)

                            fig.add_trace(
                                go.Scatter(
                                    x=times,
                                    y=intensities,
                                    mode="lines",
                                    name=lvl3,
                                    line=dict(color=color, shape="hv"),
                                    showlegend=showlegend,
                                ),
                                row=row,
                                col=col,
                            )

                    fig.update_layout(
                        title=f"Signal Intensity (dB) Over Time - Level2: {lvl2}",
                        height=300 * math.ceil(len(bead_numbers) / ncols),
                    )
                    fig.update_xaxes(title_text="Time (s)")
                    fig.update_yaxes(title_text="Intensity (dB)")
                    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload a ZIP and click **Run Global Bead Segmentation** in the sidebar to begin.")
