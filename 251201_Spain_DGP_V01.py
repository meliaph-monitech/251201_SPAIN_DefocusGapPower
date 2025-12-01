import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
import os
import plotly.graph_objects as go
from scipy.signal import savgol_filter, butter, filtfilt

st.set_page_config(layout="wide", page_title="Weld Signal Explorer")

# ---------- Utility ----------

def aggregate_for_step(x, y, interval):
    """Aggregate x and y into buckets of given interval for step plotting."""
    agg_x = x[::interval]
    agg_y = [np.mean(y[i:i+interval]) for i in range(0, len(y), interval)]
    return agg_x[:len(agg_y)], agg_y

def parse_levels_from_path(path):
    """
    Expect something like: Level1/Level2/Level3/file.csv
    Returns (level1, level2, level3, filename) or (None, None, None, filename) if not matching.
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

def build_label(level3, filename):
    """
    Color/legend label = combination of level3 folder name
    and CSV filename, but only the string before the second "_".
    """
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    if len(parts) >= 2:
        prefix = "_".join(parts[:2])
    else:
        prefix = base
    if level3 is None:
        return prefix
    return f"{level3}_{prefix}"

# ---------- Session State ----------

if "files_df" not in st.session_state:
    st.session_state.files_df = None
if "data_by_level2" not in st.session_state:
    st.session_state.data_by_level2 = None
if "signal_col" not in st.session_state:
    st.session_state.signal_col = None

# ---------- Sidebar: Step 1 & 2 ----------

st.sidebar.header("Step 1: Upload ZIP")
uploaded_zip = st.sidebar.file_uploader("Upload ZIP of CSV folders", type="zip")

sample_columns = None

if uploaded_zip:
    # Peek first CSV to get columns
    with zipfile.ZipFile(uploaded_zip, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if csv_names:
            first_csv = csv_names[0]
            with zf.open(first_csv) as f:
                sample_df = pd.read_csv(f)
            sample_columns = sample_df.columns.tolist()

if uploaded_zip and sample_columns:
    st.sidebar.header("Step 2: Select Signal Column")
    signal_col = st.sidebar.selectbox(
        "Signal column to visualize (applied to all CSVs)",
        sample_columns,
        index=0,
        key="signal_col_selector",
    )

    sampling_rate = st.sidebar.number_input(
        "Sampling Rate (Hz, used in FFT/Intensity)",
        value=10000,
        min_value=1,
        step=1,
    )

    if st.sidebar.button("Build Hierarchy & Load Signals"):
        files_rows = []
        data_by_level2 = {}

        with zipfile.ZipFile(uploaded_zip, "r") as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".csv"):
                    continue

                level1, level2, level3, filename = parse_levels_from_path(name)
                # Only use CSVs that are in Level3 folder as you described
                if level1 is None or level2 is None or level3 is None:
                    continue

                with zf.open(name) as f:
                    df = pd.read_csv(f)

                if signal_col not in df.columns:
                    continue

                sig = pd.to_numeric(df[signal_col], errors="coerce").dropna().reset_index(drop=True)
                if sig.empty:
                    continue

                label = build_label(level3, filename)

                files_rows.append({
                    "RelPath": name,
                    "Level1": level1,
                    "Level2": level2,
                    "Level3": level3,
                    "FileName": filename,
                    "Label": label,
                })

                rec = {
                    "Level1": level1,
                    "Level2": level2,
                    "Level3": level3,
                    "FileName": filename,
                    "Label": label,
                    "signal": sig,
                }
                data_by_level2.setdefault(level2, []).append(rec)

        if not files_rows:
            st.warning("No valid CSV files found with the selected signal column.")
        else:
            st.session_state.files_df = pd.DataFrame(files_rows)
            st.session_state.data_by_level2 = data_by_level2
            st.session_state.signal_col = signal_col
            st.session_state.sampling_rate = sampling_rate
            st.success("✅ Hierarchy extracted and signals loaded.")

# ---------- Main Layout ----------

st.title("Weld Signal Transformation Explorer")

if st.session_state.files_df is not None and st.session_state.data_by_level2 is not None:
    st.subheader("Extracted Folder Hierarchy (Level 1–3 + Files)")
    st.dataframe(st.session_state.files_df, use_container_width=True)

    data_by_level2 = st.session_state.data_by_level2

    if not data_by_level2:
        st.info("No data loaded. Check the ZIP structure and signal column.")
    else:
        # Tabs (same structure as original app)
        tabs = st.tabs([
            "Raw Signal",
            "Smoothed",
            "Low-pass Filter",
            "Curve Fit",
            "FFT Band Intensity",
            "Signal Intensity (dB)",
        ])

        # ---------- Tab 0: Raw Signal ----------
        with tabs[0]:
            st.subheader("Raw Signal (per Level 2 category)")

            for level2 in sorted(data_by_level2.keys()):
                with st.expander(f"{level2}", expanded=True):
                    fig = go.Figure()
                    for rec in data_by_level2[level2]:
                        y = rec["signal"].to_numpy()
                        x = np.arange(len(y))
                        fig.add_trace(go.Scatter(
                            x=x,
                            y=y,
                            mode="lines",
                            name=rec["Label"],
                        ))
                    fig.update_layout(
                        title=f"Raw Signal - {level2}",
                        xaxis_title="Index",
                        yaxis_title=st.session_state.signal_col or "Signal Value",
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # ---------- Tab 1: Smoothed ----------
        with tabs[1]:
            st.subheader("Savitzky-Golay Smoothed Signal (per Level 2 category)")

            window = st.slider("Savitzky-Golay Window Length", 5, 101, 51, step=2, key="sg_window")
            if window % 2 == 0:
                window += 1  # ensure odd
            poly = st.slider("Polynomial Order", 2, 5, 2, key="sg_poly")
            use_step = st.checkbox("Display as Step Line (Smoothed)", value=False, key="sg_step")
            step_interval = st.slider(
                "Step Interval (points) - Smoothed",
                10, 500, 70,
                key="sg_step_interval"
            ) if use_step else None

            for level2 in sorted(data_by_level2.keys()):
                with st.expander(f"{level2}", expanded=False):
                    fig = go.Figure()
                    for rec in data_by_level2[level2]:
                        data = rec["signal"].to_numpy()
                        if len(data) < window:
                            continue
                        smoothed = savgol_filter(data, window, poly)
                        x_vals = np.arange(len(smoothed))
                        if use_step:
                            x_vals, smoothed = aggregate_for_step(x_vals, smoothed, step_interval)
                            shape = "hv"
                        else:
                            shape = "linear"

                        fig.add_trace(go.Scatter(
                            x=x_vals,
                            y=smoothed,
                            mode="lines",
                            name=rec["Label"],
                            line=dict(shape=shape),
                        ))
                    fig.update_layout(
                        title=f"Smoothed Signal - {level2}",
                        xaxis_title="Index",
                        yaxis_title=st.session_state.signal_col or "Signal Value",
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # ---------- Tab 2: Low-pass Filter ----------
        with tabs[2]:
            st.subheader("Low-pass Filtered Signal (per Level 2 category)")

            cutoff = st.slider("Low-pass Cutoff Frequency (normalized 0–0.5)", 0.01, 0.5, 0.1, key="lp_cutoff")
            order = st.slider("Filter Order", 1, 5, 2, key="lp_order")
            use_step = st.checkbox("Display as Step Line (Low-pass)", value=False, key="lp_step")
            step_interval = st.slider(
                "Step Interval (points) - Low-pass",
                10, 500, 70,
                key="lp_step_interval"
            ) if use_step else None

            for level2 in sorted(data_by_level2.keys()):
                with st.expander(f"{level2}", expanded=False):
                    fig = go.Figure()
                    for rec in data_by_level2[level2]:
                        data = rec["signal"].to_numpy()
                        if len(data) < order * 3:
                            continue
                        b, a = butter(order, cutoff, btype="low", analog=False)
                        try:
                            filtered = filtfilt(b, a, data)
                        except Exception:
                            continue
                        x_vals = np.arange(len(filtered))
                        if use_step:
                            x_vals, filtered = aggregate_for_step(x_vals, filtered, step_interval)
                            shape = "hv"
                        else:
                            shape = "linear"

                        fig.add_trace(go.Scatter(
                            x=x_vals,
                            y=filtered,
                            mode="lines",
                            name=rec["Label"],
                            line=dict(shape=shape),
                        ))
                    fig.update_layout(
                        title=f"Low-pass Filtered Signal - {level2}",
                        xaxis_title="Index",
                        yaxis_title=st.session_state.signal_col or "Signal Value",
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # ---------- Tab 3: Curve Fit ----------
        with tabs[3]:
            st.subheader("Polynomial Curve Fit (per Level 2 category)")

            deg = st.slider("Curve Fit Polynomial Degree", 1, 50, 10, key="cf_deg")
            use_step = st.checkbox("Display as Step Line (Curve Fit)", value=False, key="cf_step")
            step_interval = st.slider(
                "Step Interval (points) - Curve Fit",
                10, 500, 70,
                key="cf_step_interval"
            ) if use_step else None

            for level2 in sorted(data_by_level2.keys()):
                with st.expander(f"{level2}", expanded=False):
                    fig = go.Figure()
                    for rec in data_by_level2[level2]:
                        data = rec["signal"].to_numpy()
                        x = np.arange(len(data))
                        if len(x) <= deg:
                            continue
                        try:
                            coeffs = np.polyfit(x, data, deg)
                            fitted = np.polyval(coeffs, x)
                        except Exception:
                            continue

                        x_plot = x
                        y_plot = fitted
                        if use_step:
                            x_plot, y_plot = aggregate_for_step(x_plot, y_plot, step_interval)
                            shape = "hv"
                        else:
                            shape = "linear"

                        fig.add_trace(go.Scatter(
                            x=x_plot,
                            y=y_plot,
                            mode="lines",
                            name=rec["Label"],
                            line=dict(shape=shape),
                        ))
                    fig.update_layout(
                        title=f"Curve Fit Signal - {level2}",
                        xaxis_title="Index",
                        yaxis_title=st.session_state.signal_col or "Signal Value",
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # ---------- Tab 4: FFT Band Intensity ----------
        with tabs[4]:
            st.subheader("FFT Spectrum (Zoomed Band, in dB, per Level 2 category)")

            sr_fft = st.number_input(
                "Sampling Rate (Hz) - Spectrum",
                value=int(st.session_state.get("sampling_rate", 10000)),
                min_value=1,
                key="fft_sr",
            )

            band_low, band_high = st.slider(
                "Frequency Band (Hz) - Spectrum",
                min_value=0,
                max_value=1000,
                value=(100, 200),
                step=10,
                key="fft_band",
            )

            for level2 in sorted(data_by_level2.keys()):
                with st.expander(f"{level2}", expanded=False):
                    fig = go.Figure()

                    for rec in data_by_level2[level2]:
                        data = rec["signal"].to_numpy()
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
                        magnitude_zoom = magnitude_db[mask]

                        fig.add_trace(go.Scatter(
                            x=freqs_zoom,
                            y=magnitude_zoom,
                            mode="lines",
                            name=rec["Label"],
                            fill="tozeroy",
                            fillcolor="rgba(0,0,0,0.05)",
                        ))

                    fig.update_layout(
                        title=f"FFT Spectrum (Zoomed {band_low}-{band_high} Hz, in dB) - {level2}",
                        xaxis_title="Frequency (Hz)",
                        yaxis_title="Signal Intensity (dB)",
                        xaxis=dict(range=[band_low, band_high]),
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # ---------- Tab 5: Signal Intensity (dB) ----------
        with tabs[5]:
            st.subheader("Signal Intensity (dB) Over Time (per Level 2 category)")

            sr_int = st.number_input(
                "Sampling Rate (Hz) - Intensity",
                value=int(st.session_state.get("sampling_rate", 10000)),
                min_value=1,
                key="int_sr",
            )

            use_exact_freq = st.checkbox(
                "Use Exact Frequency (Hz) Instead of Band", key="int_exact"
            )

            if use_exact_freq:
                exact_freq = st.number_input(
                    "Exact Frequency (Hz)", value=150, min_value=0, max_value=1000, step=10, key="int_exact_freq"
                )
            else:
                band_low_int, band_high_int = st.slider(
                    "Frequency Band (Hz) - Intensity",
                    0, 1000, (130, 170), step=10, key="int_band"
                )

            for level2 in sorted(data_by_level2.keys()):
                with st.expander(f"{level2}", expanded=False):
                    fig = go.Figure()

                    for rec in data_by_level2[level2]:
                        data = rec["signal"].to_numpy()
                        n = len(data)
                        if n < 16:
                            continue

                        # Spectrogram-like defaults
                        nperseg = min(1024, n // 4)
                        if nperseg < 4:
                            continue
                        noverlap = int(0.99 * nperseg)
                        step_points = max(1, nperseg - noverlap)
                        nfft = min(2048, 4 ** int(np.ceil(np.log2(nperseg * 2))))

                        times = []
                        intensities = []

                        for start in range(0, n - nperseg, step_points):
                            segment = data[start:start + nperseg]
                            fft_vals = np.fft.rfft(segment, n=nfft)
                            freqs = np.fft.rfftfreq(nfft, d=1.0 / sr_int)

                            if use_exact_freq:
                                idx = (np.abs(freqs - exact_freq)).argmin()
                                band_mag = np.abs(fft_vals[idx])
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

                        fig.add_trace(go.Scatter(
                            x=times,
                            y=intensities,
                            mode="lines",
                            name=rec["Label"],
                        ))

                    fig.update_layout(
                        title=f"Signal Intensity (dB) Over Time - {level2}",
                        xaxis_title="Time (seconds)",
                        yaxis_title="Intensity (dB)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload a ZIP and click 'Build Hierarchy & Load Signals' in the sidebar to begin.")
