"""Streamlit interactive browser demo for NeuroQA EEG artifact detection."""

import httpx
import numpy as np
import plotly.graph_objects as go
import streamlit as st

CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]
DISPLAY_CHANNELS = ["Fp1", "Fp2", "F3", "F4"]
SAMPLING_RATE = 256
N_SAMPLES = 512
TIME_AXIS = np.linspace(0.0, N_SAMPLES / SAMPLING_RATE, N_SAMPLES, endpoint=False)


@st.cache_data
def generate_demo_signal() -> np.ndarray:
    """Generate synthetic EEG signal with simulated ocular artifact.

    Returns:
        Multichannel EEG signal of shape (19, 512).
    """
    np.random.seed(42)
    sig = np.random.randn(19, 512).astype(np.float32)
    sig[0, 100:150] *= 15.0
    return sig


def create_signal_figure(
    sig: np.ndarray,
    title: str = "EEG Signal Preview (4 of 19 channels)",
    highlight_windows: list[tuple[float, float, float]] | None = None,
) -> go.Figure:
    """Create a Plotly multichannel EEG signal visualization.

    Args:
        sig: Multichannel EEG signal array.
        title: Title of the chart.
        highlight_windows: Optional list of (start_sec, end_sec, score) tuples.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()
    colors = ["#2b5c8f", "#d95f02", "#1b9e77", "#7570b3"]
    offsets = [(3 - i) * 100.0 for i in range(4)]

    for i, ch_name in enumerate(DISPLAY_CHANNELS):
        offset = offsets[i]
        trace_y = sig[i] + offset
        fig.add_trace(
            go.Scatter(
                x=TIME_AXIS,
                y=trace_y,
                mode="lines",
                name=ch_name,
                line=dict(color=colors[i % len(colors)], width=1.5),
                hovertemplate=f"<b>{ch_name}</b><br>Time: %{{x:.3f}}s<br>Amplitude: %{{customdata:.2f}} µV<extra></extra>",
                customdata=sig[i],
            )
        )

    if highlight_windows:
        for idx, (start_sec, end_sec, _) in enumerate(highlight_windows):
            fig.add_vrect(
                x0=start_sec,
                x1=end_sec,
                fillcolor="red",
                opacity=0.22,
                layer="below",
                line_width=1.5,
                line_color="crimson",
                line_dash="dot",
                annotation_text=f"Window {idx + 1}",
                annotation_position="top left",
                annotation=dict(font_size=11, font_color="#990000"),
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#2c3e50")),
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude (µV)",
        yaxis=dict(
            tickmode="array",
            tickvals=offsets,
            ticktext=DISPLAY_CHANNELS,
            zeroline=False,
            showgrid=True,
            gridcolor="#f0f0f0",
        ),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        margin=dict(l=60, r=40, t=50, b=40),
        height=380,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    return fig


def render_plotly_chart(fig: go.Figure) -> None:
    """Render Plotly figure with forward and backwards compatibility for Streamlit."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    """Main execution entry point for Streamlit application."""
    st.set_page_config(
        page_title="NeuroQA — EEG Artifact Detection",
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 NeuroQA")
    st.subheader("Explainable EEG Artifact Detection Platform")

    st.sidebar.header("Signal Input")
    input_source = st.sidebar.radio(
        "Input Mode",
        options=["Upload .npy file", "Generate demo signal"],
        index=1,
    )

    signal: np.ndarray | None = None

    if input_source == "Upload .npy file":
        uploaded_file = st.sidebar.file_uploader(
            "Upload 19-channel EEG file (.npy)",
            type=["npy"],
        )
        if uploaded_file is not None:
            signal = np.load(uploaded_file)

            # Step 1: Fix orientation if shape is (n_samples, 19)
            if signal.ndim == 2 and signal.shape[1] == 19 and signal.shape[0] != 19:
                signal = signal.T  # now shape is (19, n_samples)

            # Step 2: Take 512 samples from the middle of the signal
            if signal.ndim == 2 and signal.shape[0] == 19:
                n_samples = signal.shape[1]
                if n_samples >= 512:
                    mid = n_samples // 2
                    signal = signal[:, mid - 256 : mid + 256]  # 512 samples
                    signal = signal.astype(np.float32)
                else:
                    st.error(f"Signal too short: {n_samples} samples. Need at least 512.")
                    st.stop()
            else:
                st.error(f"Cannot process shape {signal.shape}. Need a 2D array with 19 channels.")
                st.stop()

            # Step 3: Final validation
            if signal.shape != (19, 512):
                st.error(f"Processing failed. Final shape {signal.shape} is not (19, 512).")
                st.stop()

            st.success(f"Signal loaded successfully. Shape: {signal.shape}")
        else:
            st.info("Please upload a .npy file with shape (19, 512) from the sidebar.")
    else:
        signal = generate_demo_signal()
        st.sidebar.caption(
            "Synthetic 19-channel, 512-sample recording with simulated ocular artifact on Fp1."
        )

    st.sidebar.markdown("---")
    st.sidebar.header("API Configuration")
    api_target = st.sidebar.radio(
        "API Server",
        options=[
            "Local (http://localhost:8000)",
            "Render Cloud (https://neuroqa-api.onrender.com)",
            "Custom",
        ],
        index=0,
        key="api_target_radio",
    )
    if api_target == "Local (http://localhost:8000)":
        api_url = "http://localhost:8000"
    elif api_target == "Render Cloud (https://neuroqa-api.onrender.com)":
        api_url = "https://neuroqa-api.onrender.com"
    else:
        api_url = st.sidebar.text_input(
            "Custom API URL",
            value="http://localhost:8000",
            key="custom_api_url_input",
        )

    # Show live connection badge in the sidebar
    try:
        health_resp = httpx.get(f"{api_url.rstrip('/')}/health", timeout=2.0)
        if health_resp.status_code == 200:
            st.sidebar.success(f"🟢 **Connected**: `{api_url}`")
        else:
            st.sidebar.warning(f"🟡 Server returned HTTP {health_resp.status_code}")
    except httpx.ConnectError:
        if "localhost" in api_url or "127.0.0.1" in api_url:
            st.sidebar.error("🔴 **Local server not detected**\n\nRun:\n`uvicorn neuroqa.api.app:app --reload`")
        else:
            st.sidebar.error("🔴 **Cloud server unreachable**\n\nRender may be sleeping or building.")
    except Exception:
        st.sidebar.warning(f"🟡 Connection check failed for `{api_url}`")

    run_detection = st.sidebar.button("🔍 Run Detection", type="primary")

    if signal is not None:
        st.markdown("#### Input Signal Preview")
        preview_fig = create_signal_figure(
            signal,
            title="EEG Signal Preview (4 of 19 channels)",
        )
        render_plotly_chart(preview_fig)

        if run_detection:
            with st.spinner("Analysing EEG segment..."):
                try:
                    predict_endpoint = f"{api_url.rstrip('/')}/predict"
                    response = httpx.post(
                        predict_endpoint,
                        json={"eeg_segment": signal.tolist()},
                        timeout=15.0,
                    )
                    if response.status_code == 200:
                        result = response.json()
                        prediction = result.get("prediction", "")
                        confidence = float(result.get("confidence", 0.0))
                        top_windows = result.get("top_time_windows", [])
                        processing_time_ms = float(result.get("processing_time_ms", 0.0))

                        st.markdown("---")
                        st.subheader("Classification & Explainability Analysis")

                        col_left, col_right = st.columns(2)

                        with col_left:
                            st.markdown("#### Prediction Result")
                            if prediction == "artifact":
                                st.error("⚠️ ARTIFACT DETECTED")
                            else:
                                st.success("✅ CLEAN SIGNAL")

                            m_col1, m_col2 = st.columns(2)
                            with m_col1:
                                st.metric("Confidence", f"{confidence:.1%}")
                            with m_col2:
                                st.metric("Processing Time", f"{processing_time_ms:.1f} ms")

                        with col_right:
                            st.markdown("#### 🔍 Top Suspicious Time Windows")
                            max_score = max([w[2] for w in top_windows], default=1.0)
                            for idx, (start_sec, end_sec, score) in enumerate(top_windows):
                                st.write(
                                    f"**Window {idx + 1}**: `{start_sec:.3f}s` → `{end_sec:.3f}s` "
                                    f"(importance: `{score:.4f}`)"
                                )
                                progress_val = (
                                    float(score / max_score)
                                    if max_score > 0
                                    else float(score)
                                )
                                st.progress(min(max(progress_val, 0.0), 1.0))

                            st.caption(
                                "These are the time windows the model weighted most heavily in its decision."
                            )

                        st.markdown("---")
                        st.markdown("#### Annotated Signal")
                        annotated_fig = create_signal_figure(
                            signal,
                            title="EEG Signal with Suspicious Windows Highlighted",
                            highlight_windows=top_windows,
                        )
                        render_plotly_chart(annotated_fig)

                    elif response.status_code == 404:
                        st.error(
                            f"API endpoint not found (HTTP 404) at `{predict_endpoint}`.\n\n"
                            "• If running locally, select **Local (http://localhost:8000)** in the sidebar.\n"
                            "• If using Render Cloud, the cloud service may still be building or spinning up."
                        )
                    elif response.status_code == 422:
                        detail = response.json().get("detail", response.text)
                        st.error(f"Input validation error (HTTP 422): {detail}")
                    else:
                        st.error(
                            f"API error ({response.status_code}): {response.text}"
                        )
                except httpx.ConnectError:
                    st.error(
                        "Cannot connect to API. Make sure the FastAPI server is running."
                    )
                except httpx.TimeoutException:
                    st.error("API request timed out. Please try again.")
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")

    st.divider()
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 0.9rem; line-height: 1.6;">
            <p>Built with <b>PyTorch</b> · <b>MNE-Python</b> · <b>FastAPI</b> · <b>Streamlit</b></p>
            <p>Model: <code>EEGTransformer</code> | Size: <code>0.379 MB</code> | Latency: <code>~4.5ms CPU</code></p>
            <p><a href="https://github.com/Saumyaaaaa/neuroqa" target="_blank" style="color: #2b5c8f; text-decoration: none;">GitHub Repository: Saumyaaaaa/neuroqa</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
