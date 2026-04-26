from __future__ import annotations

import streamlit as st
from PIL import Image

from src.ai_image_detector.config import (
    MODEL_PATH,
)
from src.ai_image_detector.inference import (
    CalibrationConfig,
    load_trained_model,
    predict_image_bytes,
)


st.set_page_config(
    page_title="AI Image Detector",
    page_icon="📷",
    layout="wide",
)


@st.cache_resource
def get_model():
    return load_trained_model()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

        :root {
            --bg-a: #f4f7ff;
            --bg-b: #eafaf1;
            --ink: #0f172a;
            --muted: #475569;
            --card: rgba(255, 255, 255, 0.88);
            --line: rgba(15, 23, 42, 0.12);
            --ok: #0f9f6e;
            --warn: #e09f1f;
            --bad: #d14343;
        }

        .stApp {
            background:
                radial-gradient(65rem 28rem at -10% -10%, #dbeafe 0%, transparent 65%),
                radial-gradient(60rem 24rem at 110% -15%, #dcfce7 0%, transparent 60%),
                linear-gradient(145deg, var(--bg-a), var(--bg-b));
            font-family: "Manrope", sans-serif;
            color: var(--ink);
        }

        h1, h2, h3 {
            font-family: "Space Grotesk", sans-serif !important;
            letter-spacing: 0.2px;
        }

        .hero {
            background:
                radial-gradient(circle at top left, rgba(255, 255, 255, 0.18), transparent 28%),
                linear-gradient(135deg, rgba(15, 23, 42, 0.94), rgba(30, 64, 175, 0.88) 52%, rgba(22, 163, 74, 0.84));
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 24px;
            padding: 1.5rem 1.4rem;
            color: #f8fafc;
            box-shadow: 0 20px 48px rgba(15, 23, 42, 0.24);
            margin-bottom: 1rem;
        }

        .hero .kicker {
            font-size: 0.8rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            font-weight: 700;
            opacity: 0.95;
        }

        .hero .title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-top: 0.2rem;
            margin-bottom: 0.35rem;
        }

        .hero .subtitle {
            opacity: 0.97;
            max-width: 60rem;
            line-height: 1.6;
        }

        .chip {
            display: inline-block;
            padding: 0.35rem 0.66rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.36);
            font-size: 0.82rem;
            margin-top: 0.6rem;
            margin-right: 0.35rem;
            background: rgba(255, 255, 255, 0.14);
        }

        .soft-card {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1rem 1.05rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr));
            gap: 0.75rem;
            margin-top: 0.25rem;
            margin-bottom: 0.35rem;
        }

        .metric-card {
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.8);
            border: 1px solid var(--line);
            padding: 0.8rem 0.9rem;
        }

        .metric-label {
            font-size: 0.76rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.07em;
            font-weight: 700;
        }

        .metric-value {
            font-size: 1.35rem;
            margin-top: 0.18rem;
            font-weight: 800;
        }

        .mode-intro {
            color: var(--muted);
            margin-bottom: 0.9rem;
            max-width: 48rem;
        }

        .empty-state {
            padding: 1.4rem 1.2rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.7);
            border: 1px dashed rgba(15, 23, 42, 0.16);
            color: var(--muted);
            margin-top: 0.5rem;
        }

        .empty-state strong {
            display: block;
            color: var(--ink);
            margin-bottom: 0.3rem;
        }

        .tab-note {
            color: var(--muted);
            font-size: 0.92rem;
            margin-bottom: 0.7rem;
        }

        .decision-pill {
            display: inline-block;
            padding: 0.4rem 0.75rem;
            border-radius: 999px;
            font-size: 0.88rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
            border: 1px solid transparent;
        }
        .decision-ai {
            color: #7f1d1d;
            background: rgba(220, 38, 38, 0.14);
            border-color: rgba(220, 38, 38, 0.35);
        }
        .decision-real {
            color: #14532d;
            background: rgba(22, 163, 74, 0.14);
            border-color: rgba(22, 163, 74, 0.35);
        }
        .decision-uncertain {
            color: #78350f;
            background: rgba(217, 119, 6, 0.16);
            border-color: rgba(217, 119, 6, 0.34);
        }

        .footer-note {
            color: var(--muted);
            font-size: 0.88rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.6rem;
            margin-bottom: 0.8rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.62);
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 999px;
            padding: 0.55rem 1rem;
            height: auto;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(30, 64, 175, 0.12), rgba(22, 163, 74, 0.16));
            border-color: rgba(30, 64, 175, 0.20);
            box-shadow: 0 10px 20px rgba(15, 23, 42, 0.06);
        }

        [data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.62);
            border-radius: 18px;
            padding: 0.45rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="kicker">Visual Forensics</div>
            <div class="title">AI Image Detector</div>
            <div class="subtitle">
                Check one image or a batch in a cleaner tab-based workspace.
                Use the default scan for balanced decisions or switch to the sensitive tab
                when you want the detector to lean more aggressively toward AI signals.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def decision_class(label: str) -> str:
    if label == "AI-generated":
        return "decision-pill decision-ai"
    if label == "Real":
        return "decision-pill decision-real"
    return "decision-pill decision-uncertain"


def render_empty_state(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="empty-state">
            <strong>{title}</strong>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_detection_tab(
    *,
    key: str,
    title: str,
    description: str,
    calibration: CalibrationConfig,
    orientation_conservative: bool,
    model,
) -> None:
    st.markdown(f"### {title}")
    st.markdown(f'<div class="mode-intro">{description}</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload Image(s)",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True,
        help="Upload one image or a batch to compare results quickly.",
        key=key,
    )

    if not uploaded_files:
        render_empty_state(
            "Drop files to start a scan",
            "Your results will appear here with a preview, label, AI probability, and confidence score.",
        )
        return

    rows: list[dict] = []
    previews: dict[str, Image.Image] = {}

    for file in uploaded_files:
        image = Image.open(file).convert("RGB")
        previews[file.name] = image
        result = predict_image_bytes(
            model,
            file.getvalue(),
            calibration=calibration,
            orientation_conservative=orientation_conservative,
        )

        rows.append(
            {
                "File": file.name,
                "Label": result.label,
                "AI Probability": f"{result.ai_probability:.2%}",
                "Confidence": f"{result.confidence:.2%}",
                "ai_prob_raw": result.ai_probability,
            }
        )

    if len(rows) == 1:
        item = rows[0]
        image = previews[item["File"]]
        st.image(image, caption=item["File"], use_container_width=True)
        st.markdown(
            f'<span class="{decision_class(item["Label"])}">{item["Label"]}</span>',
            unsafe_allow_html=True,
        )
        st.progress(min(max(item["ai_prob_raw"], 0.0), 1.0))
        st.markdown(
            f"""
            <div class="metric-grid">
              <div class="metric-card">
                <div class="metric-label">AI Probability</div>
                <div class="metric-value">{item["AI Probability"]}</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{item["Confidence"]}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.dataframe(
        [{k: v for k, v in row.items() if k != "ai_prob_raw"} for row in rows],
        use_container_width=True,
        hide_index=True,
    )
    selected = st.selectbox("Preview one result", [r["File"] for r in rows], key=f"{key}_preview")
    chosen = next(row for row in rows if row["File"] == selected)
    st.image(previews[selected], caption=selected, use_container_width=True)
    st.markdown(
        f'<span class="{decision_class(chosen["Label"])}">{chosen["Label"]}</span>',
        unsafe_allow_html=True,
    )
    st.progress(min(max(chosen["ai_prob_raw"], 0.0), 1.0))
    st.caption(f"AI Probability: {chosen['AI Probability']} | Confidence: {chosen['Confidence']}")


def main() -> None:
    inject_styles()

    if not MODEL_PATH.exists():
        st.warning("No trained model found. Train first with `python train.py`, then reload.")
        st.stop()

    render_hero()

    model = get_model()
    default_tab, sensitive_tab = st.tabs(["Default Scan", "AI-Sensitive"])

    with default_tab:
        st.markdown(
            '<div class="tab-note">Balanced mode for the cleanest everyday result view.</div>',
            unsafe_allow_html=True,
        )
        render_detection_tab(
            key="default_scan",
            title="Default Scan",
            description="Use this when you want a smoother, more balanced prediction flow for normal checks.",
            calibration=CalibrationConfig(
                threshold=0.65,
                uncertain_low=0.45,
                uncertain_high=0.70,
            ),
            orientation_conservative=True,
            model=model,
        )

    with sensitive_tab:
        st.markdown(
            '<div class="tab-note">More aggressive mode when you want stronger AI catching behavior.</div>',
            unsafe_allow_html=True,
        )
        render_detection_tab(
            key="sensitive_scan",
            title="AI-Sensitive Scan",
            description="This profile reacts faster to possible AI traits and is useful when you want a stricter pass.",
            calibration=CalibrationConfig(
                threshold=0.40,
                uncertain_low=0.30,
                uncertain_high=0.50,
            ),
            orientation_conservative=False,
            model=model,
        )


if __name__ == "__main__":
    main()
