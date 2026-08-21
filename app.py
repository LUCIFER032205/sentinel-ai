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
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;700;800&display=swap');

        :root {
            --bg: #080b14;
            --glass: rgba(255, 255, 255, 0.04);
            --glass-border: rgba(255, 255, 255, 0.08);
            --ink: #f1f5f9;
            --muted: #64748b;
            --accent: #22d3ee;
            --violet: #818cf8;
            --ok: rgba(34, 197, 94, 0.12);
            --ok-glow: rgba(34, 197, 94, 0.45);
            --bad: rgba(239, 68, 68, 0.12);
            --bad-glow: rgba(239, 68, 68, 0.45);
            --warn: rgba(251, 191, 36, 0.12);
            --warn-glow: rgba(251, 191, 36, 0.45);
        }

        .stApp {
            background-color: var(--bg);
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='32' height='32'%3E%3Ccircle cx='1' cy='1' r='1' fill='rgba(255,255,255,0.04)'/%3E%3C/svg%3E");
            font-family: "Inter", sans-serif;
            color: var(--ink);
        }

        h1, h2, h3 {
            font-family: "Space Grotesk", sans-serif !important;
        }

        .hero {
            background: linear-gradient(135deg, rgba(34,211,238,0.06) 0%, rgba(129,140,248,0.06) 100%);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 2.5rem 2rem;
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
        }

        .hero::before {
            content: "";
            position: absolute;
            top: -60px; left: -60px;
            width: 320px; height: 320px;
            background: radial-gradient(circle, rgba(34,211,238,0.12) 0%, transparent 70%);
            pointer-events: none;
        }

        .hero .kicker {
            font-size: 0.78rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            font-weight: 600;
            color: var(--accent);
            margin-bottom: 0.6rem;
        }

        .hero .title {
            font-family: "Space Grotesk", sans-serif;
            font-size: 2.8rem;
            font-weight: 800;
            color: var(--ink);
            text-shadow: 0 0 40px rgba(34, 211, 238, 0.5);
            margin-bottom: 0.5rem;
            line-height: 1.15;
        }

        .hero .subtitle {
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.6;
            max-width: 52rem;
            margin-bottom: 1rem;
        }

        .hero .chip {
            display: inline-block;
            padding: 0.3rem 0.75rem;
            border-radius: 999px;
            border: 1px solid rgba(34, 211, 238, 0.3);
            background: rgba(34, 211, 238, 0.08);
            color: var(--accent);
            font-size: 0.82rem;
            font-weight: 600;
            margin-right: 0.5rem;
        }

        .glass-card {
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }

        .soft-card {
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            padding: 1.5rem 1.4rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            max-width: 860px;
            margin: 0 auto;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(12rem, 1fr));
            gap: 0.75rem;
            margin-top: 0.75rem;
            margin-bottom: 0.5rem;
        }

        .metric-card {
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--glass-border);
            padding: 0.85rem 1rem;
        }

        .metric-label {
            font-size: 0.74rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
        }

        .metric-value {
            font-size: 1.4rem;
            margin-top: 0.2rem;
            font-weight: 800;
            color: var(--ink);
        }

        .conf-bar-wrap {
            width: 100%;
            height: 6px;
            background: rgba(255,255,255,0.06);
            border-radius: 999px;
            margin: 0.6rem 0 0.8rem;
            overflow: hidden;
        }

        .conf-bar-fill {
            height: 100%;
            border-radius: 999px;
            transition: width 0.4s ease;
        }

        .conf-bar-fill.ai   { background: #ef4444; box-shadow: 0 0 8px rgba(239,68,68,0.7); }
        .conf-bar-fill.real { background: #22c55e; box-shadow: 0 0 8px rgba(34,197,94,0.7); }
        .conf-bar-fill.uncertain { background: #fbbf24; box-shadow: 0 0 8px rgba(251,191,36,0.7); }

        .mode-intro {
            color: var(--muted);
            margin-bottom: 1rem;
            max-width: 48rem;
            font-size: 0.95rem;
        }

        .empty-state {
            padding: 2rem 1.5rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.02);
            border: 1.5px dashed rgba(255, 255, 255, 0.08);
            color: var(--muted);
            margin-top: 0.5rem;
            text-align: center;
        }

        .empty-state strong {
            display: block;
            color: var(--ink);
            margin-bottom: 0.4rem;
            font-size: 1rem;
        }

        [data-testid="stDataFrame"] {
            background: rgba(255,255,255,0.02) !important;
            border-radius: 12px;
            border: 1px solid var(--glass-border);
        }

        .tab-note {
            color: var(--muted);
            font-size: 0.9rem;
            margin-bottom: 0.8rem;
        }

        .decision-pill {
            display: inline-block;
            padding: 0.45rem 1rem;
            border-radius: 999px;
            font-size: 0.9rem;
            font-weight: 700;
            margin-bottom: 0.6rem;
            letter-spacing: 0.02em;
        }

        .decision-ai {
            color: #fca5a5;
            background: var(--bad);
            border: 1px solid rgba(239, 68, 68, 0.35);
            box-shadow: 0 0 16px var(--bad-glow);
        }

        .decision-real {
            color: #86efac;
            background: var(--ok);
            border: 1px solid rgba(34, 197, 94, 0.35);
            box-shadow: 0 0 16px var(--ok-glow);
        }

        .decision-uncertain {
            color: #fde68a;
            background: var(--warn);
            border: 1px solid rgba(251, 191, 36, 0.35);
            box-shadow: 0 0 16px var(--warn-glow);
        }

        .footer-note {
            color: var(--muted);
            font-size: 0.88rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            margin-bottom: 1.2rem;
            background: rgba(255,255,255,0.03);
            border-radius: 999px;
            padding: 0.25rem;
            border: 1px solid var(--glass-border);
            width: fit-content;
        }

        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border: none;
            border-radius: 999px;
            padding: 0.45rem 1.1rem;
            color: var(--muted);
            font-weight: 600;
            font-size: 0.9rem;
            height: auto;
            transition: all 0.2s;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(34, 211, 238, 0.12);
            color: var(--accent);
            border: 1px solid rgba(34, 211, 238, 0.25);
            box-shadow: 0 0 12px rgba(34, 211, 238, 0.15);
        }

        [data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 14px;
            padding: 0.5rem;
            border: 1.5px dashed rgba(34, 211, 238, 0.3);
            transition: border-color 0.2s;
        }

        [data-testid="stFileUploader"]:hover {
            border-color: rgba(34, 211, 238, 0.6);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="kicker">Visual Forensics · AI Detection</div>
            <div class="title">Can you tell what's real?</div>
            <div class="subtitle">
                Upload any image. The model tells you if a human or an AI made it.
            </div>
            <span class="chip">88.8% Accuracy</span>
            <span class="chip">1,000-image test set</span>
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


def confidence_bar_html(ai_prob: float, label: str) -> str:
    pct = int(min(max(ai_prob, 0.0), 1.0) * 100)
    css_class = (
        "ai" if label == "AI-generated"
        else "real" if label == "Real"
        else "uncertain"
    )
    return (
        f'<div class="conf-bar-wrap">'
        f'<div class="conf-bar-fill {css_class}" style="width:{pct}%"></div>'
        f'</div>'
    )



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
        st.markdown(confidence_bar_html(item["ai_prob_raw"], item["Label"]), unsafe_allow_html=True)
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
    st.markdown(confidence_bar_html(chosen["ai_prob_raw"], chosen["Label"]), unsafe_allow_html=True)
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
