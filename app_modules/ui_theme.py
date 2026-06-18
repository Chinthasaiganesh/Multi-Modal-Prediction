from __future__ import annotations

import streamlit as st

from app_modules.config import THEME_TOKENS


def inject_custom_theme(theme_name: str) -> None:
    tokens = THEME_TOKENS.get(theme_name, THEME_TOKENS["light"])

    css_vars = (
        "<style>\n"
        ":root {\n"
        f"    --bg-1: {tokens['bg_1']};\n"
        f"    --bg-2: {tokens['bg_2']};\n"
        f"    --bg-3: {tokens['bg_3']};\n"
        f"    --ink: {tokens['ink']};\n"
        f"    --muted: {tokens['ink_soft']};\n"
        f"    --heading: {tokens['heading']};\n"
        f"    --accent: {tokens['accent']};\n"
        f"    --accent-deep: {tokens['accent_deep']};\n"
        f"    --accent-2: {tokens['accent_warm']};\n"
        f"    --glass: {tokens['glass']};\n"
        f"    --glass-border: {tokens['glass_border']};\n"
        f"    --surface: {tokens['surface']};\n"
        f"    --field-bg: {tokens['field_bg']};\n"
        f"    --field-border: {tokens['field_border']};\n"
        f"    --shadow: {tokens['shadow']};\n"
        "    --radius: 18px;\n"
        "}\n"
        "</style>"
    )
    st.markdown(css_vars, unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Fraunces:opsz,wght@9..144,500;9..144,700&display=swap');

        .stApp {
            background:
                radial-gradient(1200px 500px at 7% -10%, color-mix(in srgb, var(--accent) 22%, transparent), transparent 60%),
                radial-gradient(900px 420px at 100% 15%, color-mix(in srgb, var(--accent-2) 18%, transparent), transparent 60%),
                linear-gradient(140deg, var(--bg-1), var(--bg-2) 55%, var(--bg-3));
            color: var(--ink);
            font-family: 'Space Grotesk', sans-serif;
        }

        .block-container {
            max-width: 1240px;
            padding-top: 1.4rem;
            padding-bottom: 2.2rem;
        }

        h1, h2, h3 {
            font-family: 'Fraunces', serif;
            color: var(--heading);
            letter-spacing: -0.02em;
        }

        .hero {
            border: 1px solid var(--glass-border);
            background: linear-gradient(135deg, var(--surface), var(--glass));
            box-shadow: var(--shadow);
            border-radius: calc(var(--radius) + 6px);
            padding: 1.25rem 1.4rem;
            text-align: center;
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            animation: riseIn 0.6s ease-out;
        }

        .hero-top {
            display: flex;
            align-items: center;
            justify-content: center;
            flex-wrap: wrap;
            gap: 0.8rem;
            margin-bottom: 0.35rem;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.35rem 0.68rem;
            border-radius: 999px;
            border: 1px solid color-mix(in srgb, var(--accent) 28%, transparent);
            background: color-mix(in srgb, var(--accent) 12%, transparent);
            color: var(--accent);
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .hero-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: var(--accent);
            box-shadow: 0 0 0 6px color-mix(in srgb, var(--accent) 18%, transparent);
            animation: pulseDot 2.2s ease-in-out infinite;
        }

        .hero-pill {
            padding: 0.32rem 0.62rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            color: var(--muted);
            border: 1px solid var(--glass-border);
            background: color-mix(in srgb, var(--surface) 88%, transparent);
            white-space: nowrap;
        }

        .eyebrow {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.11em;
            color: var(--accent);
            font-weight: 700;
        }

        .hero-title {
            margin: 0.3rem auto 0.2rem;
            font-family: 'Fraunces', serif;
            font-weight: 700;
            font-size: clamp(1.85rem, 3.2vw, 3.05rem);
            line-height: 1.04;
            max-width: 16ch;
            text-align: center !important;
            color: var(--heading);
        }

        .hero-gradient,
        .hero-accent,
        .hero-price {
            background: linear-gradient(95deg, #0f766e 0%, #14b8a6 48%, #06b6d4 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            -webkit-text-fill-color: transparent;
        }

        .hero-sub {
            margin: 0.22rem auto 0;
            color: var(--muted);
            font-size: 1rem;
            max-width: 62ch;
            text-align: center !important;
        }

        .hero h1,
        .hero p,
        .hero div {
            text-align: center;
        }

        .hero-meta {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.45rem;
            margin-top: 0.78rem;
            flex-wrap: wrap;
        }

        .hero-meta span {
            padding: 0.32rem 0.62rem;
            border-radius: 999px;
            font-size: 0.73rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            background: color-mix(in srgb, var(--surface) 90%, transparent);
            border: 1px solid var(--glass-border);
            color: var(--heading);
        }

        .stat-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.7rem;
            margin-top: 1rem;
        }

        .stat-card {
            padding: 0.75rem;
            border-radius: 14px;
            background: color-mix(in srgb, var(--surface) 86%, transparent);
            border: 1px solid color-mix(in srgb, var(--glass-border) 92%, transparent);
        }

        .stat-k {
            font-size: 0.72rem;
            text-transform: uppercase;
            color: var(--muted);
            letter-spacing: 0.08em;
            margin: 0;
        }

        .stat-v {
            margin: 0.18rem 0 0;
            font-weight: 700;
            font-size: 1.05rem;
            color: var(--heading);
        }

        .stTabs [role="tablist"] {
            gap: 0.45rem;
            background: color-mix(in srgb, var(--surface) 78%, transparent);
            padding: 0.4rem;
            border-radius: 999px;
            border: 1px solid var(--glass-border);
            margin-bottom: 0.8rem;
        }

        .stTabs [role="tab"] {
            border-radius: 999px;
            padding: 0.5rem 1rem;
            font-weight: 700;
            color: #4a5561;
            border: none;
            transition: all 180ms ease;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(120deg, var(--accent), var(--accent-deep));
            color: #f8fafc;
            box-shadow: 0 8px 16px color-mix(in srgb, var(--accent) 30%, transparent);
        }

        .stTextInput > div > div > input,
        .stTextArea textarea,
        .stFileUploader > div,
        .stDataFrame,
        .stAlert {
            border-radius: var(--radius) !important;
        }

        .stTextInput > div > div > input,
        .stTextArea textarea {
            background: var(--field-bg);
            border: 1px solid var(--field-border);
            color: var(--ink);
            font-family: 'Space Grotesk', sans-serif;
        }

        .stTextArea textarea::placeholder,
        .stTextInput > div > div > input::placeholder {
            color: color-mix(in srgb, var(--muted) 82%, transparent);
        }

        .stButton > button,
        .stDownloadButton > button {
            border: none;
            border-radius: 999px;
            background: linear-gradient(125deg, var(--accent), var(--accent-deep));
            color: #f7fafc;
            font-weight: 700;
            letter-spacing: 0.01em;
            padding: 0.58rem 1.12rem;
            box-shadow: 0 10px 22px color-mix(in srgb, var(--accent) 34%, transparent);
            transition: transform 140ms ease, box-shadow 140ms ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 24px color-mix(in srgb, var(--accent) 45%, transparent);
        }

        .feature-card {
            border: 1px solid var(--glass-border);
            border-radius: var(--radius);
            background: color-mix(in srgb, var(--surface) 84%, transparent);
            box-shadow: var(--shadow);
            padding: 0.9rem;
            margin-top: 0.6rem;
        }

        .prediction-output {
            margin-top: 1rem;
            border-radius: calc(var(--radius) + 2px);
            border: 1px solid color-mix(in srgb, var(--accent) 34%, var(--glass-border));
            background: linear-gradient(
                120deg,
                color-mix(in srgb, var(--accent) 12%, var(--surface)),
                color-mix(in srgb, var(--accent-deep) 10%, var(--surface))
            );
            box-shadow: 0 14px 32px color-mix(in srgb, var(--accent) 24%, transparent);
            padding: 1.15rem 1.25rem;
        }

        .prediction-title {
            margin: 0;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--muted);
        }

        .prediction-price {
            margin: 0.35rem 0 0.5rem;
            font-family: 'Fraunces', serif;
            font-size: clamp(1.8rem, 3.4vw, 2.35rem);
            line-height: 1.1;
            color: var(--heading);
        }

        .prediction-band {
            margin: 0;
            font-size: 1rem;
            color: var(--ink);
            font-weight: 600;
        }

        .stImage img {
            border-radius: calc(var(--radius) + 2px);
            box-shadow: var(--shadow);
        }

        div[data-testid="stVerticalBlock"] > div:has(> div > div > .stTextInput),
        div[data-testid="stVerticalBlock"] > div:has(> div > div > .stTextArea),
        div[data-testid="stVerticalBlock"] > div:has(> div > div > .stFileUploader) {
            animation: riseIn 0.45s ease both;
        }

        @keyframes riseIn {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes pulseDot {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.15); opacity: 0.74; }
        }

        @media (max-width: 900px) {
            .block-container {
                padding-left: 0.85rem;
                padding-right: 0.85rem;
            }

            .hero-top {
                align-items: flex-start;
                flex-direction: column;
            }

            .hero-title {
                max-width: none;
            }

            .stat-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <section class="hero">
            <div class="hero-top">
                <div class="hero-badge"><span class="hero-dot"></span> Enterprise-ready pricing intelligence</div>
            </div>
            <h1 class="hero-title" style="text-align:center; margin-left:auto; margin-right:auto;"><span class="hero-gradient">Multi Model</span> <span class="hero-price">Price</span> <span class="hero-accent">Prediction</span></h1>
            <p class="hero-sub" style="text-align:center; margin-left:auto; margin-right:auto;">
                A polished inference dashboard that fuses language and engineered product signals
                into client-ready pricing estimates.
            </p>
            <div class="hero-meta">
                <span>Predictive analytics</span>
                <span>Fast quote estimation</span>
                <span>Single + Batch scoring</span>
            </div>
            <div class="stat-grid">
                <div class="stat-card">
                    <p class="stat-k">Model Stack</p>
                    <p class="stat-v">LightGBM + XGBoost</p>
                </div>
                <div class="stat-card">
                    <p class="stat-k">Feature Fusion</p>
                    <p class="stat-v">TF-IDF + Numeric + Visual/BERT placeholders</p>
                </div>
                <div class="stat-card">
                    <p class="stat-k">Use Cases</p>
                    <p class="stat-v">Single SKU and bulk CSV scoring</p>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_theme_switcher() -> str:
    st.sidebar.header("Presentation")
    is_dark = st.sidebar.toggle("Dark Theme", value=st.session_state.get("theme_name") == "dark")
    theme_name = "dark" if is_dark else "light"
    st.session_state["theme_name"] = theme_name
    st.sidebar.caption("Theme applies instantly to the full dashboard.")

    st.sidebar.markdown("---")
    st.sidebar.header("Navigation")
    if st.sidebar.button("← Back to Home", use_container_width=True, key="back_button"):
        st.session_state.page = "landing"
        st.rerun()

    return theme_name
