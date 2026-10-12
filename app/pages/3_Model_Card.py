"""Page 3: Model Card View — Renders MODEL_CARD.md inside the Streamlit application."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_CARD_PATH = ROOT_DIR / "MODEL_CARD.md"

st.set_page_config(
    page_title="Model Card | Credit Risk Suite",
    page_icon="📋",
    layout="wide",
)


def main() -> None:
    st.title("📋 Model Card (Mitchell et al., 2019 Framework)")
    if MODEL_CARD_PATH.exists():
        content = MODEL_CARD_PATH.read_text(encoding="utf-8")
        st.markdown(content)
    else:
        st.error("MODEL_CARD.md not found in project root.")


if __name__ == "__main__":
    main()
