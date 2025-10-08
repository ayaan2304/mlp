import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.sentiment import detect_sentiment, check_alignment
from src.generator import generate_text

st.set_page_config(page_title="AI Sentiment Text Generator", page_icon="🤖", layout="centered")

st.title("🎭 Sentiment-Aligned AI Text Generator")

st.markdown("Enter a short prompt below, and the AI will generate a paragraph matching the detected or selected sentiment.")

user_input = st.text_area("📝 Enter your prompt:", height=120)

manual = st.checkbox("Select sentiment manually (optional)")
if manual:
    sentiment_choice = st.selectbox("Select sentiment:", ["positive", "neutral", "negative"])
else:
    sentiment_choice = None

length = st.slider("🧩 Max words (approx):", min_value=50, max_value=300, value=150)

if st.button("🚀 Generate Text"):
    if not user_input.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Analyzing sentiment..."):
            detected, score = detect_sentiment(user_input)
            used_sentiment = sentiment_choice if manual else detected

        st.write(f"**Detected sentiment:** {detected} (confidence {score:.2f})")
        st.write(f"**Using sentiment:** {used_sentiment}")

        with st.spinner("Generating text..."):
            gen = generate_text(user_input, used_sentiment, max_length=length)

        st.subheader("🪶 Generated paragraph")
        st.write(gen)

        # ✅ Alignment check
        with st.spinner("Checking sentiment alignment..."):
            aligned, al_pred, al_score = check_alignment(gen, used_sentiment)

        st.markdown(f"**Alignment check:** predicted `{al_pred}` "
                    f"(confidence {al_score:.2f}) — {'✅ Aligned' if aligned else '❌ Not aligned'}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Hugging Face Transformers.")
