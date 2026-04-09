import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")

# 🌗 Dark mode toggle
dark_mode = st.toggle("🌗 Dark Mode")

if dark_mode:
    st.markdown("""
        <style>
        .main {background-color: #121212; color: white;}
        </style>
    """, unsafe_allow_html=True)

# Load model
classifier = pipeline("sentiment-analysis")

# Title
st.title("💬 Sentiment Analysis App")
st.write("Analyze text sentiment using BERT 🤖")

st.write("---")

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

# 🌍 Multi-language input (basic)
language = st.selectbox("🌍 Select Language", ["English", "Other"])

user_input = st.text_area("✍️ Enter your text:", height=150)

# Buttons
col1, col2 = st.columns(2)
with col1:
    analyze = st.button("🔍 Analyze")
with col2:
    clear = st.button("🗑️ Clear History")

# Analyze
if analyze:
    if user_input.strip() != "":
        text = user_input

        # (Simple handling for non-English)
        if language != "English":
            text = user_input  # can integrate translator later

        result = classifier(text)
        label = result[0]['label']
        score = result[0]['score']

        # Save history
        st.session_state.history.append((user_input, label, score))

        st.write("### 📊 Result")

        if label == "POSITIVE":
            st.success(f"😊 Positive (Confidence: {score:.2f})")
        else:
            st.error(f"😡 Negative (Confidence: {score:.2f})")

        # 🎯 Confidence gauge (progress)
        st.write("### 🎯 Confidence Meter")
        st.progress(int(score * 100))

    else:
        st.warning("⚠️ Enter some text")

# Clear history
if clear:
    st.session_state.history = []
    st.success("History cleared!")

# 📊 Charts (Sentiment stats)
st.write("---")
st.write("### 📊 Sentiment Statistics")

if st.session_state.history:
    pos = sum(1 for x in st.session_state.history if x[1] == "POSITIVE")
    neg = sum(1 for x in st.session_state.history if x[1] == "NEGATIVE")

    fig, ax = plt.subplots()
    ax.pie([pos, neg], labels=["Positive", "Negative"], autopct='%1.1f%%')
    st.pyplot(fig)
else:
    st.info("No data for chart yet")

# History
st.write("---")
st.write("### 🕒 History")

for text, lab, sc in st.session_state.history[-5:][::-1]:
    st.write(f"{text} → {lab} ({sc:.2f})")

# Footer
st.write("---")
st.caption("Built using BERT + Streamlit 🚀")
