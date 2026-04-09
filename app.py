import streamlit as st
from transformers import pipeline

# Page config
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")

# Load model
classifier = pipeline("sentiment-analysis")

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

# Title
st.markdown("<h1 style='text-align: center;'>💬 Sentiment Analysis Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Analyze text sentiment using BERT 🤖</p>", unsafe_allow_html=True)

st.write("---")

# 🧠 DASHBOARD CARDS
total = len(st.session_state.history)
pos = sum(1 for x in st.session_state.history if x[1] == "POSITIVE")
neg = sum(1 for x in st.session_state.history if x[1] == "NEGATIVE")
avg_conf = sum(x[2] for x in st.session_state.history)/total if total > 0 else 0

col1, col2, col3, col4 = st.columns(4)

col1.metric("📝 Total", total)
col2.metric("😊 Positive", pos)
col3.metric("😡 Negative", neg)
col4.metric("📊 Avg Confidence", f"{avg_conf:.2f}")

st.write("---")

# Input
user_input = st.text_area("✍️ Enter your text below:", height=150)

# Analyze
if st.button("🔍 Analyze Sentiment"):
    
    if user_input.strip() != "":
        result = classifier(user_input)
        
        label = result[0]['label']
        score = result[0]['score']
        
        st.session_state.history.append((user_input, label, score))
        
        st.write("### 📊 Result")

        if label == "POSITIVE":
            st.success(f"😊 Positive (Confidence: {score:.2f})")
        else:
            st.error(f"😡 Negative (Confidence: {score:.2f})")

        st.write("### 🎯 Confidence Meter")
        st.progress(int(score * 100))
        st.write(f"{score*100:.1f}%")

    else:
        st.warning("⚠️ Enter some text")

st.write("---")

# History
st.write("### 🕒 Recent Activity")

if st.session_state.history:
    for text, lab, sc in st.session_state.history[-5:][::-1]:
        st.write(f"➡️ {text}")
        st.write(f"{lab} | {sc:.2f}")
        st.write("---")
else:
    st.info("No history yet")

# Clear history
if st.button("🗑️ Clear History"):
    st.session_state.history = []
    st.success("History cleared!")

# Footer
st.markdown("<p style='text-align:center;'>Built using BERT & Streamlit 🚀</p>", unsafe_allow_html=True)
