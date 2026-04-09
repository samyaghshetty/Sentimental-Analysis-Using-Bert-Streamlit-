import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")

# Load model
classifier = pipeline("sentiment-analysis")

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

# Title
st.markdown("<h1 style='text-align: center;'>💬 Sentiment Analysis App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Analyze text sentiment using BERT 🤖</p>", unsafe_allow_html=True)

st.write("---")

# Input box
user_input = st.text_area("✍️ Enter your text below:", height=150)

# Analyze button
if st.button("🔍 Analyze Sentiment"):
    
    if user_input.strip() != "":
        result = classifier(user_input)
        
        label = result[0]['label']
        score = result[0]['score']
        
        # Save history
        st.session_state.history.append((user_input, label, score))
        
        # Show result
        st.write("### 📊 Result:")
        
        if label == "POSITIVE":
            st.success(f"😊 Positive Sentiment\n\nConfidence Score: {score:.2f}")
        else:
            st.error(f"😡 Negative Sentiment\n\nConfidence Score: {score:.2f}")
        
        # 🎯 Improved Confidence Meter
        st.write("### 🎯 Confidence Meter")
        st.progress(int(score * 100))
        st.write(f"Confidence: {score*100:.1f}%")

    else:
        st.warning("⚠️ Please enter some text!")

# Divider
st.write("---")

# 📊 Sentiment Statistics Chart
st.write("### 📊 Sentiment Statistics")

if st.session_state.history:
    pos = sum(1 for x in st.session_state.history if x[1] == "POSITIVE")
    neg = sum(1 for x in st.session_state.history if x[1] == "NEGATIVE")

    fig, ax = plt.subplots()
    ax.pie(
        [pos, neg],
        labels=["Positive 😊", "Negative 😡"],
        autopct='%1.1f%%'
    )
    st.pyplot(fig)
else:
    st.info("No data for chart yet")

# Divider
st.write("---")

# History section
st.write("### 🕒 History (Last 5 Results)")

# Clear history button
if st.button("🗑️ Clear History"):
    st.session_state.history = []
    st.success("History cleared!")

# Show history
if st.session_state.history:
    for text, lab, sc in st.session_state.history[-5:][::-1]:
        st.write(f"➡️ **{text}**")
        st.write(f"Sentiment: {lab} | Confidence: {sc:.2f}")
        st.write("---")
else:
    st.write("No history yet.")

# Footer
st.markdown("<p style='text-align: center;'>Built using BERT & Streamlit 🚀</p>", unsafe_allow_html=True)
