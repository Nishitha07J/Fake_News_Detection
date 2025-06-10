import streamlit as st
from utils import get_embedding, classifier

st.set_page_config(page_title="Fake News Detector", page_icon="🧠")

st.title("🧠 Fake News Detector (BERT Model)")
st.markdown("Enter a news article below and we’ll tell you if it's **real or fake** — along with a confidence score.")

# Text input
user_input = st.text_area("Paste your news article here", height=200)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            emb = get_embedding(user_input)
            prediction = classifier.predict(emb)[0]
            confidence = classifier.predict_proba(emb)[0][prediction]

            st.markdown(f"### 🧾 Prediction: **{'REAL ✅' if prediction == 1 else 'FAKE ❌'}**")
            st.markdown(f"#### 🔢 Confidence: {confidence*100:.2f}%")

