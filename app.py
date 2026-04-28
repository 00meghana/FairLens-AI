import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import google.generativeai as genai

# 🔑 Add your API key here
genai.configure(api_key="YOUR_API_KEY")

st.set_page_config(page_title="FairLens AI", layout="wide")

st.title("FairLens AI")
st.subheader("Bias Detection in Automated Decisions")

file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📊 Data Preview")
    st.dataframe(df)

    # Encode data
    df_encoded = df.copy()
    le = LabelEncoder()

    if 'gender' in df.columns:
        df_encoded['gender'] = le.fit_transform(df_encoded['gender'])

    if 'decision' in df.columns:
        df_encoded['decision'] = df_encoded['decision'].map({'Approved':1, 'Rejected':0})

    features = [col for col in df_encoded.columns if col != 'decision']
    X = df_encoded[features]
    y = df_encoded['decision']

    # Model
    model = LogisticRegression()
    model.fit(X, y)
    preds = model.predict(X)

    df['Prediction'] = preds

    st.subheader("🤖 AI Decision Output")
    st.dataframe(df)

    # Bias Detection
    st.subheader("⚖️ Bias Detection")

    if 'gender' in df.columns:
        group = df.groupby('gender')['Prediction'].mean()

        col1, col2 = st.columns(2)

        with col1:
            st.write("Selection Rate by Group")
            fig, ax = plt.subplots()
            group.plot(kind='bar', ax=ax)
            st.pyplot(fig)

        fairness_gap = abs(group.max() - group.min())

        with col2:
            st.metric("Fairness Gap", round(fairness_gap, 2))

        # 🔥 GEMINI AI EXPLANATION
        st.subheader("🧠 AI Bias Explanation")

        prompt = f"""
        We analyzed a dataset for bias in automated decisions.

        Selection rates by group: {group.to_dict()}
        Fairness gap: {fairness_gap}

        Explain:
        1. Is there bias?
        2. Why might it be happening?
        3. How can it be fixed?

        Keep it simple and clear.
        """

        try:
            model_gemini = genai.GenerativeModel("gemini-pro")
            response = model_gemini.generate_content(prompt)

            st.write(response.text)

        except Exception as e:
            st.error("Gemini API error. Check your API key.")

        # Insights
        st.subheader("📌 System Insight")

        if fairness_gap > 0.2:
            st.error("Potential bias detected")
        else:
            st.success("No major bias detected")

    else:
        st.warning("No sensitive attribute found (e.g., gender)")