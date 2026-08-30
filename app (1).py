import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="AI Factory Command Center", layout="wide")
st.title("🏭 AI Factory Intelligence Command Center")
st.markdown("### Food Packaging Factory — Autonomous Monitoring System")

model = joblib.load("best_model.pkl")

st.sidebar.header("Input Machine Data")
machine = st.sidebar.selectbox("Select Machine", ["Machine_A", "Machine_B", "Machine_C"])
temperature = st.sidebar.slider("Temperature (°C)", 150, 250, 190)
vibration = st.sidebar.slider("Vibration (mm/s)", 0.0, 5.0, 1.0)
units_produced = st.sidebar.number_input("Units Produced", 500, 2000, 1000)
downtime = st.sidebar.number_input("Downtime (minutes)", 0, 100, 0)

if st.sidebar.button("Run Analysis"):

    st.subheader("1️⃣ Prediction & Confidence")
    machine_map = {"Machine_A": 0, "Machine_B": 1, "Machine_C": 2}
    features_input = [[machine_map[machine], 0, units_produced, downtime, temperature, vibration, 0]]
    prediction = model.predict(features_input)[0]
    st.metric("Predicted Defect Rate", f"{prediction:.3f}")

    st.subheader("2️⃣ Explainability")
    if temperature > 210:
        st.warning(f"High temperature detected ({temperature}°C) — main contributing factor.")
    elif vibration > 2.0:
        st.warning(f"High vibration detected ({vibration} mm/s) — possible mechanical wear.")
    else:
        st.success("All readings within normal operating range.")

    st.subheader("3️⃣ Knowledge Base (RAG) Evidence")
    st.info("SOP Reference: Normal sealing temperature 180-200°C. Above 210°C — stop machine immediately.")

    st.subheader("4️⃣ AI Recommendation")
    if prediction > 0.05 or temperature > 210 or vibration > 2.0:
        recommendation = "⚠️ STOP machine and perform inspection immediately"
    else:
        recommendation = "✅ Continue normal operation, monitor closely"
    st.write(f"**Recommendation:** {recommendation}")

    st.subheader("5️⃣ Human Supervisor Decision")
    decision = st.radio("Your Decision:", ["APPROVE", "REJECT", "MODIFY"])
    reason = st.text_area("Reason / Feedback (optional)")

    if st.button("Submit Decision"):
        log = pd.DataFrame([{
            "machine": machine, "prediction": prediction,
            "recommendation": recommendation, "human_decision": decision, "reason": reason
        }])
        log.to_csv("decision_log.csv", mode='a', header=False, index=False)
        st.success("Decision recorded successfully!")
