import streamlit as st
import joblib
import numpy as np
import csv
from math import exp

st.set_page_config(page_title="Lower Limb Pain CDSS")
st.title("Clinical Decision Support Tool – Lower Limb MSK Pain")

# --- DIAGNOSTIC MODE TOGGLE ---
mode = st.radio("Select Diagnosis Mode:", ["Manual", "AI"], horizontal=True)

st.markdown("---")
st.subheader("Step 1: Patient Symptom Input")

# Symptom input checkboxes
symptoms_selected = {
    "pain_at_rest": st.checkbox("Pain at rest"),
    "night_pain": st.checkbox("Night pain"),
    "radiating_pain": st.checkbox("Pain radiates below the knee"),
    "numbness": st.checkbox("Numbness or tingling"),
    "weakness": st.checkbox("Motor weakness (e.g. dorsiflexion)"),
    "bladder": st.checkbox("Bladder or bowel dysfunction"),
    "calf_swelling": st.checkbox("Unilateral calf swelling"),
    "exertional_pain": st.checkbox("Pain increases with exertion and resolves with rest"),
    "posterior_leg_pain": st.checkbox("Posterior leg pain"),
    "medial_leg_pain": st.checkbox("Medial tibial pain"),
    "lateral_leg_pain": st.checkbox("Lateral fibular pain"),
    "sudden_onset": st.checkbox("Sudden onset of pain"),
    "swelling": st.checkbox("Swelling in the calf or shin"),
    "tenderness": st.checkbox("Localised tenderness on palpation"),
    "morning_stiffness": st.checkbox("Morning stiffness"),
    "constant_pain": st.checkbox("Constant, non-mechanical pain"),
    "pain_with_resisted_pf": st.checkbox("Pain with resisted plantarflexion"),
    "pain_with_dorsiflex_straight_knee": st.checkbox("Pain on dorsiflexion with knee straight"),
    "pain_with_dorsiflex_bent_knee": st.checkbox("Pain on dorsiflexion with knee bent"),
    "palpation_medial_tibia": st.checkbox("Tenderness on medial tibia palpation"),
    "palpation_fibula": st.checkbox("Tenderness on lateral fibula palpation")
}

st.markdown("---")
st.subheader("Step 2: Suggested Clinical Action")

# Red flag override logic
if symptoms_selected["bladder"]:
    st.error("⚠️ Red Flag: Possible Cauda Equina – Urgent Referral Required")
elif symptoms_selected["calf_swelling"] and symptoms_selected["constant_pain"]:
    st.error("⚠️ Red Flag: Possible DVT – Immediate Medical Assessment Required")
elif symptoms_selected["night_pain"] and symptoms_selected["constant_pain"]:
    st.error("⚠️ Red Flag: Suspicious non-mechanical pain pattern – Investigate malignancy or infection")
elif symptoms_selected["morning_stiffness"]:
    st.warning("⚠️ Inflammatory indicator – Consider rheumatologic assessment")
else:
    scores = {}

    if mode == "Manual":
        weights = {
            "Lumbar Radiculopathy": {"radiating_pain": 2.5, "numbness": 2.0, "weakness": 2.2, "night_pain": 1.2, "pain_at_rest": 1.0},
            "Deep Vein Thrombosis (DVT)": {"calf_swelling": 3.0, "posterior_leg_pain": 1.8, "swelling": 2.5, "constant_pain": 2.7},
            "Medial Tibial Stress Syndrome (MTSS)": {"medial_leg_pain": 2.6, "exertional_pain": 2.4, "palpation_medial_tibia": 2.2},
            "Tibial Stress Fractures": {"medial_leg_pain": 2.3, "night_pain": 1.8, "pain_at_rest": 2.0, "tenderness": 2.5, "palpation_medial_tibia": 2.6},
            "Fibular Stress Fractures": {"lateral_leg_pain": 2.3, "night_pain": 1.5, "pain_at_rest": 1.6, "tenderness": 2.4, "palpation_fibula": 2.7},
            "Gastrocnemius Muscle Tears": {"posterior_leg_pain": 2.6, "sudden_onset": 2.8, "swelling": 2.0, "pain_with_resisted_pf": 2.7},
            "Soleus Muscle Tears": {"posterior_leg_pain": 2.1, "exertional_pain": 2.4, "pain_with_dorsiflex_bent_knee": 2.8},
            "Chronic Exertional Compartment Syndrome": {"exertional_pain": 2.5, "swelling": 1.8, "numbness": 1.6, "pain_with_dorsiflex_straight_knee": 2.0},
            "Myopathies": {"weakness": 2.9, "night_pain": 1.1, "pain_at_rest": 1.4},
            "Peroneal Nerve Entrapment": {"radiating_pain": 1.6, "numbness": 2.7, "weakness": 2.5}
        }

        for condition, feature_weights in weights.items():
            score = 0
            for symptom, present in symptoms_selected.items():
                weight = feature_weights.get(symptom, 0)
                score += weight * int(present)
            probability = 1 / (1 + exp(-score))
            scores[condition] = probability

    else:
        try:
            model = joblib.load("model.pkl")
            encoder = joblib.load("label_encoder.pkl")
            symptom_order = [
                "pain_at_rest", "night_pain", "radiating_pain", "numbness", "weakness", "bladder",
                "calf_swelling", "exertional_pain", "posterior_leg_pain", "medial_leg_pain", "lateral_leg_pain",
                "sudden_onset", "swelling", "tenderness", "morning_stiffness", "constant_pain",
                "pain_with_resisted_pf", "pain_with_dorsiflex_straight_knee", "pain_with_dorsiflex_bent_knee",
                "palpation_medial_tibia", "palpation_fibula"
            ]
            input_vector = np.array([int(symptoms_selected[s]) for s in symptom_order]).reshape(1, -1)
            probs = model.predict_proba(input_vector)[0]
            for i, class_index in enumerate(model.classes_):
                label = encoder.inverse_transform([class_index])[0]
                scores[label] = probs[i]
        except Exception as e:
            st.error(f"⚠️ AI Mode failed: {e}")

    total = sum(scores.values())
    if total > 0:
        for condition in scores:
            scores[condition] /= total

        # Ensure all conditions are present in scores even if 0
    for condition in [
        "Lumbar Radiculopathy",
        "Deep Vein Thrombosis (DVT)",
        "Medial Tibial Stress Syndrome (MTSS)",
        "Tibial Stress Fractures",
        "Fibular Stress Fractures",
        "Gastrocnemius Muscle Tears",
        "Soleus Muscle Tears",
        "Chronic Exertional Compartment Syndrome",
        "Myopathies",
        "Peroneal Nerve Entrapment"
    ]:
        if condition not in scores:
            scores[condition] = 0.0

    top_diagnoses = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def display_group(title, group_list):
        st.markdown(f"### {title}")
        for diagnosis in group_list:
            confidence = scores.get(diagnosis, 0)
            st.write(f"{diagnosis} – {int(confidence * 100)}% confidence")
            st.progress(confidence)
            with st.expander(f"More about {diagnosis}"):
                st.markdown(f"**{diagnosis}**\n- Diagnostic summary coming soon.")  # fixed unterminated f-string

    if any(symptoms_selected.values()):
        st.subheader("All Likely Diagnoses (Grouped by Clinical Urgency)")
        display_group("🟥 Urgent or Red Flag Diagnoses", ["Deep Vein Thrombosis (DVT)", "Lumbar Radiculopathy"])
        display_group("🟨 Moderate – Imaging or Specialist May Be Needed", ["Medial Tibial Stress Syndrome (MTSS)", "Tibial Stress Fractures", "Fibular Stress Fractures", "Chronic Exertional Compartment Syndrome"])
        display_group("🟩 Routine – Consider Conservative Management First", ["Gastrocnemius Muscle Tears", "Soleus Muscle Tears", "Peroneal Nerve Entrapment", "Myopathies"])

        st.markdown("---")
        st.markdown("<h4 style='color:#1f77b4;'>✅ Confirm Final Diagnosis</h4>", unsafe_allow_html=True)
        st.success("Help us improve the model. Select the correct diagnosis below:")
        diagnosis_options = sorted([
            "Lumbar Radiculopathy",
            "Deep Vein Thrombosis (DVT)",
            "Medial Tibial Stress Syndrome (MTSS)",
            "Tibial Stress Fractures",
            "Fibular Stress Fractures",
            "Gastrocnemius Muscle Tears",
            "Soleus Muscle Tears",
            "Chronic Exertional Compartment Syndrome",
            "Myopathies",
            "Peroneal Nerve Entrapment"
        ])
        confirmed = st.selectbox("🔍 Confirmed Diagnosis:", [""] + diagnosis_options)
        if st.button("✅ Submit Case to Training Log"):
            case_row = [int(symptoms_selected[s]) for s in symptoms_selected] + [confirmed]
            with open("confirmed_cases.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(case_row)
            st.success("Case logged successfully to 'confirmed_cases.csv'. You can use this to retrain the model later.")

retrain_output = None
if st.button("🔁 Retrain Model Now", key="retrain_button"):
    with st.spinner("Retraining the model..."):
        import subprocess
        result = subprocess.run(["python3", "retrain_from_confirmed_safe.py"], capture_output=True, text=True)
        retrain_output = result.stdout if result.returncode == 0 else result.stderr

st.markdown("---")
show_trainer = st.checkbox("👨‍🏫 Show Trainer Panel", value=False, key="trainer_panel_toggle")

if show_trainer:
    if retrain_output:
        st.markdown("### 🧪 Classification Report")
        if "Classification Report:" in retrain_output:
            st.code(retrain_output.split("Classification Report:")[-1].strip())
        else:
            st.code(retrain_output.strip())



if show_trainer and mode == "AI" and any(symptoms_selected.values()) and scores:
    import matplotlib.pyplot as plt
    selected_count = sum(symptoms_selected.values())
    top_confidence = max(scores.values()) if scores else 0
    st.subheader("🧠 Confidence Calibration Graph")
    fig, ax = plt.subplots()
    ax.bar(["Symptoms Selected"], [selected_count], color="lightblue", label="Symptoms")
    ax.bar(["AI Confidence %"], [int(top_confidence * 100)], color="orange", label="Confidence")
    ax.set_ylim(0, max(10, selected_count + 2, 100))
    ax.set_ylabel("Value")
    ax.set_title("Confidence vs Symptom Count")
    ax.legend()
    st.pyplot(fig)

    st.markdown("### 📋 Trainer Summary")
    st.write(f"**Symptoms selected:** {selected_count}")
    st.write(f"**Top AI confidence:** {int(top_confidence * 100)}%")
    st.write(f"**Total Diagnoses Scored:** {len(scores)}")
    st.write("**Top 3 AI Suggestions:**")
    for name, conf in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]:
        st.write(f"- {name}: {int(conf * 100)}% confidence")

st.markdown("---")
st.markdown("### 📚 Diagnostic Scope")
diagnoses = sorted([
    "Chronic Exertional Compartment Syndrome",
    "Deep Vein Thrombosis (DVT)",
    "Fibular Stress Fractures",
    "Gastrocnemius Muscle Tears",
    "Lumbar Radiculopathy",
    "Medial Tibial Stress Syndrome (MTSS)",
    "Myopathies",
    "Peroneal Nerve Entrapment",
    "Soleus Muscle Tears",
    "Tibial Stress Fractures"
])
for d in diagnoses:
    st.markdown(f"- {d}")

