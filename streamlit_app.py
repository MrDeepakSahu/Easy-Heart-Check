import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pyngrok import ngrok
import os

# --- Configuration & Constants ---
PAGE_TITLE = "Easy Heart Check"
PAGE_ICON = "❤️"
THEME_COLOR = "#FF4B4B"

# Questions Configuration
@dataclass
class Question:
    key: str
    text: str
    type: str  # 'number', 'choice', 'float'
    options: Optional[List[str]] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    icon: str = ""

QUESTIONS = [
    Question("age", "How old are you?", "number", min_val=1, max_val=100, icon="🎂"),
    Question("sex", "Are you Male or Female?", "choice", options=["Male", "Female"], icon="⚧️"),
    Question("chest_pain_type", "Do you have chest pain?", "choice", 
             options=["No Pain", "A little pain", "Bad pain", "Very bad pain"], icon="💔"),
    Question("resting_bp", "What is your Blood Pressure?", "number", min_val=80, max_val=200, icon="🩸"),
    Question("cholesterol", "Cholesterol Level (if you know)", "number", min_val=100, max_val=400, icon="🍔"),
    Question("fasting_bs", "Do you have High Blood Sugar?", "choice", options=["Yes", "No"], icon="🍬"),
    Question("resting_ecg", "ECG / Heart Scan Result", "choice", 
             options=["Normal", "A bit weird", "Problem found"], icon="📈"),
    Question("max_hr", "Max Heart Rate (Running)", "number", min_val=60, max_val=220, icon="🏃"),
    Question("exercise_angina", "Does exercise hurt your chest?", "choice", options=["Yes", "No"], icon="😫"),
    Question("oldpeak", "Tiredness after exercise (0-10)", "float", min_val=0.0, max_val=6.0, icon="📉"),
    Question("st_slope", "Recovery speed after exercise", "choice", 
             options=["Good (Upsloping)", "Okay (Flat)", "Bad (Downsloping)"], icon="📉"),
]

# --- Model Logic ---
class PredictionModel:
    # Machine Learning logic (data synth + model train + predict).
    # The model is trained once and cached to keep the app fast.
    
    @staticmethod
    @st.cache_resource(show_spinner="Preparing AI Doctor...")
    def get_trained_model() -> RandomForestClassifier:
        # Train and return the Random Forest model on synthetic data.
        # Synthetic Data Generation mimicking Heart Disease dataset
        np.random.seed(42)
        n_samples = 2000

        data = {
            'Age': np.random.randint(25, 80, n_samples),
            'Sex': np.random.randint(0, 2, n_samples),
            'ChestPainType': np.random.randint(0, 4, n_samples),
            'RestingBP': np.random.randint(90, 200, n_samples),
            'Cholesterol': np.random.randint(100, 400, n_samples),
            'FastingBS': np.random.randint(0, 2, n_samples),
            'RestingECG': np.random.randint(0, 3, n_samples),
            'MaxHR': np.random.randint(60, 220, n_samples),
            'ExerciseAngina': np.random.randint(0, 2, n_samples),
            'Oldpeak': np.random.uniform(0, 6, n_samples),
            'ST_Slope': np.random.randint(0, 3, n_samples)
        }

        df = pd.DataFrame(data)
        
        # Target Generation Logic (Correlated)
        risk_score = (
            (df['Age'] > 50).astype(int) * 1.5 + 
            (df['RestingBP'] > 140).astype(int) + 
            (df['Cholesterol'] > 240).astype(int) +
            df['ExerciseAngina'] * 2 +
            (df['MaxHR'] < 140).astype(int) +
            (df['ChestPainType'] > 0).astype(int) +
            np.random.normal(0, 0.5, n_samples)
        )
        df['HeartDisease'] = (risk_score >= 3.5).astype(int)

        X = df.drop('HeartDisease', axis=1)
        y = df['HeartDisease']

        model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        model.fit(X, y)
        return model

    @staticmethod
    def predict_risk(data: Dict[str, Any]) -> Dict[str, Any]:
        # Predict heart disease risk based on the user's answers.
        model = PredictionModel.get_trained_model()
        
        # Map simple choices back to numbers for the model
        # Note: This mapping must match the training data logic roughly or be consistent
        
        # Chest Pain: No Pain(0), Little(1), Bad(2), Very Bad(3) - Approximate mapping
        cp_map = {"No Pain": 0, "A little pain": 1, "Bad pain": 2, "Very bad pain": 3}
        cp_val = cp_map.get(data.get('chest_pain_type'), 0)
        if isinstance(data.get('chest_pain_type'), int): cp_val = data.get('chest_pain_type')

        # ECG
        ecg_map = {"Normal": 0, "A bit weird": 1, "Problem found": 2}
        ecg_val = ecg_map.get(data.get('resting_ecg'), 0)
        
        # ST Slope
        slope_map = {"Good (Upsloping)": 0, "Okay (Flat)": 1, "Bad (Downsloping)": 2}
        slope_val = slope_map.get(data.get('st_slope'), 1)

        # Prepare input vector
        input_vector = np.array([[
            data['age'], 
            1 if data['sex'] == 'Male' else 0, 
            cp_val, 
            data['resting_bp'],
            data['cholesterol'], 
            1 if data['fasting_bs'] == 'Yes' else 0, 
            ecg_val,
            data['max_hr'], 
            1 if data['exercise_angina'] == 'Yes' else 0, 
            data['oldpeak'], 
            slope_val
        ]])
        
        prediction = model.predict(input_vector)[0]
        probability = model.predict_proba(input_vector)[0][1]
        
        return {
            "is_high_risk": bool(prediction == 1),
            "probability": probability,
            "risk_level": "High" if prediction == 1 else "Low"
        }

class MedicalAnalysis:
    # Provides medical reasoning and educational medication info from inputs.
    @staticmethod
    def analyze(data: Dict[str, Any]) -> Dict[str, Any]:
        # Build a human-readable list of reasons and medication suggestions.
        reasons = []
        meds = set()
        
        # 1. Blood Pressure Analysis
        bp = data.get('resting_bp', 120)
        if bp > 140:
            reasons.append(f"⚠️ **High Blood Pressure ({bp})**: This puts strain on your heart.")
            meds.add("ACE Inhibitors (e.g., Lisinopril)")
            meds.add("Beta-Blockers (e.g., Metoprolol)")
        elif bp > 120:
            reasons.append(f"⚠️ **Elevated Blood Pressure ({bp})**: Watch your salt intake.")
            
        # 2. Cholesterol Analysis
        chol = data.get('cholesterol', 200)
        if chol > 240:
            reasons.append(f"⚠️ **High Cholesterol ({chol})**: Can clog arteries.")
            meds.add("Statins (e.g., Atorvastatin)")
        
        # 3. Blood Sugar
        if data.get('fasting_bs') == 'Yes':
            reasons.append("⚠️ **High Blood Sugar**: Diabetes is a major risk factor.")
            meds.add("Metformin (if diabetic)")
            
        # 4. Chest Pain & Angina
        cp = data.get('chest_pain_type')
        angina = data.get('exercise_angina')
        if cp in ["Bad pain", "Very bad pain"] or angina == "Yes":
            reasons.append("⚠️ **Chest Pain/Angina**: Indicates poor blood flow.")
            meds.add("Nitroglycerin (for sudden pain)")
            meds.add("Aspirin (blood thinner)")
            
        # 5. Age Factor
        age = data.get('age', 30)
        if age > 60:
            reasons.append(f"ℹ️ **Age ({age})**: Risk increases with age.")

        # Default advice if no specific reasons
        if not reasons:
            reasons.append("✅ No obvious specific risk factors found in input.")

        return {
            "reasons": reasons,
            "medications": list(meds) if meds else ["Lifestyle changes are the best medicine for now."]
        }

    @staticmethod
    def create_report_text(data: Dict[str, Any], result: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        # Generate a plain-text report with inputs, results, and analysis.
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
==================================================
       HEART HEALTH ASSESSMENT REPORT
==================================================
Date: {date_str}
ID: {hash(str(data)) % 10000:04d}

--------------------------------------------------
1. PATIENT INPUTS
--------------------------------------------------
"""
        for k, v in data.items():
            report += f"- {k.replace('_', ' ').title()}: {v}\n"

        report += f"""
--------------------------------------------------
2. ASSESSMENT RESULTS
--------------------------------------------------
Risk Level:   {'HIGH' if result['is_high_risk'] else 'LOW'}
Probability:  {result['probability']:.1%}

--------------------------------------------------
3. DETAILED ANALYSIS
--------------------------------------------------
"""
        if analysis['reasons']:
            for reason in analysis['reasons']:
                # Clean emoji for text file
                clean_reason = reason.replace('⚠️', '').replace('ℹ️', '').replace('✅', '').replace('**', '').strip()
                report += f"[!] {clean_reason}\n"
        else:
            report += "No specific risk factors identified.\n"

        report += f"""
--------------------------------------------------
4. EDUCATIONAL MEDICATION INFO
--------------------------------------------------
(For information only - NOT A PRESCRIPTION)
"""
        for med in analysis['medications']:
            report += f"- {med}\n"

        report += """
==================================================
DISCLAIMER:
This report is generated by an AI prototype.
It is NOT a medical diagnosis.
Consult a qualified doctor for professional advice.
==================================================
"""
        return report

# --- Application Logic ---
class HeartHealthApp:
    # Streamlit UI controller: manages state, flow, and rendering.
    def __init__(self):
        st.set_page_config(
            page_title=PAGE_TITLE,
            page_icon=PAGE_ICON,
            layout="centered", # Better for mobile/simple view
            initial_sidebar_state="collapsed"
        )
        self._inject_custom_css()
        self._init_session_state()
        self._init_public_url()

    def _inject_custom_css(self):
        # Add custom CSS for a friendly, clean UI look.
        st.markdown(f"""
        <style>
            .stApp {{
                background-color: #ffffff;
            }}
            .main-header {{
                font-size: 2rem;
                font-weight: 800;
                color: {THEME_COLOR};
                text-align: center;
                margin-bottom: 0.5rem;
                font-family: 'Arial', sans-serif;
            }}
            .sub-header {{
                font-size: 1.2rem;
                color: #555;
                text-align: center;
                margin-bottom: 2rem;
            }}
            .chat-bubble {{
                padding: 15px;
                border-radius: 20px;
                margin-bottom: 10px;
                max-width: 80%;
                font-size: 1.1rem;
            }}
            .bot-bubble {{
                background-color: #f0f2f6;
                color: #31333F;
                border-bottom-left-radius: 5px;
            }}
            .user-bubble {{
                background-color: {THEME_COLOR};
                color: white;
                border-bottom-right-radius: 5px;
                margin-left: auto;
            }}
            .big-button {{
                width: 100%;
                padding: 15px;
                font-size: 1.2rem;
                font-weight: bold;
                border-radius: 12px;
                margin-bottom: 10px;
                cursor: pointer;
            }}
            /* Hide default streamlit elements to simplify */
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}
        </style>
        """, unsafe_allow_html=True)

    def _init_session_state(self):
        # Initialize conversation, state machine, and data store.
        if "history" not in st.session_state:
            st.session_state.history = []
        if "state" not in st.session_state:
            st.session_state.state = "HOME" # HOME, ASKING, RESULT
        if "q_index" not in st.session_state:
            st.session_state.q_index = 0
        if "data" not in st.session_state:
            st.session_state.data = {}
        if "public_url" not in st.session_state:
            st.session_state.public_url = ""

    def _init_public_url(self):
        # Start public tunnel and store URL if ngrok is available.
        if not st.session_state.public_url:
            st.session_state.public_url = _start_public_tunnel(8504)

    def _reset(self):
        # Reset the app to the home screen and clear conversation/data.
        st.session_state.state = "HOME"
        st.session_state.history = []
        st.session_state.q_index = 0
        st.session_state.data = {}

    def _add_chat(self, role, text):
        # Append a message to the chat history (bot/user bubbles).
        st.session_state.history.append({"role": role, "text": text})

    def render_home(self):
        # Home screen with intro and Start button; shows Public URL if present.
        st.markdown(f'<div class="main-header">{PAGE_ICON} Easy Heart Check</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Check your heart health in 1 minute. No typing needed!</div>', unsafe_allow_html=True)
        if st.session_state.public_url:
            st.info(f"Public URL: {st.session_state.public_url}")
        
        st.image("https://img.freepik.com/free-vector/heart-health-concept-illustration_114360-1206.jpg?w=740", width="stretch")
        
        st.markdown("### 👋 Hello!")
        st.write("I am your AI Doctor helper. I will ask you simple questions to check your heart.")
        
        if st.button("🚀 Start Checkup", type="primary", width="stretch"):
            st.session_state.state = "ASKING"
            st.session_state.q_index = 0
            self._add_chat("bot", "Let's start! " + QUESTIONS[0].text)
            st.rerun()

    def render_chat(self):
        # Main interview UI: shows chat history and asks the current question.
        # Display Chat History
        for msg in st.session_state.history:
            cls = "bot-bubble" if msg["role"] == "bot" else "user-bubble"
            align = "left" if msg["role"] == "bot" else "right"
            st.markdown(f"""
            <div style="display: flex; justify-content: {align};">
                <div class="chat-bubble {cls}">
                    {msg['text']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Current Question Interaction
        if st.session_state.state == "ASKING" and st.session_state.q_index < len(QUESTIONS):
            q = QUESTIONS[st.session_state.q_index]
            
            st.markdown("---")
            st.markdown(f"### {q.icon} {q.text}")
            
            # Input Widgets based on type
            val = None
            confirmed = False
            
            if q.type == "choice":
                cols = st.columns(len(q.options))
                for i, opt in enumerate(q.options):
                    if cols[i].button(opt, width="stretch", key=f"btn_{q.key}_{i}"):
                        val = opt
                        confirmed = True
            
            elif q.type == "number" or q.type == "float":
                if q.type == "number":
                    step = 1
                    min_v = int(q.min_val)
                    max_v = int(q.max_val)
                else:
                    step = 0.1
                    min_v = float(q.min_val)
                    max_v = float(q.max_val)
                
                val = st.slider("Slide to select:", min_value=min_v, max_value=max_v, step=step, key=f"sl_{q.key}")
                if st.button("✅ Confirm", type="primary", width="stretch"):
                    confirmed = True

            # Handle Confirmation
            if confirmed:
                st.session_state.data[q.key] = val
                self._add_chat("user", str(val))
                
                st.session_state.q_index += 1
                if st.session_state.q_index < len(QUESTIONS):
                    next_q = QUESTIONS[st.session_state.q_index]
                    self._add_chat("bot", next_q.text)
                else:
                    st.session_state.state = "RESULT"
                st.rerun()

        elif st.session_state.state == "RESULT":
            self.render_result()

    def render_result(self):
        # Result screen: shows risk, reasons, medications, report download, and caution.
        with st.spinner("Checking your heart..."):
            time.sleep(1.5)
            data = st.session_state.data
            result = PredictionModel.predict_risk(data)
            analysis = MedicalAnalysis.analyze(data)
        
        st.markdown("---")
        
        # 1. Main Result
        if result['is_high_risk']:
            st.error("## ⚠️ High Risk Detected")
            st.markdown(f"**Possibility of Heart Disease: {result['probability']:.0%}**")
            st.write("Your answers suggest you might be at higher risk.")
        else:
            st.success("## ✅ Heart Looks Good!")
            st.markdown(f"**Possibility of Heart Disease: {result['probability']:.0%}**")
            st.write("Your risk appears low based on these answers.")

        # 2. Reasons (Why?)
        st.markdown("### 🔍 Why?")
        for reason in analysis['reasons']:
            st.write(f"- {reason}")

        # 3. Medications (Educational)
        st.markdown("### 💊 Common Medicines for These Conditions")
        st.info("**Note:** This is educational only. Never take medicine without a doctor's prescription.")
        for med in analysis['medications']:
            st.write(f"- {med}")

        # 4. General Advice
        st.markdown("### 🍎 General Advice")
        st.write("- Eat healthy food (Less salt/fat)")
        st.write("- Do not smoke")
        st.write("- Exercise daily (Walk for 30 mins)")
        
        # 5. Reset Button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Check Someone Else", width="stretch"):
                self._reset()
                st.rerun()
        with col2:
            report_text = MedicalAnalysis.create_report_text(data, result, analysis)
            st.download_button(
                label="📄 Download Report",
                data=report_text,
                file_name=f"heart_report_{int(time.time())}.txt",
                mime="text/plain",
                width="stretch"
            )

        # 6. Caution / Disclaimer Footer
        st.markdown("---")
        st.warning("""
        **🚨 CAUTION & DISCLAIMER:**
        This is an AI prototype and **NOT a real doctor**. 
        - The results could be wrong.
        - Do not stop or start taking medicines based on this app.
        - **If you have chest pain, dizziness, or trouble breathing, go to the hospital IMMEDIATELY.**
        """)

    def main(self):
        # Entry point: routes to Home or Chat depending on current state.
        if st.session_state.state == "HOME":
            self.render_home()
        else:
            # Header
            st.markdown(f'<div class="main-header" style="font-size:1.5rem;">{PAGE_ICON} Easy Heart Check</div>', unsafe_allow_html=True)
            if st.button("🏠 Home", key="home_btn"):
                self._reset()
                st.rerun()
            self.render_chat()
@st.cache_resource
def _start_public_tunnel(port: int = 8504) -> str:
    # Create an ngrok tunnel if NGROK_AUTHTOKEN is set; otherwise return blank.
    try:
        token = os.environ.get("NGROK_AUTHTOKEN")
        if not token:
            return ""
        ngrok.set_auth_token(token)
        configured_port = st.get_option("server.port") or port
        t = ngrok.connect(configured_port)
        return t.public_url
    except Exception:
        return ""


if __name__ == "__main__":
    app = HeartHealthApp()
    app.main()

