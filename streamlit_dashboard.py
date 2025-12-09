"""
Minimal Streamlit UI:

1. Login page (email + password, local only).
2. After login:
   - Show model feature schema from Prediction.py joblibs
   - Let user input key clinical attributes
   - Predict disease_group (with NEURO/RENAL refinement if available)
   - Recommend medicines from disease_group_top_drugs.joblib
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ======================================================
# >>>>>>>>>>>>>>> COLOR / STYLE ADJUSTMENTS <<<<<<<<<<<<
# ======================================================
APP_COLORS = {

    # Layout / theme
    "bg_main": "#f5f7fb",      # main body "ice white"
    "text_main": "#000000",    # main body text: black

    "sidebar_bg": "#34708B",   # LEFT PANEL background
    "sidebar_text": "#ffffff", # LEFT PANEL text / labels

    "header_bg": "#34708B",    # TOP BAR background
    "header_text": "#ffffff",  # TOP BAR title text

    "button_bg": "#4ba1c92b",    # buttons (blue)
    "button_text": "#ffffff",  # button text: white
}

# ======================================================
# STREAMLIT CONFIG
# ======================================================

st.set_page_config(
    page_title="Disease & Medicine Recommendation Model",
    layout="wide",
)

# ======================================================
# >>>>>>>>>>> BACKGROUND / SIDEBAR COLOR SECTION <<<<<<<<
# ======================================================
# This CSS block applies:
# - Main app background & text
# - Sidebar background & text
# - Styling for inputs/buttons/dropdowns/sliders
# - Blue widgets (selects, +/- buttons) with white text

st.markdown(
    f"""
    <style>

    /* Remove some default padding at the top */
    .block-container {{
        padding-top: 0.5rem;
    }}

    /* Main app background & text */
    .stApp {{
        background-color: {APP_COLORS['bg_main']};
        color: {APP_COLORS['text_main']};
    }}

    [data-testid="stAppViewContainer"] {{
        background-color: {APP_COLORS['bg_main']};
        color: {APP_COLORS['text_main']};
    }}

    /* FORCE MAIN-AREA TEXT TO BLACK */
    [data-testid="stAppViewContainer"] * {{
        color: {APP_COLORS['text_main']} !important;
    }}

    /* Sidebar background */
    [data-testid="stSidebar"] {{
        background-color: {APP_COLORS['sidebar_bg']};
    }}

    /* Sidebar text (fallback) */
    [data-testid="stSidebar"] * {{
        color: {APP_COLORS['sidebar_text']} !important;
    }}

    /* Sidebar labels / headings / paragraphs specifically */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stText,
    [data-testid="stSidebar"] p {{
        color: {APP_COLORS['sidebar_text']} !important;
    }}

    /* Sidebar input backgrounds */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] textarea {{
        background-color: #ffffff !important;
        color: #000000 !important;
    }}

    /* Sidebar slider labels / tick labels */
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stSlider div {{
        color: {APP_COLORS['sidebar_text']} !important;
    }}

    /* Buttons (everywhere) */
    .stButton > button {{
        color: {APP_COLORS['sidebar_text']} !important;
        border-radius: 999px;
        border: 1px solid #ffffff55;
        background: rgba(0, 0, 0, 0.08);
    }}
    .stButton > button:hover {{
        background: rgba(255, 255, 255, 0.20);
    }}

    /* Main text color for common widgets */
    .stMarkdown, .stText, .stMetric, .stSelectbox label,
    .stSlider label, .stRadio label {{
        color: {APP_COLORS['text_main']} !important;
    }}

    /* Main area input backgrounds (text / number fields) */
    input, textarea {{
        background-color: #ffffff !important;
        color: #000000 !important;
    }}

    /* ================================
       SELECTBOX / MULTISELECT
       ================================ */

    /* Closed dropdown box (main area) → BLUE with white text */
    div[data-baseweb="select"] > div {{
        background-color: {APP_COLORS['header_bg']} !important;
        border: 1px solid {APP_COLORS['header_bg']} !important;
    }}

    /* Selected value text and placeholder in selectboxes */
    div[data-baseweb="select"] span {{
        color: #ffffff !important;
    }}

    /* Dropdown arrow icon */
    div[data-baseweb="select"] svg {{
        fill: #ffffff !important;
    }}

    /* Dropdown floating container */
    div[data-baseweb="popover"] {{
        background-color: #ffffff !important;
        color: #000000 !important;
    }}

    /* Dropdown list panel */
    ul[role="listbox"] {{
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 6px !important;
    }}

    /* List items */
    ul[role="listbox"] li {{
        background-color: #ffffff !important;
        color: #000000 !important;
    }}

    /* List hover highlight */
    ul[role="listbox"] li:hover {{
        background-color: #e6e6e6 !important;
        color: #000000 !important;
    }}

    /* Selected item highlight */
    ul[role="listbox"] li[aria-selected="true"] {{
        background-color: #d9d9d9 !important;
        color: #000000 !important;
    }}

    /* Sidebar-specific select (override blue text to black inside box) */
    [data-testid="stSidebar"] div[data-baseweb="select"] span {{
        color: #000000 !important;
    }}

    /* ================================
       NUMBER INPUT (+/-) BUTTONS
       ================================ */

    .stNumberInput button,
    button[aria-label="Increment"],
    button[aria-label="Decrement"] {{
        background-color: {APP_COLORS['button_bg']} !important;
        color: {APP_COLORS['button_text']} !important;
    }}
    .stNumberInput button:hover,
    button[aria-label="Increment"]:hover,
    button[aria-label="Decrement"]:hover {{
        background-color: #285a70 !important;
        color: #ffffff !important;
    }}



    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# >>>>>>>>>>>>> CUSTOM HEADER / TOP BAR <<<<<<<<<<<<<<<<
# ======================================================

st.markdown(
    f"""
    <div style="
        margin-top: 2.5rem;      /* gap below Streamlit's built-in bar */
        background-color: {APP_COLORS['header_bg']} !important;
        padding: 0.9rem 1.4rem;
        border-radius: 12px;
        display: flex;
        align-items: center;
        margin-bottom: 1.5rem;
    ">
        <h1 style="
            color: {APP_COLORS['header_text']};
            margin: 0;
            font-size: 1.6rem;
            font-weight: 700;
        ">
            Disease &amp; Medicine Recommendation Model – Feature &amp; Prediction Model
        </h1>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------
# SESSION STATE: LOGIN
# ---------------------------------------------------
if "auth" not in st.session_state:
    st.session_state.auth = {
        "is_logged_in": False,
        "email": None,
    }

# ---------------------------------------------------
# LOAD ML ARTEFACTS FROM Prediction.py
# ---------------------------------------------------

@st.cache_resource(show_spinner=True)
def load_prediction_artifacts():
    """
    Load artefacts created in Prediction.py:

    - final_disease_model.joblib          (top-level multi-class model)
    - disease_label_encoder.joblib        (map encoded → disease_group)
    - neuro_renal_model.joblib            (optional 2nd-stage model)
    - neuro_renal_label_encoder.joblib    (NEURO vs RENAL)
    - final_disease_model_meta.joblib     (categorical_cols, numeric_cols, etc.)
    - disease_group_top_drugs.joblib      (disease_group → list of drugs)
    """
    try:
        top_model = joblib.load("final_disease_model.joblib")
        top_label_encoder = joblib.load("disease_label_encoder.joblib")
        meta = joblib.load("final_disease_model_meta.joblib")
        drug_map = joblib.load("disease_group_top_drugs.joblib")

        # Try to load NEURO/RENAL second stage (may or may not exist)
        try:
            nr_model = joblib.load("neuro_renal_model.joblib")
            nr_label_encoder = joblib.load("neuro_renal_label_encoder.joblib")
        except Exception:
            nr_model = None
            nr_label_encoder = None

        categorical_cols = meta["categorical_cols"]
        numeric_cols = meta["numeric_cols"]
        id_cols = meta["id_cols"]
        target_col = meta["target_col"]
        model_name = meta.get("final_model_name", "Unknown model")

        return {
            "top_model": top_model,
            "top_label_encoder": top_label_encoder,
            "nr_model": nr_model,
            "nr_label_encoder": nr_label_encoder,
            "categorical_cols": categorical_cols,
            "numeric_cols": numeric_cols,
            "id_cols": id_cols,
            "target_col": target_col,
            "model_name": model_name,
            "drug_map": drug_map,
        }
    except FileNotFoundError as e:
        st.error(
            "Could not find one of the required joblib files.\n\n"
            "Make sure this Streamlit script is in the same folder as:\n"
            "- final_disease_model.joblib\n"
            "- disease_label_encoder.joblib\n"
            "- final_disease_model_meta.joblib\n"
            "- disease_group_top_drugs.joblib\n"
            "(and optionally neuro_renal_model.joblib + neuro_renal_label_encoder.joblib)\n\n"
            f"Details: {e}"
        )
        return None
    except Exception as e:
        st.error(f"Error while loading model artefacts: {e}")
        return None

# ---------------------------------------------------
# LOGIN PAGE
# ---------------------------------------------------

def login_page():
    st.subheader("Login")

    with st.container():
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            if not email or not password:
                st.warning("Please enter both email and password.")
            else:
                st.session_state.auth["is_logged_in"] = True
                st.session_state.auth["email"] = email
                st.success(f"Logged in as {email}.")
                st.rerun()


# ---------------------------------------------------
# HELPER: BUILD ONE-ROW DF FROM USER INPUTS
# ---------------------------------------------------

def build_feature_row(artifacts, user_inputs):
    """
    Build a single-row DataFrame with all categorical + numeric cols
    expected by the model. For columns not covered by user inputs,
    we put NaN so the model's SimpleImputer can handle them.
    """
    cat_cols = artifacts["categorical_cols"]
    num_cols = artifacts["numeric_cols"]

    row = {}

    # Categorical columns
    for col in cat_cols:
        lc = col.lower()
        if "gender" in lc:
            row[col] = user_inputs.get("gender")
        else:
            row[col] = np.nan  # imputed with most frequent

    # Numeric columns
    for col in num_cols:
        lc = col.lower()
        val = np.nan  # default → imputed with median in pipeline

        # Demographics
        if "age" in lc:
            val = user_inputs.get("age")

        # Vitals (24h)
        elif "hr" in lc or "heart" in lc:
            val = user_inputs.get("hr")
        elif "sbp" in lc:
            val = user_inputs.get("sbp")
        elif "dbp" in lc:
            val = user_inputs.get("dbp")
        elif "map" in lc:
            val = user_inputs.get("map_bp")
        elif "resp" in lc:
            val = user_inputs.get("resp")
        elif "temp" in lc:
            val = user_inputs.get("temp")
        elif "spo2" in lc or "o2" in lc:
            val = user_inputs.get("spo2")

        # Labs (24h)
        elif "wbc" in lc:
            val = user_inputs.get("wbc")
        elif "hgb" in lc or "hemoglobin" in lc:
            val = user_inputs.get("hgb")
        elif "platelet" in lc or "plt" in lc:
            val = user_inputs.get("platelets")
        elif "creat" in lc:
            val = user_inputs.get("creat")
        elif "bun" in lc:
            val = user_inputs.get("bun")
        elif "lactate" in lc:
            val = user_inputs.get("lactate")
        elif lc.startswith("na") or "sodium" in lc:
            val = user_inputs.get("na")
        elif lc.startswith("k") or "potassium" in lc:
            val = user_inputs.get("k")

        row[col] = val

    row_df = pd.DataFrame([row])
    return row_df[cat_cols + num_cols]

# ---------------------------------------------------
# MAIN PAGE AFTER LOGIN:
# SHOW MODEL INFO + GET USER INPUTS + PREDICT + RECOMMEND
# ---------------------------------------------------

def main_app_page():
    email = st.session_state.auth["email"]
    st.markdown(f"**Logged in as:** {email}")

    artifacts = load_prediction_artifacts()
    if artifacts is None:
        return

    cat_cols = artifacts["categorical_cols"]
    num_cols = artifacts["numeric_cols"]
    id_cols = artifacts["id_cols"]
    target_col = artifacts["target_col"]
    model_name = artifacts["model_name"]
    drug_map = artifacts["drug_map"]

    # ---------------- MODEL SUMMARY CARD ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Model Summary", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Model", model_name)
    with c2:
        st.metric("Categorical Features", len(cat_cols))
    with c3:
        st.metric("Numeric Features", len(num_cols))

    st.write(f"Target column (encoded in training): **{target_col}**")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- FEATURE SCHEMA (OPTIONAL VIEW) ----------------
    with st.expander("View full feature schema (from Prediction.py)", expanded=False):
        c_feat1, c_feat2 = st.columns(2)
        with c_feat1:
            st.markdown("**Categorical features**")
            st.dataframe(pd.DataFrame({"categorical_features": cat_cols}), use_container_width=True)
        with c_feat2:
            st.markdown("**Numeric features**")
            st.dataframe(pd.DataFrame({"numeric_features": num_cols}), use_container_width=True)


    # Demographics
    st.markdown("#### Demographics")
    d1, d2 = st.columns(2)
    with d1:
        age = st.number_input("Age (years)", min_value=18, max_value=120, value=65)
    with d2:
        gender = st.selectbox("Gender", ["M", "F"], index=0)

    # Vitals
    st.markdown("#### First 24h ICU Vitals (approximate)")
    v1, v2, v3 = st.columns(3)
    with v1:
        sbp = st.number_input("Systolic BP (SBP) mmHg", 60, 260, 120)
        dbp = st.number_input("Diastolic BP (DBP) mmHg", 30, 150, 70)
        map_bp = st.number_input("Mean Arterial Pressure (MAP) mmHg", 40, 200, 90)
    with v2:
        hr = st.number_input("Heart rate (HR) bpm", 20, 220, 90)
        resp = st.number_input("Respiratory rate (RESP) /min", 5, 60, 18)
        spo2 = st.number_input("SpO₂ (%)", 50, 100, 96)
    with v3:
        temp = st.number_input("Temperature (°C)", 30.0, 42.0, 37.0, step=0.1)

    # Labs
    st.markdown("#### First 24h Lab Values (approximate)")
    l1, l2, l3, l4 = st.columns(4)
    with l1:
        wbc = st.number_input("WBC (10⁹/L)", 0.0, 100.0, 9.0, step=0.1)
        hgb = st.number_input("Hemoglobin (g/dL)", 0.0, 25.0, 12.0, step=0.1)
    with l2:
        platelets = st.number_input("Platelets (10⁹/L)", 0.0, 1000.0, 250.0, step=1.0)
        bun = st.number_input("BUN (mg/dL)", 0.0, 200.0, 18.0, step=0.5)
    with l3:
        creat = st.number_input("Creatinine (mg/dL)", 0.0, 20.0, 1.0, step=0.1)
        lactate = st.number_input("Lactate (mmol/L)", 0.0, 20.0, 1.5, step=0.1)
    with l4:
        na = st.number_input("Sodium (Na, mEq/L)", 100.0, 180.0, 138.0, step=0.5)
        k = st.number_input("Potassium (K, mEq/L)", 1.0, 10.0, 4.0, step=0.1)

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- PREDICT BUTTON ----------------
    predict_btn = st.button("Predict disease group & recommend medicines")

    if predict_btn:
        # 1) Bundle user inputs
        user_inputs = {
            "age": age,
            "gender": gender,
            "sbp": sbp,
            "dbp": dbp,
            "map_bp": map_bp,
            "hr": hr,
            "resp": resp,
            "spo2": spo2,
            "temp": temp,
            "wbc": wbc,
            "hgb": hgb,
            "platelets": platelets,
            "bun": bun,
            "creat": creat,
            "lactate": lactate,
            "na": na,
            "k": k,
        }

        # 2) Build feature row compatible with model
        row_df = build_feature_row(artifacts, user_inputs)


                # 3) Predict disease group (with NORMAL + RESP override + optional NEURO/RENAL refinement)
        top_model = artifacts["top_model"]
        top_le = artifacts["top_label_encoder"]
        nr_model = artifacts["nr_model"]
        nr_le = artifacts["nr_label_encoder"]

        with st.spinner("Running ML model on your inputs..."):

            # --- 3.1: probabilities from top-level model ---
            proba = top_model.predict_proba(row_df)[0]   # shape: (num_classes,)
            max_prob = float(proba.max())
            best_idx = int(proba.argmax())
            raw_top_group = top_le.inverse_transform([best_idx])[0]

            # --- 3.2: physiological "normal" checks using the UI values ---
            NORMAL_VITAL = (
                60 <= hr   <= 100 and
                100 <= sbp <= 140 and
                65 <= map_bp <= 100 and
                12 <= resp <= 20 and
                36.0 <= temp <= 38.0 and
                spo2 >= 95
            )

            NORMAL_LABS = (
                4   <= wbc       <= 12   and
                hgb >= 10                and
                platelets >= 150         and
                creat <= 1.5             and
                bun   <= 25              and
                135 <= na        <= 145  and
                3.5 <= k         <= 5.2  and
                lactate <= 2.0
            )

            PROB_THRESHOLD = 0.40

            # -------------------------
            # 3.3: NORMAL decision
            # -------------------------
            # NORMAL only if:
            #   - model is NOT confident
            #   - AND vitals + labs look normal
            if (max_prob < PROB_THRESHOLD) and (NORMAL_VITAL and NORMAL_LABS):
                top_group = "NORMAL"
                refined_group = "NORMAL"

            else:
                # Start from raw model prediction
                top_group = raw_top_group
                refined_group = top_group

                # -------------------------
                # 3.4: RESP override rule
                # -------------------------
                classes = list(top_le.classes_)
                if "RESP" in classes:
                    resp_idx = classes.index("RESP")
                    resp_prob = float(proba[resp_idx])

                    RESP_RULE = (
                        resp >= 25 and      # tachypnea
                        spo2 <= 90 and      # hypoxia
                        temp >= 38.0 and    # fever
                        wbc  >= 14.0        # leukocytosis
                    )

                    # If strong respiratory pattern and RESP prob not tiny → force RESP
                    if RESP_RULE and resp_prob >= 0.15:
                        top_group = "RESP"
                        refined_group = "RESP"

                # -------------------------
                # 3.5: NEURO/RENAL refinement
                # -------------------------
                if top_group == "NEURO_RENAL" and nr_model is not None and nr_le is not None:
                    nr_pred_encoded = nr_model.predict(row_df)[0]
                    refined_group = nr_le.inverse_transform([nr_pred_encoded])[0]


        # 4) Show predictions
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Disease Group Prediction", unsafe_allow_html=True)
        st.success(f"Top-level predicted group: **{top_group}**")
        if refined_group != top_group:
            st.info(f"Refined NEURO/RENAL subgroup: **{refined_group}**")
        st.markdown("</div>", unsafe_allow_html=True)

        # 5) Recommend medicines from drug_map
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Medicine Recommendations", unsafe_allow_html=True)

        # Use refined group if present in map, otherwise top_group
        group_for_drugs = refined_group if refined_group in drug_map else top_group

        meds = drug_map.get(group_for_drugs, [])
        if not meds:
            st.info(f"No medicines stored for disease group '{group_for_drugs}'.")
        else:
            meds_df = pd.DataFrame(meds).sort_values("freq", ascending=False)
            st.write(
                f"Top medicines observed in first 24h for **{group_for_drugs}** "
                "(based on historical prescription frequencies):"
            )
            st.dataframe(meds_df[["drug", "freq", "count"]], use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------
# SIDEBAR: LOGOUT
# ---------------------------------------------------
st.sidebar.header("Session")
if st.session_state.auth["is_logged_in"]:
    st.sidebar.write(f"**User:** {st.session_state.auth['email']}")
    if st.sidebar.button("Logout"):
        st.session_state.auth = {"is_logged_in": False, "email": None}
        st.rerun()
else:
    st.sidebar.write("Not logged in.")

# ---------------------------------------------------
# ROUTER
# ---------------------------------------------------
if not st.session_state.auth["is_logged_in"]:
    login_page()
else:
    main_app_page()
