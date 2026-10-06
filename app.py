import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f7f9f5;
    }

    /* Main title */
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        color: #214d32;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #657565;
        margin-bottom: 35px;
    }

    /* Card */
    .card {
        background-color: white;
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #e0e8e1;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
    }

    /* Result card */
    .result {
        background-color: #e9f7ed;
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        border: 1px solid #c8e6cf;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    /* Result title */
    .result-title {
        color: #487052;
        font-size: 18px;
        margin-bottom: 8px;
    }

    /* Disease name */
    .result-disease {
        color: #1d6334;
        font-size: 32px;
        font-weight: bold;
    }

    /* Section heading */
    .section-title {
        color: #214d32;
        font-size: 24px;
        font-weight: 600;
    }

    /* Small information boxes */
    .info-box {
        background-color: white;
        padding: 18px;
        border-radius: 15px;
        border: 1px solid #e0e8e1;
        text-align: center;
    }

    /* Button */
    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        border: none;
        background-color: #2e7d4f;
        color: white;
        font-size: 17px;
        font-weight: 600;
        padding: 12px;
    }

    div.stButton > button:hover {
        background-color: #245f3d;
        color: white;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_excel(
        "crop_disease_sample_dataset.xlsx"
    )

    # Remove missing values
    df = df.dropna()

    return df


df = load_data()


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(df):

    # Features
    X = df.drop(
        "Disease",
        axis=1
    )

    # Target
    y = df["Disease"]

    # Convert categorical columns into numerical columns
    X = pd.get_dummies(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # Decision Tree Classifier
    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )

    # Train model
    model.fit(
        X_train,
        y_train
    )

    return model, X, X_test, y_test


model, X, X_test, y_test = train_model(df)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🌱 Crop Disease Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered crop disease prediction '
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MAIN LAYOUT
# ============================================================

left, right = st.columns(
    [1, 1],
    gap="large"
)


# ============================================================
# LEFT SIDE - INPUT
# ============================================================

with left:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">🌿 Crop Information</div>',
        unsafe_allow_html=True
    )

    st.write("")

    # --------------------------------------------------------
    # Crop
    # --------------------------------------------------------

    crops = sorted(
        df["Crop"].unique()
    )

    crop = st.selectbox(
        "Crop",
        crops
    )

    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        max_value=60.0,
        value=25.0,
        step=0.5
    )

    # --------------------------------------------------------
    # Humidity
    # --------------------------------------------------------

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    )

    # --------------------------------------------------------
    # Rainfall
    # --------------------------------------------------------

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=500.0,
        value=100.0,
        step=1.0
    )

    # --------------------------------------------------------
    # Leaf Color
    # --------------------------------------------------------

    leaf_colors = sorted(
        df["Leaf_Color"].unique()
    )

    leaf_color = st.selectbox(
        "Leaf Color",
        leaf_colors
    )

    # --------------------------------------------------------
    # Spots
    # --------------------------------------------------------

    spots = st.selectbox(
        "Are there spots on the leaf?",
        ["Yes", "No"]
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Prediction button
    # --------------------------------------------------------

    predict_button = st.button(
        "🔍 Predict Disease",
        use_container_width=True
    )


# ============================================================
# RIGHT SIDE - RESULT
# ============================================================

with right:

    st.markdown(
        '<div class="section-title">🩺 Prediction Result</div>',
        unsafe_allow_html=True
    )

    st.write("")

    if predict_button:

        # ----------------------------------------------------
        # Create dataframe from user input
        # ----------------------------------------------------

        user_data = pd.DataFrame({

            "Crop": [crop],

            "Temperature": [temperature],

            "Humidity": [humidity],

            "Rainfall": [rainfall],

            "Leaf_Color": [leaf_color],

            "Spots": [spots]

        })

        # ----------------------------------------------------
        # Encode user input
        # ----------------------------------------------------

        user_encoded = pd.get_dummies(
            user_data
        )

        # ----------------------------------------------------
        # Match training columns
        # ----------------------------------------------------

        user_encoded = user_encoded.reindex(
            columns=X.columns,
            fill_value=0
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            user_encoded
        )

        # ----------------------------------------------------
        # Prediction probability
        # ----------------------------------------------------

        probability = model.predict_proba(
            user_encoded
        )

        # Get disease
        disease = prediction[0]

        # Get confidence
        confidence = max(
            probability[0]
        ) * 100

        # ----------------------------------------------------
        # Display prediction
        # ----------------------------------------------------

        st.markdown("### 🩺 Prediction Result")

        st.info(f"🌱 **Predicted Disease:** {disease}")

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.write("")

        # ----------------------------------------------------
        # Probability chart
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            '📊 Prediction Probabilities'
            '</div>',
            unsafe_allow_html=True
        )

        probability_df = pd.DataFrame({

            "Disease": model.classes_,

            "Probability": probability[0] * 100

        })

        probability_df = probability_df.sort_values(
            "Probability",
            ascending=False
        )

        st.bar_chart(
            probability_df.set_index(
                "Disease"
            )
        )

    else:

        st.info(
            "🌱 Enter the crop information on the left "
            "and click **Predict Disease**."
        )


# ============================================================
# DATASET INFORMATION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">📊 Dataset Information</div>',
    unsafe_allow_html=True
)

st.write("")

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Total Samples",
        len(df)
    )


with col2:

    st.metric(
        "Features",
        len(X.columns)
    )


with col3:

    st.metric(
        "Disease Classes",
        df["Disease"].nunique()
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">🤖 Model Information</div>',
    unsafe_allow_html=True
)

st.write("")

info1, info2, info3 = st.columns(3)


with info1:

    st.markdown(
        """
        <div class="info-box">
            <b>Algorithm</b><br>
            Decision Tree Classifier
        </div>
        """,
        unsafe_allow_html=True
    )


with info2:

    st.markdown(
        """
        <div class="info-box">
            <b>Max Tree Depth</b><br>
            5
        </div>
        """,
        unsafe_allow_html=True
    )


with info3:

    st.markdown(
        f"""
        <div class="info-box">
            <b>Disease Classes</b><br>
            {df["Disease"].nunique()}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "⚠️ This project is an educational ML demonstration. "
    "The prediction should not be treated as a professional "
    "agricultural diagnosis."
)