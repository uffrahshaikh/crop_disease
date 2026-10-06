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

.main {
    background-color: #f7f9f5;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #214d32;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #657565;
    margin-bottom: 35px;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.result {
    background-color: #e9f7ed;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #c8e6cf;
}

.result-title {
    color: #487052;
    font-size: 18px;
}

.result-disease {
    color: #1d6334;
    font-size: 32px;
    font-weight: bold;
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

    df = df.dropna()

    return df


df = load_data()


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(df):

    X = df.drop(
        "Disease",
        axis=1
    )

    y = df["Disease"]

    # Encode categorical columns
    X = pd.get_dummies(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # Decision Tree
    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model, X


model, X = train_model(df)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🌱 Crop Disease Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered crop disease prediction using Decision Tree Classifier'
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
# INPUT SECTION
# ============================================================

with left:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🌿 Crop Information")

    # Get unique crop values
    crops = sorted(
        df["Crop"].unique()
    )

    crop = st.selectbox(
        "Crop",
        crops
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        max_value=60.0,
        value=25.0,
        step=0.5
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=500.0,
        value=100.0,
        step=1.0
    )

    leaf_colors = sorted(
        df["Leaf_Color"].unique()
    )

    leaf_color = st.selectbox(
        "Leaf Color",
        leaf_colors
    )

    spots = st.selectbox(
        "Are there spots on the leaf?",
        ["Yes", "No"]
    )

    st.markdown("</div>", unsafe_allow_html=True)


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    predict_button = st.button(
        "🔍 Predict Disease",
        use_container_width=True
    )


# ============================================================
# RESULT SECTION
# ============================================================

with right:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🩺 Prediction Result")

    if predict_button:

        # Create user dataframe
        user_data = pd.DataFrame({

            "Crop": [crop],

            "Temperature": [temperature],

            "Humidity": [humidity],

            "Rainfall": [rainfall],

            "Leaf_Color": [leaf_color],

            "Spots": [spots]
        })


        # Encode user input
        user_encoded = pd.get_dummies(
            user_data
        )


        # Match training columns
        user_encoded = user_encoded.reindex(
            columns=X.columns,
            fill_value=0
        )


        # Prediction
        prediction = model.predict(
            user_encoded
        )


        # Probability
        probability = model.predict_proba(
            user_encoded
        )


        disease = prediction[0]

        confidence = max(
            probability[0]
        ) * 100


        # Display result

        st.markdown(
            f"""
            <div class="result">

                <div class="result-title">
                    Predicted Disease
                </div>

                <div class="result-disease">
                    {disease}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.write("")

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )


        st.write("### 📊 Prediction Probabilities")

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
            "Enter the crop information on the left "
            "and click **Predict Disease**."
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# DATASET INFORMATION
# ============================================================

st.divider()

st.subheader("📊 Dataset Information")

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
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "⚠️ This project is an educational ML demonstration. "
    "The prediction should not be treated as a professional "
    "agricultural diagnosis."
)