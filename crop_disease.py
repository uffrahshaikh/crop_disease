import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_excel("crop_disease_sample_dataset.xlsx")

print("Dataset loaded successfully!")
print(df.head())


# ==========================================
# 2. REMOVE MISSING VALUES
# ==========================================

df = df.dropna()


# ==========================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Disease", axis=1)
y = df["Disease"]


# ==========================================
# 4. ENCODE CATEGORICAL DATA
# ==========================================

X = pd.get_dummies(X)


# ==========================================
# 5. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ==========================================
# 6. CREATE DECISION TREE
# ==========================================

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)


# ==========================================
# 7. TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)

print("\nModel trained successfully!")


# ==========================================
# 8. TEST MODEL
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL ACCURACY")
print("==============================")

print("Accuracy:", round(accuracy * 100, 2), "%")


print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==========================================
# 9. USER INPUT
# ==========================================

print("\n======================================")
print("     CROP DISEASE PREDICTION")
print("======================================")

crop = input("Enter Crop: ")

temperature = float(
    input("Enter Temperature: ")
)

humidity = float(
    input("Enter Humidity: ")
)

rainfall = float(
    input("Enter Rainfall: ")
)

leaf_color = input(
    "Enter Leaf Color: "
)

spots = input(
    "Are there spots on the leaf? (Yes/No): "
)


# ==========================================
# 10. CREATE USER DATA
# ==========================================

user_data = pd.DataFrame({
    "Crop": [crop],
    "Temperature": [temperature],
    "Humidity": [humidity],
    "Rainfall": [rainfall],
    "Leaf_Color": [leaf_color],
    "Spots": [spots]
})


# ==========================================
# 11. ENCODE USER DATA
# ==========================================

user_encoded = pd.get_dummies(user_data)

user_encoded = user_encoded.reindex(
    columns=X.columns,
    fill_value=0
)


# ==========================================
# 12. PREDICT DISEASE
# ==========================================

prediction = model.predict(user_encoded)

probability = model.predict_proba(user_encoded)


# ==========================================
# 13. DISPLAY RESULT
# ==========================================

print("\n======================================")
print("             RESULT")
print("======================================")

print("Crop:", crop)
print("Temperature:", temperature)
print("Humidity:", humidity)
print("Rainfall:", rainfall)
print("Leaf Color:", leaf_color)
print("Spots:", spots)

print("\n--------------------------------------")

print("PREDICTED DISEASE:", prediction[0])

print("--------------------------------------")

print("\nPrediction Probability:")

for disease, prob in zip(
    model.classes_,
    probability[0]
):
    print(
        disease,
        "->",
        round(prob * 100, 2),
        "%"
    )

print("\n======================================")
print("       PREDICTION COMPLETED")
print("======================================")