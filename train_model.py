from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# =====================================================
# DRONEPROPA - MACHINE LEARNING MODEL TRAINING
# =====================================================

DATA_FILE = Path("drone_features.csv")

print("=" * 60)
print("DRONE PROPELLER FAULT DETECTION")
print("=" * 60)

# 1. Load the dataset
if not DATA_FILE.exists():
    raise FileNotFoundError(
        "drone_features.csv not found. Run main.py first."
    )

df = pd.read_csv(DATA_FILE)

print("\nDataset shape:", df.shape)
print("\nClass distribution:")
print(df["label"].value_counts().sort_index())

# 2. Prepare features and labels
# Exclude identifiers and labels from the input features.
excluded_columns = ["filename", "fault_group", "label"]

X = df.drop(columns=excluded_columns, errors="ignore")
y = df["label"]

# Keep numeric features only.
X = X.select_dtypes(include=[np.number])

# Replace infinite values with missing values.
X = X.replace([np.inf, -np.inf], np.nan)

# Remove features that contain no usable values.
X = X.dropna(axis=1, how="all")

# Remove constant features.
X = X.loc[:, X.nunique(dropna=True) > 1]

if X.empty:
    raise ValueError("No usable numerical features were found.")

if y.nunique() < 2:
    raise ValueError("The dataset must contain both healthy and faulty flights.")

print("\nNumber of usable features:", X.shape[1])
print("Number of flights:", len(X))

# 3. Split into training and testing sets.
# Stratification preserves the healthy/faulty class proportions.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining flights:", len(X_train))
print("Testing flights:", len(X_test))

# 4. Define the models.
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=5000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ),

    "SVM": SVC(
        kernel="rbf",
        class_weight="balanced"
    )
}

results = []
trained_models = {}

# 5. Train and evaluate each model.
for model_name, classifier in models.items():

    print("\n" + "-" * 60)
    print("Training:", model_name)

    # Imputation and scaling are fitted only on training data.
    # This avoids leaking test-set statistics into training.
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    trained_models[model_name] = pipeline

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Healthy", "Faulty"],
        zero_division=0
    ))

# 6. Compare model performance.
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1-Score",
    ascending=False
)

results_df.to_csv("model_comparison.csv", index=False)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(results_df.to_string(index=False))

# 7. Plot model comparison.
plt.figure(figsize=(10, 6))

results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1-Score"]
].plot(kind="bar", ax=plt.gca())

plt.title("DronePropA Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=20, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300)
plt.close()

print("\nSaved: model_comparison.png")

# 8. Save the confusion matrix for the best model.
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
best_predictions = best_model.predict(X_test)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    best_predictions,
    labels=[0, 1],
    display_labels=["Healthy", "Faulty"],
    cmap="Blues"
)

plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

print("Saved: confusion_matrix.png")

# 9. Save the best model.
joblib.dump(best_model, "best_drone_fault_model.joblib")

print("Saved: best_drone_fault_model.joblib")

print("\nBest model by F1-score:", best_model_name)
print("\nTRAINING COMPLETED!")