from pathlib import Path
import re

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_FILE = Path("drone_features.csv")

df = pd.read_csv(DATA_FILE)

# Extract trajectory ID from names such as t1, t2, ..., t5.
def get_trajectory(filename):
    match = re.search(r"_t([1-5])(?:_|\.mat$)", str(filename))
    return match.group(1) if match else "unknown"

df["trajectory_group"] = df["filename"].apply(get_trajectory)

# Prepare numeric features.
excluded = ["filename", "fault_group", "label", "trajectory_group"]
X = df.drop(columns=excluded, errors="ignore")
X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
X = X.dropna(axis=1, how="all")
X = X.loc[:, X.nunique(dropna=True) > 1]
y = df["label"]

def make_model(kind):
    if kind == "Logistic Regression":
        classifier = LogisticRegression(
            max_iter=5000,
            class_weight="balanced"
        )
    else:
        classifier = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42
        )

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", classifier)
    ])

def evaluate(name, X_train, X_test, y_train, y_test):
    print(f"\n{name}")
    print("-" * 45)

    for model_name in ["Logistic Regression", "Random Forest"]:
        model = make_model(model_name)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        print(
            f"{model_name}: "
            f"Accuracy={accuracy_score(y_test, pred):.3f}, "
            f"Precision={precision_score(y_test, pred, zero_division=0):.3f}, "
            f"Recall={recall_score(y_test, pred, zero_division=0):.3f}, "
            f"F1={f1_score(y_test, pred, zero_division=0):.3f}"
        )

print("Dataset shape:", df.shape)
print("Trajectory groups:", df["trajectory_group"].value_counts().to_dict())

# Test A: random split for comparison with the initial experiment.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
evaluate("Random split", X_train, X_test, y_train, y_test)

# Test B: hold out complete trajectories.
valid = df["trajectory_group"] != "unknown"
X_group = X.loc[valid]
y_group = y.loc[valid]
groups = df.loc[valid, "trajectory_group"]

splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.25, random_state=42
)
train_idx, test_idx = next(splitter.split(X_group, y_group, groups))

X_train, X_test = X_group.iloc[train_idx], X_group.iloc[test_idx]
y_train, y_test = y_group.iloc[train_idx], y_group.iloc[test_idx]

print("\nHeld-out trajectories:", sorted(groups.iloc[test_idx].unique()))
print("Training classes:", y_train.value_counts().to_dict())
print("Testing classes:", y_test.value_counts().to_dict())

if y_train.nunique() < 2 or y_test.nunique() < 2:
    print(
        "\nWARNING: This trajectory split does not contain both classes "
        "in training and testing. Do not interpret its metrics; we need "
        "a better grouped split."
    )
else:
    evaluate("Held-out trajectory split", X_train, X_test, y_train, y_test)