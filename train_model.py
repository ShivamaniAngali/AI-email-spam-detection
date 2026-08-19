import pandas as pd
import pickle
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n========================================")
print("       EMAIL SPAM MODEL COMPARISON")
print("========================================\n")

df = pd.read_csv("spam.csv")

print("Dataset loaded successfully.")
print("Total emails:", len(df))


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = ["label", "subject", "message"]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column '{column}' not found in spam.csv"
        )


# ============================================================
# 3. CLEAN DATA
# ============================================================

df["subject"] = df["subject"].fillna("")
df["message"] = df["message"].fillna("")

df["text"] = (
    df["subject"].astype(str)
    + " "
    + df["message"].astype(str)
)


# ============================================================
# 4. CONVERT LABELS
# ============================================================

df["label"] = df["label"].astype(str).str.lower().str.strip()

# Convert spam/ham into 1/0

label_mapping = {
    "spam": 1,
    "ham": 0,
    "1": 1,
    "0": 0
}

df["target"] = df["label"].map(label_mapping)


# Check for invalid labels

if df["target"].isna().any():

    print("\nInvalid labels found:")

    print(
        df.loc[
            df["target"].isna(),
            "label"
        ].unique()
    )

    raise ValueError(
        "spam.csv contains labels other than spam/ham or 1/0."
    )


# ============================================================
# 5. DATASET INFORMATION
# ============================================================

print("\nDataset Information")
print("----------------------------------------")

print(
    "Spam emails:",
    int(df["target"].sum())
)

print(
    "Ham emails:",
    int((df["target"] == 0).sum())
)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X = df["text"]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining emails:", len(X_train))
print("Testing emails:", len(X_test))


# ============================================================
# 7. TF-IDF VECTORIZATION
# ============================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print(
    "TF-IDF features:",
    X_train_tfidf.shape[1]
)


# ============================================================
# 8. DEFINE MODELS
# ============================================================

models = {

    "Multinomial Naive Bayes":
        MultinomialNB(),

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Linear SVM":
        LinearSVC(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
}


# ============================================================
# 9. TRAIN AND COMPARE MODELS
# ============================================================

results = []

trained_models = {}

print("\n========================================")
print("       TRAINING MODELS")
print("========================================\n")


for model_name, model in models.items():

    print(f"Training {model_name}...")

    start_time = time.time()

    model.fit(
        X_train_tfidf,
        y_train
    )

    end_time = time.time()

    training_time = end_time - start_time

    # Prediction

    y_pred = model.predict(
        X_test_tfidf
    )

    # Metrics

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    results.append({

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1-Score": f1,

        "Training Time (seconds)": training_time

    })

    trained_models[model_name] = model

    print(
        f"Accuracy  : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision : {precision * 100:.2f}%"
    )

    print(
        f"Recall    : {recall * 100:.2f}%"
    )

    print(
        f"F1-Score  : {f1 * 100:.2f}%"
    )

    print(
        f"Time      : {training_time:.4f} seconds"
    )

    print("----------------------------------------")


# ============================================================
# 10. CREATE COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)


# Convert metrics to percentages for display

display_df = results_df.copy()

display_df["Accuracy"] = (
    display_df["Accuracy"] * 100
).round(2)

display_df["Precision"] = (
    display_df["Precision"] * 100
).round(2)

display_df["Recall"] = (
    display_df["Recall"] * 100
).round(2)

display_df["F1-Score"] = (
    display_df["F1-Score"] * 100
).round(2)

display_df["Training Time (seconds)"] = (
    display_df["Training Time (seconds)"].round(4)
)


# ============================================================
# 11. DISPLAY TABLE
# ============================================================

print("\n========================================")
print("       MODEL COMPARISON")
print("========================================\n")

print(
    display_df.to_string(
        index=False
    )
)


# ============================================================
# 12. SAVE COMPARISON TABLE
# ============================================================

display_df.to_csv(
    "model_comparison.csv",
    index=False
)

print(
    "\nSaved: model_comparison.csv"
)


# ============================================================
# 13. SELECT BEST MODEL
# ============================================================

# Select model based on highest F1-score

best_index = results_df[
    "F1-Score"
].idxmax()

best_model_name = results_df.loc[
    best_index,
    "Model"
]

best_model = trained_models[
    best_model_name
]

best_accuracy = results_df.loc[
    best_index,
    "Accuracy"
]

best_precision = results_df.loc[
    best_index,
    "Precision"
]

best_recall = results_df.loc[
    best_index,
    "Recall"
]

best_f1 = results_df.loc[
    best_index,
    "F1-Score"
]


# ============================================================
# 14. SAVE BEST MODEL
# ============================================================

with open(
    "spam_model.pkl",
    "wb"
) as file:

    pickle.dump(
        best_model,
        file
    )


# ============================================================
# 15. SAVE TF-IDF VECTORIZER
# ============================================================

with open(
    "vectorizer.pkl",
    "wb"
) as file:

    pickle.dump(
        vectorizer,
        file
    )


# ============================================================
# 16. SAVE MODEL ACCURACY
# ============================================================

with open(
    "model_accuracy.txt",
    "w"
) as file:

    file.write(
        f"{best_accuracy * 100:.2f}"
    )


# ============================================================
# 17. SAVE BEST MODEL INFORMATION
# ============================================================

with open(
    "best_model.txt",
    "w"
) as file:

    file.write(
        f"Best Model: {best_model_name}\n"
    )

    file.write(
        f"Accuracy: {best_accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Precision: {best_precision * 100:.2f}%\n"
    )

    file.write(
        f"Recall: {best_recall * 100:.2f}%\n"
    )

    file.write(
        f"F1-Score: {best_f1 * 100:.2f}%\n"
    )


# ============================================================
# 18. GENERATE COMPARISON GRAPH
# ============================================================

# ============================================================
# 18. GENERATE IMPROVED COMPARISON GRAPH
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]

# Convert metrics to percentages
graph_df = results_df.copy()

for metric in metrics:
    graph_df[metric] = graph_df[metric] * 100


# ------------------------------------------------------------
# CREATE GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(14, 8))

x = range(len(graph_df))

width = 0.18


# Accuracy
bars1 = plt.bar(
    [value - 1.5 * width for value in x],
    graph_df["Accuracy"],
    width=width,
    label="Accuracy"
)

# Precision
bars2 = plt.bar(
    [value - 0.5 * width for value in x],
    graph_df["Precision"],
    width=width,
    label="Precision"
)

# Recall
bars3 = plt.bar(
    [value + 0.5 * width for value in x],
    graph_df["Recall"],
    width=width,
    label="Recall"
)

# F1-Score
bars4 = plt.bar(
    [value + 1.5 * width for value in x],
    graph_df["F1-Score"],
    width=width,
    label="F1-Score"
)


# ------------------------------------------------------------
# IMPORTANT: ZOOM Y-AXIS
# ------------------------------------------------------------

plt.ylim(85, 94)


# ------------------------------------------------------------
# ADD EXACT VALUES ABOVE EACH BAR
# ------------------------------------------------------------

def add_values(bars):

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.08,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )


add_values(bars1)
add_values(bars2)
add_values(bars3)
add_values(bars4)


# ------------------------------------------------------------
# GRAPH LABELS
# ------------------------------------------------------------

plt.xticks(
    list(x),
    graph_df["Model"],
    rotation=10
)

plt.ylabel(
    "Performance (%)",
    fontsize=12
)

plt.xlabel(
    "Machine Learning Model",
    fontsize=12
)

plt.title(
    "Machine Learning Model Performance Comparison",
    fontsize=16,
    fontweight="bold"
)


# ------------------------------------------------------------
# GRID AND LEGEND
# ------------------------------------------------------------

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.legend(
    fontsize=10
)


# ------------------------------------------------------------
# SAVE GRAPH
# ------------------------------------------------------------

plt.tight_layout()

plt.savefig(
    "model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved: model_comparison.png"
)

# ============================================================
# 19. FINAL RESULT
# ============================================================

print("\n========================================")
print("          BEST MODEL SELECTED")
print("========================================")

print(
    f"\nBest Model: {best_model_name}"
)

print(
    f"Accuracy : {best_accuracy * 100:.2f}%"
)

print(
    f"Precision: {best_precision * 100:.2f}%"
)

print(
    f"Recall   : {best_recall * 100:.2f}%"
)

print(
    f"F1-Score : {best_f1 * 100:.2f}%"
)

print("\n----------------------------------------")

print(
    "Best model saved as: spam_model.pkl"
)

print(
    "Vectorizer saved as: vectorizer.pkl"
)

print(
    "Comparison table saved as: model_comparison.csv"
)

print(
    "Comparison graph saved as: model_comparison.png"
)

print("\n========================================")
print("          TRAINING COMPLETE")
print("========================================\n")
