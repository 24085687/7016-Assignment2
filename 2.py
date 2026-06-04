# -*- coding: utf-8 -*-
"""
Assignment 2 - AI for Medicine
Topic: Diabetes Risk Prediction and Patient Segmentation
Dataset:
diabetes_binary_5050split_health_indicators_BRFSS2015.csv
"""

# ============================================================
# 1. Import libraries
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    silhouette_score
)

# Plot style
sns.set_theme(style="whitegrid")

# Create output folders
os.makedirs("figures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ============================================================
# 2. Load dataset
# ============================================================

file_name = "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"

if not os.path.exists(file_name):
    print("Current working directory:")
    print(os.getcwd())
    raise FileNotFoundError(
        "Dataset not found. Please place the CSV file in the same folder as this Python script, "
        "or set the working directory in Spyder to the folder containing the dataset."
    )

df = pd.read_csv(file_name)

print("\n==============================")
print("1. Dataset Loaded Successfully")
print("==============================")
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 3. Data understanding
# ============================================================

print("\n==============================")
print("2. Data Understanding")
print("==============================")

print("\nData information:")
print(df.info())

print("\nMissing values:")
missing_values = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": df.isnull().mean() * 100
})
print(missing_values)

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nTarget variable distribution:")
target_counts = df["Diabetes_binary"].value_counts().sort_index()
print(target_counts)

print("\nTarget variable percentage:")
target_percent = df["Diabetes_binary"].value_counts(normalize=True).sort_index() * 100
print(target_percent)

print("\nSummary statistics:")
summary_statistics = df.describe().T
print(summary_statistics)

missing_values.to_csv("outputs/missing_values.csv")
summary_statistics.to_csv("outputs/summary_statistics.csv")


# ============================================================
# 4. Data preprocessing
# ============================================================

print("\n==============================")
print("3. Data Preprocessing")
print("==============================")

# Remove missing values
df_clean = df.dropna().copy()

# Note: Duplicates are NOT removed
# Reason: This is survey data; identical responses are valid and not errors

# Ensure target variable is integer type
df_clean["Diabetes_binary"] = df_clean["Diabetes_binary"].astype(int)

print("Data shape after preprocessing:", df_clean.shape)

print("\nRange check:")
range_check = df_clean.describe().T[["min", "max", "mean", "std"]]
print(range_check)

df_clean.to_csv("outputs/diabetes_cleaned.csv", index=False)


# ============================================================
# 5. EDA Figure 1: Diabetes status distribution
# ============================================================

plt.figure(figsize=(7, 5))
ax = sns.countplot(data=df_clean, x="Diabetes_binary")

plt.title("Distribution of Diabetes Status")
plt.xlabel("Diabetes Status (0 = No Diabetes, 1 = Prediabetes or Diabetes)")
plt.ylabel("Number of Records")

for p in ax.patches:
    ax.annotate(
        str(int(p.get_height())),
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig("figures/fig1_diabetes_status_distribution.png", dpi=300)
plt.show()


# ============================================================
# 6. EDA Figure 2: BMI by diabetes status
# ============================================================

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_clean, x="Diabetes_binary", y="BMI")

plt.title("BMI by Diabetes Status")
plt.xlabel("Diabetes Status (0 = No Diabetes, 1 = Prediabetes or Diabetes)")
plt.ylabel("BMI")
plt.tight_layout()
plt.savefig("figures/fig2_bmi_by_diabetes_status.png", dpi=300)
plt.show()

print("\nMean BMI by diabetes status:")
print(df_clean.groupby("Diabetes_binary")["BMI"].mean())


# ============================================================
# 7. EDA Figure 3: HighBP and HighChol by diabetes status
# ============================================================

risk_factor_summary = df_clean.groupby("Diabetes_binary")[["HighBP", "HighChol"]].mean().reset_index()

risk_factor_long = risk_factor_summary.melt(
    id_vars="Diabetes_binary",
    value_vars=["HighBP", "HighChol"],
    var_name="Risk_Factor",
    value_name="Proportion"
)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=risk_factor_long,
    x="Risk_Factor",
    y="Proportion",
    hue="Diabetes_binary"
)

plt.title("High Blood Pressure and High Cholesterol by Diabetes Status")
plt.xlabel("Risk Factor")
plt.ylabel("Proportion")
plt.legend(title="Diabetes Status", labels=["No Diabetes", "Prediabetes/Diabetes"])
plt.tight_layout()
plt.savefig("figures/fig3_highbp_highchol_by_diabetes_status.png", dpi=300)
plt.show()

print("\nHighBP and HighChol summary:")
print(risk_factor_summary)


# ============================================================
# 8. EDA Figure 4: Diabetes risk by age group
# ============================================================

age_diabetes_table = pd.crosstab(
    df_clean["Age"],
    df_clean["Diabetes_binary"],
    normalize="index"
)

plt.figure(figsize=(10, 5))
age_diabetes_table[1].plot(kind="bar")

plt.title("Proportion of Prediabetes or Diabetes by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Proportion of Prediabetes or Diabetes")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("figures/fig4_diabetes_rate_by_age_group.png", dpi=300)
plt.show()

print("\nDiabetes rate by age group:")
print(age_diabetes_table)


# ============================================================
# 9. EDA Figure 5: Correlation heatmap
# ============================================================

corr_matrix = df_clean.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0,
    linewidths=0.2
)

plt.title("Correlation Heatmap of Health Indicators")
plt.tight_layout()
plt.savefig("figures/fig5_correlation_heatmap.png", dpi=300)
plt.show()

diabetes_corr = corr_matrix["Diabetes_binary"].drop("Diabetes_binary").sort_values(ascending=False)

print("\nTop 10 correlations with Diabetes_binary:")
print(diabetes_corr.head(10))

diabetes_corr.to_csv("outputs/correlation_with_diabetes.csv")


# ============================================================
# 10. Prepare data for classification
# ============================================================

print("\n==============================")
print("4. Classification Data Preparation")
print("==============================")

target_col = "Diabetes_binary"
feature_cols = [col for col in df_clean.columns if col != target_col]

X = df_clean[feature_cols]
y = df_clean[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True))

print("\nTest target distribution:")
print(y_test.value_counts(normalize=True))


# ============================================================
# 11. Classification model 1: Logistic Regression
# ============================================================

print("\n==============================")
print("5. Logistic Regression")
print("==============================")

# Standardization required for Logistic Regression
scaler_cls = StandardScaler()
X_train_scaled = scaler_cls.fit_transform(X_train)
X_test_scaled = scaler_cls.transform(X_test)

log_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

log_model.fit(X_train_scaled, y_train)

y_pred_log = log_model.predict(X_test_scaled)
y_prob_log = log_model.predict_proba(X_test_scaled)[:, 1]

log_accuracy = accuracy_score(y_test, y_pred_log)
log_precision = precision_score(y_test, y_pred_log)
log_recall = recall_score(y_test, y_pred_log)
log_f1 = f1_score(y_test, y_pred_log)
log_auc = roc_auc_score(y_test, y_prob_log)

print("Accuracy:", log_accuracy)
print("Precision:", log_precision)
print("Recall:", log_recall)
print("F1-score:", log_f1)
print("ROC-AUC:", log_auc)

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred_log,
    target_names=["No Diabetes", "Prediabetes/Diabetes"]
))

cm_log = confusion_matrix(y_test, y_pred_log)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_log,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Diabetes", "Prediabetes/Diabetes"],
    yticklabels=["No Diabetes", "Prediabetes/Diabetes"]
)

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("figures/fig6_confusion_matrix_logistic_regression.png", dpi=300)
plt.show()


# ============================================================
# 12. Classification model 2: Decision Tree
# ============================================================

print("\n==============================")
print("6. Decision Tree")
print("==============================")

tree_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=50,
    min_samples_leaf=25,
    random_state=42
)

tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)
y_prob_tree = tree_model.predict_proba(X_test)[:, 1]

tree_accuracy = accuracy_score(y_test, y_pred_tree)
tree_precision = precision_score(y_test, y_pred_tree)
tree_recall = recall_score(y_test, y_pred_tree)
tree_f1 = f1_score(y_test, y_pred_tree)
tree_auc = roc_auc_score(y_test, y_prob_tree)

print("Accuracy:", tree_accuracy)
print("Precision:", tree_precision)
print("Recall:", tree_recall)
print("F1-score:", tree_f1)
print("ROC-AUC:", tree_auc)

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred_tree,
    target_names=["No Diabetes", "Prediabetes/Diabetes"]
))

cm_tree = confusion_matrix(y_test, y_pred_tree)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_tree,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["No Diabetes", "Prediabetes/Diabetes"],
    yticklabels=["No Diabetes", "Prediabetes/Diabetes"]
)

plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("figures/fig7_confusion_matrix_decision_tree.png", dpi=300)
plt.show()


# ============================================================
# 13. Model comparison
# ============================================================

print("\n==============================")
print("7. Classification Model Comparison")
print("==============================")

model_comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree"],
    "Accuracy": [log_accuracy, tree_accuracy],
    "Precision": [log_precision, tree_precision],
    "Recall": [log_recall, tree_recall],
    "F1-score": [log_f1, tree_f1],
    "ROC-AUC": [log_auc, tree_auc]
})

print(model_comparison)

model_comparison.to_csv("outputs/model_comparison.csv", index=False)

ax = model_comparison.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]].plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Classification Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("figures/fig8_model_comparison.png", dpi=300)
plt.show()


# ============================================================
# 14. Decision Tree feature importance
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": tree_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Decision Tree feature importances:")
print(feature_importance.head(10))

feature_importance.to_csv("outputs/decision_tree_feature_importance.csv", index=False)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=feature_importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Feature Importances - Decision Tree")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("figures/fig9_decision_tree_feature_importance.png", dpi=300)
plt.show()


# ============================================================
# 15. Decision Tree visualization - improved clear version
# ============================================================

# 15.1 Full tree: For GitHub / Appendix
# Large figure, saved as PNG + PDF + SVG
# Vector formats remain clear when zoomed

fig, ax = plt.subplots(figsize=(46, 24))

plot_tree(
    tree_model,
    feature_names=feature_cols,
    class_names=["No Diabetes", "Prediabetes/Diabetes"],
    filled=True,
    rounded=True,
    fontsize=7,
    impurity=True,
    proportion=False,
    precision=2,
    ax=ax
)

plt.title("Decision Tree Structure - Full Version", fontsize=20, pad=25)
plt.tight_layout(pad=3.0)

plt.savefig(
    "figures/fig10_decision_tree_structure_full_clear.png",
    dpi=250,
    bbox_inches="tight"
)

plt.savefig(
    "figures/fig10_decision_tree_structure_full_clear.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "figures/fig10_decision_tree_structure_full_clear.svg",
    bbox_inches="tight"
)

plt.show()


# 15.2 Simplified tree: For report and presentation
# Top 3 levels only, clear text for explanation

fig, ax = plt.subplots(figsize=(28, 15))

plot_tree(
    tree_model,
    feature_names=feature_cols,
    class_names=["No Diabetes", "Prediabetes/Diabetes"],
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=3,
    impurity=True,
    proportion=False,
    precision=2,
    ax=ax
)

plt.title("Simplified Decision Tree Structure (Top 3 Levels)", fontsize=18, pad=25)
plt.tight_layout(pad=3.0)

plt.savefig(
    "figures/fig10_decision_tree_structure_simplified.png",
    dpi=400,
    bbox_inches="tight"
)

plt.savefig(
    "figures/fig10_decision_tree_structure_simplified.pdf",
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 16. Prepare data for clustering
# ============================================================

print("\n==============================")
print("8. Clustering Data Preparation")
print("==============================")

# Clustering is unsupervised; Diabetes_binary is excluded
cluster_features = [
    "HighBP",
    "HighChol",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Age",
    "Education",
    "Income"
]

X_cluster = df_clean[cluster_features].copy()

scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

print("Clustering features:")
print(cluster_features)


# ============================================================
# 17. K selection: Elbow Method and Silhouette Score
# ============================================================

print("\n==============================")
print("9. K-Means K Selection")
print("==============================")

k_values = range(2, 8)

inertias = []
silhouette_scores = []

# Silhouette calculation is slow; sample 5000 points
sample_size = min(5000, X_cluster_scaled.shape[0])
np.random.seed(42)
sample_indices = np.random.choice(X_cluster_scaled.shape[0], sample_size, replace=False)
X_sample = X_cluster_scaled[sample_indices]

for k in k_values:
    kmeans_temp = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels_full = kmeans_temp.fit_predict(X_cluster_scaled)
    inertias.append(kmeans_temp.inertia_)

    labels_sample = labels_full[sample_indices]
    sil_score = silhouette_score(X_sample, labels_sample)
    silhouette_scores.append(sil_score)

k_selection_table = pd.DataFrame({
    "k": list(k_values),
    "Inertia": inertias,
    "Silhouette_Score": silhouette_scores
})

print(k_selection_table)

k_selection_table.to_csv("outputs/kmeans_k_selection.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, marker="o")

plt.title("Elbow Method for K-Means")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.tight_layout()
plt.savefig("figures/fig11_elbow_method.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), silhouette_scores, marker="o")

plt.title("Silhouette Score for Different k Values")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig("figures/fig12_silhouette_scores.png", dpi=300)
plt.show()


# ============================================================
# 18. Final K-Means clustering
# ============================================================

print("\n==============================")
print("10. Final K-Means Clustering")
print("==============================")

# k = 3 for clinical interpretability: Low, Medium, High risk
final_k = 3

kmeans = KMeans(
    n_clusters=final_k,
    random_state=42,
    n_init=10
)

df_clustered = df_clean.copy()
df_clustered["Cluster"] = kmeans.fit_predict(X_cluster_scaled)

cluster_counts = df_clustered["Cluster"].value_counts().sort_index()

print("Cluster counts:")
print(cluster_counts)

cluster_counts.to_csv("outputs/cluster_counts.csv")


# ============================================================
# 19. Cluster profile
# ============================================================

cluster_profile = df_clustered.groupby("Cluster")[cluster_features + ["Diabetes_binary"]].mean()

cluster_profile = cluster_profile.rename(columns={
    "Diabetes_binary": "Diabetes_Risk_Rate"
})

print("\nCluster profile:")
print(cluster_profile)

cluster_profile.to_csv("outputs/cluster_profile.csv")

key_cluster_vars = [
    "BMI",
    "HighBP",
    "HighChol",
    "GenHlth",
    "Age",
    "Diabetes_Risk_Rate"
]

cluster_profile_plot = cluster_profile[key_cluster_vars].reset_index().melt(
    id_vars="Cluster",
    value_vars=key_cluster_vars,
    var_name="Variable",
    value_name="Mean_Value"
)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=cluster_profile_plot,
    x="Variable",
    y="Mean_Value",
    hue="Cluster"
)

plt.title("Cluster Profiles Based on Key Health Indicators")
plt.xlabel("Variable")
plt.ylabel("Mean Value")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("figures/fig13_cluster_profiles.png", dpi=300)
plt.show()


# ============================================================
# 20. PCA visualization for clusters
# ============================================================

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_cluster_scaled)

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Cluster": df_clustered["Cluster"].values,
    "Diabetes_binary": df_clustered["Diabetes_binary"].values
})

pca_sample = pca_df.sample(n=min(5000, len(pca_df)), random_state=42)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=pca_sample,
    x="PC1",
    y="PC2",
    hue="Cluster",
    alpha=0.6,
    s=20
)

plt.title("K-Means Clusters Visualized by PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig("figures/fig14_pca_cluster_visualization.png", dpi=300)
plt.show()

print("\nPCA explained variance ratio:")
print(pca.explained_variance_ratio_)


# ============================================================
# 21. Diabetes rate by cluster
# ============================================================

cluster_diabetes_table = pd.crosstab(
    df_clustered["Cluster"],
    df_clustered["Diabetes_binary"],
    normalize="index"
)

print("\nDiabetes status distribution by cluster:")
print(cluster_diabetes_table)

cluster_diabetes_table.to_csv("outputs/cluster_diabetes_distribution.csv")

plt.figure(figsize=(8, 5))
cluster_diabetes_table[1].plot(kind="bar")

plt.title("Prediabetes or Diabetes Rate by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Proportion of Prediabetes or Diabetes")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("figures/fig15_diabetes_rate_by_cluster.png", dpi=300)
plt.show()


# ============================================================
# 22. Clinical and business intelligence summary
# ============================================================

print("\n==============================")
print("11. Clinical and Business Intelligence")
print("==============================")

highest_risk_cluster = cluster_profile["Diabetes_Risk_Rate"].idxmax()
highest_risk_rate = cluster_profile.loc[highest_risk_cluster, "Diabetes_Risk_Rate"]

lowest_risk_cluster = cluster_profile["Diabetes_Risk_Rate"].idxmin()
lowest_risk_rate = cluster_profile.loc[lowest_risk_cluster, "Diabetes_Risk_Rate"]

print("Highest-risk cluster: Cluster", highest_risk_cluster)
print("Diabetes risk rate:", round(highest_risk_rate, 3))

print("\nLowest-risk cluster: Cluster", lowest_risk_cluster)
print("Diabetes risk rate:", round(lowest_risk_rate, 3))

print("\nClinical interpretation:")
print("1. Classification models can support preliminary diabetes risk screening.")
print("2. Recall is important because missing high-risk individuals may delay early intervention.")
print("3. K-Means clustering identifies patient groups with different health risk profiles.")
print("4. The highest-risk cluster can be prioritized for lifestyle intervention and blood glucose testing.")
print("5. Healthcare managers can use patient segmentation to allocate screening resources more efficiently.")


# ============================================================
# 23. Suggested report sentences
# ============================================================

print("\n==============================")
print("12. Suggested Report Sentences")
print("==============================")

best_model_row = model_comparison.sort_values(by="F1-score", ascending=False).iloc[0]

print("\nDataset:")
print(
    "The dataset contains {} records and {} variables. "
    "The target variable is Diabetes_binary, with two balanced classes.".format(
        df_clean.shape[0],
        df_clean.shape[1]
    )
)

print("\nClassification:")
print(
    "The best-performing classification model based on F1-score was {}, "
    "with accuracy = {:.3f}, precision = {:.3f}, recall = {:.3f}, "
    "F1-score = {:.3f}, and ROC-AUC = {:.3f}.".format(
        best_model_row["Model"],
        best_model_row["Accuracy"],
        best_model_row["Precision"],
        best_model_row["Recall"],
        best_model_row["F1-score"],
        best_model_row["ROC-AUC"]
    )
)

print("\nClustering:")
print(
    "K-Means clustering with k = {} identified patient groups with different health profiles. "
    "Cluster {} showed the highest diabetes risk rate ({:.3f}), while Cluster {} showed "
    "the lowest diabetes risk rate ({:.3f}).".format(
        final_k,
        highest_risk_cluster,
        highest_risk_rate,
        lowest_risk_cluster,
        lowest_risk_rate
    )
)

print("\nClinical Value:")
print(
    "The results suggest that AI-based classification and clustering can support diabetes risk screening, "
    "patient segmentation, and healthcare resource allocation."
)

print("\nAll figures are saved in the 'figures' folder.")
print("All output tables are saved in the 'outputs' folder.")
print("\nScript finished successfully.")