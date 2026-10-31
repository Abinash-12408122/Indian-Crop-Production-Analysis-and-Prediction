import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.cluster import KMeans

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create output folder
os.makedirs("outputs", exist_ok =True)

# Load the dataset
df = pd.read_csv("crop_production.csv")

print("===============================")
print("INDIAN CROP PRODUCTION ANALYSIS")
print("===============================")

# =======================================
# STEP 1: ALL THE INFORMATION OF DATASET
# =======================================

# Printing the shape & size information of the dataset
print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# To check numeric data from the dataset
for col in ["Crop_Year", "Area", "Production"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nYEAR RANGE:")
print("Minimum year:", df["Crop_Year"].min())
print("Maximum year:", df["Crop_Year"].max())
print("Number of different years:", df["Crop_Year"].nunique())

print("\nUNIQUE SEASONS:")
print("\nUnique seasons:", df["Season"].nunique())
print(df["Season"].value_counts(dropna=False))

print("\nUNIQUE CROPS:")
print("\nUnique crops:", df["Crop"].nunique())
print(df["Crop"].value_counts(dropna=False))

print("\nUNIQUE STATES:")
print("\nUnique states:", df["State_Name"].nunique())
print(df["State_Name"].value_counts(dropna=False))

# Save initial inspection
df.head(100).to_csv(
    "outputs/initial_data_sample.csv",
    index=False
)

print("\nInitial inspection completed.")
print("Sample saved in outputs/initial_data_sample.csv")


# ==============================================
# STEP 2: DATA CLEANING & FEATURE ENGINEERING
# ==============================================

print("==============================================")
print("STEP 2: DATA CLEANING & FEATURE ENGINEERING")
print("==============================================")

# 1. Create a copy of the original dataset
df_clean = df.copy()

# 2. Convert numeric columns into numeric data types
numeric_columns = ["Crop_Year", "Area", "Production"]

for col in numeric_columns:
    df_clean[col] = pd.to_numeric(
        df_clean[col],
        errors="coerce"
    )

# 3. Remove duplicate records
duplicate_count = df_clean.duplicated().sum()
df_clean = df_clean.drop_duplicates()

print("\nDuplicate records removed:", duplicate_count)

# 4. Check missing values
print("\nMissing values before cleaning:")
print(df_clean.isnull().sum())

# 5. Remove records with missing essential input information
df_clean = df_clean.dropna(
    subset=[
        "State_Name",
        "District_Name",
        "Crop_Year",
        "Season",
        "Crop",
        "Area"
    ]
)

# 6. Remove invalid area and negative production records
invalid_area = (df_clean["Area"] <= 0).sum()
negative_production = (df_clean["Production"] < 0).sum()

df_clean = df_clean[
    (df_clean["Area"] > 0) &
    (
        df_clean["Production"].isna() |
        (df_clean["Production"] >= 0)
    )
].copy()

print("\nInvalid area records removed:", invalid_area)
print("Negative production records removed:", negative_production)

# 7. Create a separate dataset for production analysis
df_analysis = df_clean.dropna(
    subset=["Production"]
).copy()

# 8. Display cleaning results
print("\nOriginal dataset shape:", df.shape)
print("Cleaned dataset shape:", df_clean.shape)
print("Production analysis dataset shape:", df_analysis.shape)

print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())

# 9. Save cleaned datasets
df_clean.to_csv(
    "outputs/crop_data_cleaned.csv",
    index=False
)

df_analysis.to_csv(
    "outputs/crop_data_analysis.csv",
    index=False
)

print("\nData cleaning completed successfully.")



# Work with a copy of the cleaned analysis dataset
df_analysis = df_analysis.copy()

# 1. Calculate Yield
df_analysis["Yield"] = (
    df_analysis["Production"] / df_analysis["Area"]
)

# 2. Calculate total production within each group
group_columns = [
    "State_Name",
    "District_Name",
    "Crop_Year",
    "Season"
]

group_production = (
    df_analysis.groupby(group_columns)["Production"]
    .transform("sum")
)

# Calculate crop production contribution percentage
df_analysis["Production_Percentage"] = np.where(
    group_production > 0,
    (df_analysis["Production"] / group_production) * 100,
    0
)

# 3. Calculate total cultivated area within each group
group_area = (
    df_analysis.groupby(group_columns)["Area"]
    .transform("sum")
)

# Calculate crop area contribution percentage
df_analysis["Area_Percentage"] = (
    df_analysis["Area"] / group_area
) * 100

# 4. Apply logarithmic transformation to Yield
df_analysis["Production_Per_Area_Log"] = np.log1p(
    df_analysis["Yield"]
)

# 5. Categorize cultivated area into three groups
df_analysis["Crop_Area_Category"] = pd.qcut(
    df_analysis["Area"].rank(method="first"),
    q=3,
    labels=["Small", "Medium", "Large"]
)

print("\nNow Five new features created successfully.")

# Save feature-engineered dataset
df_analysis.to_csv(
    "outputs/crop_data_feature_engineered.csv",
    index=False
)


# ===================================
# STEP 3: UPDATED DATASET INSPECTION
# ===================================

print("===================================")
print("STEP 3: UPDATED DATASET INSPECTION")
print("===================================")

# Display the first five rows
print("\nFirst 5 rows of the updated dataset:")
print(df_analysis.head())

# Display all column names
print("\nUpdated column names:")
print(df_analysis.columns.tolist())

# Display dataset shape
print("\nUpdated dataset shape:")
print(df_analysis.shape)

# Display data types and non-null values
print("\nUpdated dataset information:")
df_analysis.info()

# Check missing values
print("\nMissing values:")
print(df_analysis.isnull().sum())

# Display statistical summary
print("\nStatistical summary:")
print(df_analysis.describe())

# Display crop area categories
print("\nCrop area category counts:")
print(df_analysis["Crop_Area_Category"].value_counts())

# Save updated first five rows
df_analysis.head().to_csv(
    "outputs/updated_data_sample.csv",
    index=False
)

print("\nNow Updated dataset inspection completed.")


# =======================================
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# =======================================

print("===================================")
print("STEP 4: EXPLORATORY DATA ANALYSIS")
print("===================================")

# 1. Year-wise cultivated area trend
yearly_area = (
    df_analysis.groupby("Crop_Year")["Area"]
    .sum()
    .sort_index()
)

plt.figure(figsize=(12, 6))
plt.plot(
    yearly_area.index,
    yearly_area.values,
    marker="o",
    linewidth=2
)

plt.title("Year-wise Cultivated Area Trend (1997-2022)")
plt.xlabel("Crop Year")
plt.ylabel("Total Cultivated Area")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/yearly_area_trend.png", dpi=300)
plt.show()


# 2. Top 10 crops by number of records
top_crops_count = (
    df_analysis["Crop"]
    .value_counts()
    .head(10)
    .sort_values()
)

plt.figure(figsize=(11, 7))
plt.barh(
    top_crops_count.index,
    top_crops_count.values
)

plt.title("Top 10 Crops by Number of Records")
plt.xlabel("Number of Records")
plt.ylabel("Crop")
plt.tight_layout()
plt.savefig("outputs/top_10_crops_count.png", dpi=300)
plt.show()


# 3. Top 10 states by cultivated area
top_states_area = (
    df_analysis.groupby("State_Name")["Area"]
    .sum()
    .nlargest(10)
    .sort_values()
)

plt.figure(figsize=(11, 7))
plt.barh(
    top_states_area.index,
    top_states_area.values
)

plt.title("Top 10 States by Cultivated Area")
plt.xlabel("Total Cultivated Area")
plt.ylabel("State")
plt.tight_layout()
plt.savefig("outputs/top_10_states_area.png", dpi=300)
plt.show()


# 4. Cultivated area by season
season_area = (
    df_analysis.groupby("Season")["Area"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))
plt.bar(
    season_area.index,
    season_area.values
)

plt.title("Cultivated Area by Season")
plt.xlabel("Season")
plt.ylabel("Total Cultivated Area")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig("outputs/season_area_comparison.png", dpi=300)
plt.show()


# 5. Distribution of crop yield
plt.figure(figsize=(11, 6))
plt.hist(
    df_analysis["Yield"],
    bins=60,
    log=True
)

plt.title("Distribution of Crop Yield")
plt.xlabel("Yield (Production / Area)")
plt.ylabel("Frequency (Log Scale)")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("outputs/yield_distribution.png", dpi=300)
plt.show()


# 6. Crop area category distribution
category_counts = (
    df_analysis["Crop_Area_Category"]
    .value_counts()
    .reindex(["Small", "Medium", "Large"])
)

plt.figure(figsize=(8, 6))
plt.bar(
    category_counts.index.astype(str),
    category_counts.values
)

plt.title("Distribution of Crop Area Categories")
plt.xlabel("Crop Area Category")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig("outputs/crop_area_categories.png", dpi=300)
plt.show()


# 7. Save EDA summary tables
yearly_area.to_csv(
    "outputs/yearly_area_summary.csv",
    header=["Total_Area"]
)

top_crops_count.to_csv(
    "outputs/top_crops_count_summary.csv",
    header=["Number_of_Records"]
)

top_states_area.to_csv(
    "outputs/top_states_area_summary.csv",
    header=["Total_Area"]
)

season_area.to_csv(
    "outputs/season_area_summary.csv",
    header=["Total_Area"]
)

category_counts.to_csv(
    "outputs/crop_area_category_summary.csv",
    header=["Number_of_Records"]
)


# 8. Display summary information
print("\nYear-wise cultivated area:")
print(yearly_area)

print("\nTop 10 crops by number of records:")
print(top_crops_count.sort_values(ascending=False))

print("\nTop 10 states by cultivated area:")
print(top_states_area.sort_values(ascending=False))

print("\nCultivated area by season:")
print(season_area)

print("\nCrop area category distribution:")
print(category_counts)

print("\nEDA completed successfully.")
print("All charts and summary tables saved in outputs folder.")



# ===========================================
# STEP 5: MACHINE LEARNING MODEL DEVELOPMENT
# ===========================================

print("===========================================")
print("STEP 5: MACHINE LEARNING MODEL DEVELOPMENT")
print("===========================================")

# Select a sample for model training
ml_data = df_analysis.sample(
    n=min(30000, len(df_analysis)),
    random_state=42
).copy()

# Encode categorical columns
ml_data["State_Code"] = LabelEncoder().fit_transform(
    ml_data["State_Name"]
)

ml_data["District_Code"] = LabelEncoder().fit_transform(
    ml_data["District_Name"]
)

ml_data["Season_Code"] = LabelEncoder().fit_transform(
    ml_data["Season"]
)

ml_data["Crop_Code"] = LabelEncoder().fit_transform(
    ml_data["Crop"]
)

print("Categorical encoding completed.")


# =======================
# STEP 6: MODEL CREATION
# =======================

# 1. Simple Linear Regression
#(here I took input as Area and target is Production)

X = ml_data[["Area"]]
y = ml_data["Production"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nSIMPLE LINEAR REGRESSION")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(9, 6))
plt.scatter(X_test["Area"], y_test, alpha=0.4)
plt.plot(X_test["Area"], y_pred, color="red")
plt.title("Simple Linear Regression")
plt.xlabel("Cultivated Area")
plt.ylabel("Production")
plt.tight_layout()
plt.show()

# 2. Multiple Linear Regression
# (here I took input as Area, year, state, district, season and crop)

X = ml_data[
    [
        "Area",
        "Crop_Year",
        "State_Code",
        "District_Code",
        "Season_Code",
        "Crop_Code"
    ]
]

y = ml_data["Production"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nMULTIPLE LINEAR REGRESSION")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 3. Polynomial Regression
# (here I took input as Area, Area² and Crop Year)

X = ml_data[["Area", "Crop_Year"]]

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

y = ml_data["Production"]

X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nPOLYNOMIAL REGRESSION")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 4. KNN Regression
# (here; KNN predicts production based on nearby records.
#  We must standardize the inputs because KNN uses distances.)

X = ml_data[
    [
        "Area",
        "Crop_Year",
        "State_Code",
        "District_Code",
        "Season_Code",
        "Crop_Code"
    ]
]

y = ml_data["Production"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nKNN REGRESSION")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 5. Decision Tree Regression

X = ml_data[
    [
        "Area",
        "Crop_Year",
        "State_Code",
        "District_Code",
        "Season_Code",
        "Crop_Code"
    ]
]

y = ml_data["Production"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nDECISION TREE REGRESSION")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(18, 9))

plot_tree(
    model,
    filled=True,
    feature_names=X.columns,
    rounded=True,
    max_depth=3
)

plt.title("Decision Tree Regression")
plt.show()

# 6. Random Forest Regression

X = ml_data[
    [
        "Area",
        "Crop_Year",
        "State_Code",
        "District_Code",
        "Season_Code",
        "Crop_Code"
    ]
]

y = ml_data["Production"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nRANDOM FOREST REGRESSION")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(10, 6))

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values()

importance.plot(kind="barh")

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# 7. Support Vector Regression (SVR)[Family of SVM]
# (We'll use LinearSVRbecause the dataset is relatively large and
#  a standardRBF SVR can take too long.)

X = ml_data[
    [
        "Area",
        "Crop_Year",
        "State_Code",
        "District_Code",
        "Season_Code",
        "Crop_Code"
    ]
]

y = ml_data["Production"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearSVR(
    C=1.0,
    max_iter=3000,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nSUPPORT VECTOR REGRESSION")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 8. K-Means Clustering
# (K-Means is an unsupervised learning algorithm. It groups records based on
#   similarity;it does not predict production.)

X = ml_data[["Area", "Yield"]].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

ml_data["Cluster"] = model.fit_predict(X_scaled)

print("\nK-MEANS CLUSTERING")
print(ml_data["Cluster"].value_counts())

plt.figure(figsize=(10, 6))

plt.scatter(
    ml_data["Area"],
    ml_data["Yield"],
    c=ml_data["Cluster"],
    cmap="viridis",
    alpha=0.5
)

plt.title("K-Means Crop Clustering")
plt.xlabel("Cultivated Area")
plt.ylabel("Yield")
plt.tight_layout()
plt.show()
