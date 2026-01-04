"""
COMPLETE USED CAR PRICE PREDICTION SYSTEM (FIXED)
Author: Niteesh Pandey
Description: End-to-end ML solution (No Data Leakage, CV Added)
"""

# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-darkgrid")

print("="*60)
print("USED CAR PRICE PREDICTION SYSTEM (FIXED VERSION)")
print("="*60)

# =========================
# LOAD DATA
# =========================
def load_data():
    df = pd.read_csv("car_data.csv")
    print(f"Dataset Loaded: {df.shape}")
    return df

df = load_data()

# =========================
# EDA (NO CHANGES – SAFE)
# =========================
def basic_eda(data):
    print("\nBASIC STATS")
    print(data.describe())

    print("\nMissing Values")
    print(data.isnull().sum())

basic_eda(df)

# =========================
# FEATURE ENGINEERING (SAFE)
# =========================
def feature_engineering(data):
    df_fe = data.copy()

    # Vehicle Age (Dynamic year)
    current_year = datetime.now().year
    df_fe["Vehicle_Age"] = current_year - df_fe["Year"]

    # Brand
    df_fe["Brand"] = df_fe["Car_Name"].apply(lambda x: x.split()[0])

    # Brand Category (NO TARGET USAGE)
    luxury = ["fortuner", "innova", "camry", "corolla", "land"]
    mid = ["city", "verna", "i20", "creta", "elantra"]

    def brand_category(brand):
        brand = brand.lower()
        if any(b in brand for b in luxury):
            return "Luxury"
        elif any(b in brand for b in mid):
            return "Mid_Range"
        else:
            return "Budget"

    df_fe["Brand_Category"] = df_fe["Brand"].apply(brand_category)

    # Log transform (ONLY INPUT FEATURE)
    df_fe["Log_Kms"] = np.log1p(df_fe["Kms_Driven"])

    return df_fe

df = feature_engineering(df)

# =========================
# ML DATA PREP (NO LEAKAGE)
# =========================
def prepare_ml_data(data):
    features = [
        "Present_Price",
        "Kms_Driven",
        "Log_Kms",
        "Vehicle_Age",
        "Owner"
    ]

    categorical = ["Fuel_Type", "Seller_Type", "Transmission", "Brand_Category"]

    data_encoded = pd.get_dummies(
        data,
        columns=categorical,
        drop_first=True
    )

    encoded_features = [col for col in data_encoded.columns if col not in
                        ["Car_Name", "Selling_Price", "Year", "Brand"]]

    X = data_encoded[encoded_features]
    y = data_encoded["Selling_Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, encoded_features

X_train, X_test, y_train, y_test, feature_names = prepare_ml_data(df)

# =========================
# MODEL TRAINING + CV
# =========================
def train_models(X_train, X_test, y_train, y_test):
    results = {}

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    }

    for name, model in models.items():
        print(f"\nTraining {name}")

        if name == "Linear Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            cv_score = cross_val_score(
                model, X_train_scaled, y_train, cv=5, scoring="r2"
            ).mean()
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            cv_score = cross_val_score(
                model, X_train, y_train, cv=5, scoring="r2"
            ).mean()

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "CV_R2": cv_score
        }

        print(f"R2: {r2:.3f}")
        print(f"CV R2: {cv_score:.3f}")
        print(f"MAE: ₹{mae:.2f} Lakh")

    return results

results = train_models(X_train, X_test, y_train, y_test)

# =========================
# BUSINESS INSIGHTS (SAFE)
# =========================
def business_insights(data, results):
    best_model = max(results.items(), key=lambda x: x[1]["R2"])

    print("\nBUSINESS INSIGHTS")
    print("-"*50)
    print(f"Best Model: {best_model[0]}")
    print(f"Accuracy (R2): {best_model[1]['R2']:.2%}")

    avg_price = data["Selling_Price"].mean()
    print(f"Average Used Car Price: ₹{avg_price:.2f} Lakhs")

    fuel_price = data.groupby("Fuel_Type")["Selling_Price"].mean()
    print("\nFuel Type Impact")
    print(fuel_price)

business_insights(df, results)

# =========================
# FINAL SUMMARY
# =========================
print("\nPROJECT COMPLETED SUCCESSFULLY")
print("No Data Leakage | CV Applied | Interview Ready")
