
import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ---------------------------
# Read Excel and clean numeric columns
# ---------------------------
def read_excel(path):
    df = pd.read_excel(path, engine="openpyxl")
    for col in df.columns:
        if col == "Date":
            continue
        df[col] = (
            df[col].astype(str)
            .str.replace(",", ".", regex=False)
            .replace("-", np.nan)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df

# ---------------------------
# Feature engineering (selected features)
# ---------------------------
def make_selected_features(df):
    df = df.copy()
    step = (df["Date"].dt.hour * 60 + df["Date"].dt.minute) // 10
    df["mod_sin"] = np.sin(2 * np.pi * step / 144)
    df["mod_cos"] = np.cos(2 * np.pi * step / 144)
    df["Speed_cubed"] = df["Speed (m/s)"] ** 3
    for lag in [6, 7, 8]:
        df[f"Energy_lag{lag}"] = df["Energy (kWh)"].shift(lag)
    return df

# ---------------------------
# Train model
# ---------------------------
def train_model(train_df, feature_cols):
    df_feat = make_selected_features(train_df)
    # Drop rows with NaNs for lag features
    df_feat = df_feat.dropna(subset=feature_cols + ["Energy (kWh)"])
    X = df_feat[feature_cols].values
    y = df_feat["Energy (kWh)"].values
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )
    model.fit(X, y)
    return model

# ---------------------------
# Predict
# ---------------------------
def predict(model, df, feature_cols):
    df_feat = make_selected_features(df)
    df_feat = df_feat.dropna(subset=feature_cols)
    df_feat["predicted_energy"] = model.predict(df_feat[feature_cols].values)
    return df_feat

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    train_file = "Sotavento_3month.xlsx"
    test_file = "17march.xlsx"

    # Load raw data
    train_df = read_excel(train_file)
    test_df = read_excel(test_file)

    # Feature columns
    feature_cols = ["mod_sin", "mod_cos", "Speed_cubed", "Energy_lag6", "Energy_lag7", "Energy_lag8"]

    # ---------------------------
    # Create feature-engineered Excels
    # ---------------------------
    train_feat = make_selected_features(train_df)
    test_feat = make_selected_features(test_df)

    train_feat.to_excel("train_features.xlsx", index=False)
    test_feat.to_excel("test_features.xlsx", index=False)
    print("✅ Feature-engineered Excels saved: train_features.xlsx, test_features.xlsx")

    # ---------------------------
    # Train model
    # ---------------------------
    model = train_model(train_df, feature_cols)

    # ---------------------------
    # Predict
    # ---------------------------
    preds = predict(model, test_df, feature_cols)
    preds.to_excel("predictions_selected_features.xlsx", index=False)
    print("✅ Predictions saved → predictions_selected_features.xlsx")

    # Optional: plot
    if "Energy (kWh)" in test_df.columns:
        plt.figure(figsize=(12, 4))
        plt.plot(test_df["Date"], test_df["Energy (kWh)"], label="Actual")
        plt.plot(preds["Date"], preds["predicted_energy"], label="Predicted")
        plt.title("Energy (kWh): Actual vs Predicted")
        plt.legend()
        plt.show()
