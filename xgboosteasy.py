import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# ---------------------------
# Data loading
# ---------------------------
def read_excel(path):
    df = pd.read_excel(path, engine="openpyxl")

    # Fix EU decimal commas
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
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df

# ---------------------------
# Feature engineering
# ---------------------------
def make_features(df):
    # Time features
    step = (df["Date"].dt.hour * 60 + df["Date"].dt.minute) // 10
    df["mod"] = step.astype(int)
    df["mod_sin"] = np.sin(2 * np.pi * df["mod"] / 144)
    df["mod_cos"] = np.cos(2 * np.pi * df["mod"] / 144)

    dow = df["Date"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Lags and rolling
    for lag in range(1, 19):
        df[f"Energy_lag{lag}"] = df["Energy (kWh)"].shift(lag)
    for w in [6, 12, 18]:
        df[f"Energy_roll{w}"] = df["Energy (kWh)"].rolling(w).mean()

    # Speed + Direction lags
    if "Speed (m/s)" in df.columns:
        for lag in range(1, 13):
            df[f"Speed_lag{lag}"] = df["Speed (m/s)"].shift(lag)
    if "Direction (º)" in df.columns:
        for lag in range(1, 13):
            df[f"Dir_lag{lag}"] = df["Direction (º)"].shift(lag)

    feature_cols = [c for c in df.columns if c not in ["Date", "Energy (kWh)"]]
    df = df.dropna(subset=feature_cols + ["Energy (kWh)"])
    return df, feature_cols

# ---------------------------
# Train & Predict
# ---------------------------
def train_model(train_df):
    df_feat, feat_cols = make_features(train_df)
    X = df_feat[feat_cols].values
    y = df_feat["Energy (kWh)"].values

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X, y)
    return model, feat_cols

def predict_next_day(model, hist_df, day_df, feat_cols):
    buffer = pd.Timedelta(hours=24)
    context = hist_df[hist_df["Date"] >= day_df["Date"].min() - buffer]
    combo = pd.concat([context, day_df], ignore_index=True).sort_values("Date")

    combo_feat, _ = make_features(combo)
    combo_feat = combo_feat.dropna(subset=feat_cols)

    y_pred = model.predict(combo_feat[feat_cols].values)
    out = combo_feat[["Date"]].copy()
    out["predicted_energy"] = y_pred

    mask = (out["Date"] >= day_df["Date"].min()) & (out["Date"] <= day_df["Date"].max())
    return out[mask].reset_index(drop=True)

# ---------------------------
# Evaluation
# ---------------------------
def evaluate(pred_df, actual_df):
    df = pd.merge(pred_df, actual_df[["Date", "Energy (kWh)"]], on="Date", how="inner")
    y_true = df["Energy (kWh)"].values
    y_pred = df["predicted_energy"].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100

    df["pct_diff(%)"] = (y_pred - y_true) / df["Energy (kWh)"].clip(lower=1e-6) * 100
    return df, {"MAE": mae, "RMSE": rmse, "MAPE(%)": mape}

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    train_file = "Sotavento_3month.xlsx"
    test_file = "17march.xlsx"

    train_df = read_excel(train_file)

    # Train on Jan–Mar 16
    cutoff = pd.Timestamp(2020, 3, 16, 23, 50)
    train_cut = train_df[train_df["Date"] <= cutoff].copy()

    model, feat_cols = train_model(train_cut)

    # Create/Load test set (Mar 17)
    if os.path.exists(test_file):
        test_df = read_excel(test_file)
    else:
        test_df = pd.DataFrame({"Date": pd.date_range("2020-03-17 00:00", "2020-03-17 23:50", freq="10min")})

    preds = predict_next_day(model, train_cut, test_df, feat_cols)

    if os.path.exists(test_file):
        actual_df = read_excel(test_file)
        eval_df, metrics = evaluate(preds, actual_df)
        print("📊 Evaluation:", metrics)

        eval_df.to_excel("predictions_vs_actual_2020-03-17.xlsx", index=False)
        print("✅ Saved predictions → predictions_vs_actual_2020-03-17.xlsx")

        plt.figure(figsize=(12, 4))
        plt.plot(eval_df["Date"], eval_df["Energy (kWh)"], label="Actual")
        plt.plot(eval_df["Date"], eval_df["predicted_energy"], label="Predicted")
        plt.title("Energy (kWh): Actual vs Predicted — 2020-03-17")
        plt.legend()
        plt.show()
