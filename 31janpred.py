import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------- CONFIG ----------
DATA_FILE  = "31jan25training.xlsx"   # full Jan dataset
OUT_FILE   = "sliding_predictions_jan2025.xlsx"
TARGET_COL = "Energy (kWh)"
# ----------------------------

# ----------- Load + Clean ----------
def clean_dataframe(path):
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in df.columns:
        if col != "Date":
            df[col] = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[TARGET_COL])
    return df

# ----------- Main ----------
if __name__ == "__main__":
    df = clean_dataframe(DATA_FILE)

    # Store all predictions
    from sklearn.metrics import mean_absolute_error, mean_squared_error

# Store all predictions
all_results = []
all_y_true, all_y_pred = [], []   # 👉 keep cumulative actuals & predictions

# Loop through sliding windows: predict Jan 22 to Jan 31
for day in range(22, 32):  # 22 .. 31 inclusive
    train_start = f"2025-01-{day-21:02d}"
    train_end   = f"2025-01-{day-1:02d}"
    test_day    = f"2025-01-{day:02d}"

    # Train and test splits
    train_df = df[(df["Date"] >= train_start) & (df["Date"] <= train_end)].copy()
    test_df  = df[df["Date"].dt.strftime("%Y-%m-%d") == test_day].copy()

    # Features
    feature_cols = [c for c in train_df.columns if c not in ["Date", TARGET_COL]]
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
    X_test,  y_test  = test_df[feature_cols],  test_df[TARGET_COL]

    # Train model
    model = XGBRegressor(
     n_estimators=1200,        # more trees
     learning_rate=0.03,       # smaller learning rate for stability
     max_depth=8,              # deeper trees to capture complexity
      subsample=0.9,            # keep more data per tree
     colsample_bytree=0.9,     # use more features per tree
     reg_lambda=1.5,           # L2 regularization
     reg_alpha=0.5,            # L1 regularization
     min_child_weight=2,       # avoid overfitting small leaf splits
     gamma=0.2,                # require a minimum gain to split
     objective="reg:squarederror",
     random_state=42,
     n_jobs=4
    )

    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Store results
    result = test_df[["Date", TARGET_COL]].copy()
    result["Predicted_Energy"] = y_pred
    result["Abs_Error"] = np.abs(y_test - y_pred)
    result["Pct_Error(%)"] = (y_pred - y_test) / np.clip(y_test, 1e-6, None) * 100

    # Print daily predictions
    print("\nPredictions for", test_day)
    print(result)

    # Update cumulative lists
    all_y_true.extend(y_test.tolist())
    all_y_pred.extend(y_pred.tolist())

    # --- Calculate cumulative metrics ---
    mae = mean_absolute_error(all_y_true, all_y_pred)
    rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))

    mask = np.array(all_y_true) != 0
    mape = (np.mean(np.abs((np.array(all_y_true)[mask] - np.array(all_y_pred)[mask]) / np.array(all_y_true)[mask])) * 100)

    print(f"📊 Cumulative metrics up to {test_day}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")

    all_results.append(result)

# Concatenate all results and save
final_out = pd.concat(all_results, ignore_index=True)
final_out.to_excel(OUT_FILE, index=False)
print(f"\n📁 Saved all sliding predictions to {OUT_FILE}")

