import pandas as pd
import numpy as np

# --- Step 1: Read Excel ---
df = pd.read_excel("Sotavento_3month.xlsx")  # replace with your filename

# --- Step 2: Convert Date to datetime ---
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# --- Step 3: Clean numeric columns ---
df["Speed (m/s)"] = pd.to_numeric(df["Speed (m/s)"].astype(str).str.replace(",", "."), errors="coerce")
df["Energy (kWh)"] = pd.to_numeric(df["Energy (kWh)"].astype(str).str.replace(",", "."), errors="coerce")

# --- Step 4: Feature generation ---
# Time-of-day features
step = (df["Date"].dt.hour * 60 + df["Date"].dt.minute) // 10
df["mod_sin"] = np.sin(2 * np.pi * step / 144)
df["mod_cos"] = np.cos(2 * np.pi * step / 144)

# Speed cubed
df["Speed_cubed"] = df["Speed (m/s)"] ** 3

# Energy lags
for lag in [6, 7, 8]:
    df[f"Energy_lag{lag}"] = df["Energy (kWh)"].shift(lag)

# --- Step 5: Save to a new Excel file ---
df.to_excel("features_with_shift.xlsx", index=False)

print("Excel file 'features_with_shift.xlsx' created with all rows and lag shifts visible.")
