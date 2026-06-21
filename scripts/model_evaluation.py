"""
model_evaluation.py
===================
Loads the trained forecast model and runs a detailed evaluation:
  - Residual analysis
  - Per-site accuracy breakdown
  - Worst-predicted sites
  - Forecast vs actual chart
  - Model performance summary report

Run after train_model.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC    = os.path.join(BASE, "data", "processed")
MODELS  = os.path.join(BASE, "models")
OUTPUTS = os.path.join(BASE, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

FEATURES = [
    "ma_7d","ma_14d","ma_30d",
    "lag_1d","lag_3d","lag_7d","lag_14d",
    "day_of_week","month","week_of_year","is_month_end","is_weekend","is_holiday",
    "grid_reliability_score","tank_capacity","tech_code",
]
TARGET = "daily_consumption_litres"

print("="*55)
print("  FORECAST MODEL EVALUATION")
print("="*55)

# Load model
model_path = f"{MODELS}/forecast_model.pkl"
if not os.path.exists(model_path):
    print("❌ Model not found. Run train_model.py first.")
    exit()

with open(model_path,"rb") as f:
    model = pickle.load(f)

df = pd.read_csv(f"{PROC}/features.csv", parse_dates=["date"])
X  = df[FEATURES]; y = df[TARGET]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
test_df = df.loc[X_test.index].copy()

preds              = model.predict(X_test)
test_df["pred"]    = preds
test_df["residual"]= test_df[TARGET] - test_df["pred"]
test_df["abs_err"] = test_df["residual"].abs()
test_df["pct_err"] = (test_df["abs_err"] / test_df[TARGET] * 100).round(2)

mae  = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds) ** 0.5
r2   = r2_score(y_test, preds)
mape = test_df["pct_err"].mean()

print(f"\nOverall Performance:")
print(f"  MAE:  {mae:.2f} litres")
print(f"  RMSE: {rmse:.2f} litres")
print(f"  R²:   {r2:.3f}")
print(f"  MAPE: {mape:.2f}%")

# Per-site accuracy
print("\nPer-Site MAE (Worst 10 Sites):")
site_acc = test_df.groupby("site_id").agg(
    mae=("abs_err","mean"), mape=("pct_err","mean"), samples=("site_id","count")
).round(2).sort_values("mae", ascending=False)
print(site_acc.head(10).to_string())
site_acc.to_csv(f"{OUTPUTS}/per_site_accuracy.csv")

# Charts
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Forecast Model Evaluation", fontsize=14, fontweight="bold")

# Actual vs Predicted
sample = test_df.sample(300, random_state=42).sort_values("date")
axes[0,0].scatter(sample[TARGET], sample["pred"], alpha=0.4, color="steelblue", s=10)
mn,mx = sample[TARGET].min(), sample[TARGET].max()
axes[0,0].plot([mn,mx],[mn,mx],"r--",alpha=0.7)
axes[0,0].set_xlabel("Actual (litres)"); axes[0,0].set_ylabel("Predicted (litres)")
axes[0,0].set_title("Actual vs Predicted")

# Residual distribution
axes[0,1].hist(test_df["residual"], bins=40, color="steelblue", edgecolor="white", alpha=0.8)
axes[0,1].axvline(0, color="red", linestyle="--")
axes[0,1].set_title("Residual Distribution")
axes[0,1].set_xlabel("Residual (litres)")

# Time series (one site)
site_sample = test_df[test_df["site_id"]==test_df["site_id"].iloc[0]].sort_values("date")
if len(site_sample) > 5:
    axes[1,0].plot(site_sample["date"], site_sample[TARGET], label="Actual", color="steelblue")
    axes[1,0].plot(site_sample["date"], site_sample["pred"], label="Forecast", color="orange", linestyle="--")
    axes[1,0].set_title(f"Forecast vs Actual — {site_sample['site_id'].iloc[0]}")
    axes[1,0].legend(); axes[1,0].tick_params(axis="x", rotation=45)

# Site MAE distribution
axes[1,1].hist(site_acc["mae"], bins=20, color="steelblue", edgecolor="white", alpha=0.8)
axes[1,1].axvline(site_acc["mae"].median(), color="orange", linestyle="--", label=f"Median MAE: {site_acc['mae'].median():.1f}L")
axes[1,1].set_title("Distribution of Per-Site MAE")
axes[1,1].set_xlabel("MAE (litres)")
axes[1,1].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUTS}/model_evaluation_charts.png", dpi=150, bbox_inches="tight")
print(f"\n✅ Evaluation complete. Charts saved.")
