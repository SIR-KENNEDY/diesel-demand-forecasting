"""
forecast_next7days.py
=====================
Uses the trained model to generate a 7-day forward forecast
for all sites and outputs a procurement planning table.

Run after train_model.py
"""
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta

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

model_path = f"{MODELS}/forecast_model.pkl"
if not os.path.exists(model_path):
    print("❌ Model not found. Run train_model.py first."); exit()

with open(model_path,"rb") as f:
    model = pickle.load(f)

df = pd.read_csv(f"{PROC}/features.csv", parse_dates=["date"])

# Use last known record per site as baseline
latest = df.sort_values("date").groupby("site_id").last().reset_index()

print("="*55)
print("  7-DAY FORWARD FORECAST")
print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("="*55)

rows = []
for _, site in latest.iterrows():
    for d in range(1, 8):
        future_date = site["date"] + timedelta(days=d)
        row = site.copy()
        row["date"]         = future_date
        row["day_of_week"]  = future_date.dayofweek
        row["month"]        = future_date.month
        row["week_of_year"] = future_date.isocalendar()[1]
        row["is_month_end"] = int(future_date.day >= 28)
        row["is_weekend"]   = int(future_date.dayofweek >= 5)
        row["is_holiday"]   = 0
        row["forecast_day"] = d
        rows.append(row)

forecast_df = pd.DataFrame(rows)
forecast_df["predicted_consumption"] = model.predict(forecast_df[FEATURES]).round(1)

# 7-day totals per site
summary = forecast_df.groupby(["site_id","region","tank_capacity"]).agg(
    forecast_7d_litres=("predicted_consumption","sum"),
    avg_daily_litres=("predicted_consumption","mean"),
).reset_index()
summary["procurement_recommendation"] = (summary["forecast_7d_litres"] * 1.10).round(0)  # +10% buffer
summary["urgency"] = pd.cut(summary["avg_daily_litres"],
    bins=[0,100,180,300,9999], labels=["Low","Medium","High","Critical"])
summary = summary.sort_values("forecast_7d_litres", ascending=False)

print("\nTop 15 Sites by 7-Day Forecast Demand:")
print(summary.head(15)[["site_id","region","forecast_7d_litres","procurement_recommendation","urgency"]].to_string(index=False))

summary.to_csv(f"{OUTPUTS}/7day_procurement_forecast.csv", index=False)
forecast_df[["site_id","date","forecast_day","predicted_consumption"]].to_csv(
    f"{OUTPUTS}/7day_daily_forecast.csv", index=False)
print(f"\n✅ Forecast saved to outputs/")
