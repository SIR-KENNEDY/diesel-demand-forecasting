# 📓 Diesel Demand Forecasting — Guide

## Why Forecasting Matters
Reactive diesel procurement creates two costly problems:
1. **Under-stock** → site fuel-out → network downtime → SLA breach
2. **Over-stock** → capital tied up → spoilage risk → wasted budget

A 7-day demand forecast lets procurement teams order the right amount,
at the right time, for the right sites.

---

## Execution Order
```bash
python scripts/generate_data.py         # 18 months daily readings, 50 sites
python scripts/feature_engineering.py   # Rolling averages, lags, calendar features
python scripts/train_model.py           # Train & evaluate XGBoost model
python scripts/model_evaluation.py      # Detailed residual analysis + charts
python scripts/forecast_next7days.py    # Generate procurement planning table
```

---

## Feature Engineering Explained

| Feature | Type | Why It Matters |
|---------|------|---------------|
| `ma_7d`, `ma_14d`, `ma_30d` | Rolling average | Captures consumption trends |
| `lag_1d`, `lag_3d`, `lag_7d` | Lag features | Yesterday's usage predicts today's |
| `day_of_week` | Calendar | Weekend vs weekday patterns |
| `is_holiday` | Calendar | Public holidays = more network traffic |
| `grid_reliability_score` | Site metadata | Low grid = more generator = more diesel |
| `tech_code` | Site metadata | 4G sites consume differently than 2G |

---

## Model Performance
| Model | MAE | R² | Notes |
|-------|-----|-----|-------|
| Baseline (7-day MA) | ~48L | 0.71 | Simple rolling average |
| Random Forest | ~21L | 0.93 | Good, fast to train |
| **XGBoost** | **~19L** | **0.95** | Best performance |

MAE of 19 litres means on average the model is within **19 litres** of actual daily consumption.
Across a 50-site portfolio, this translates to significantly more accurate procurement orders.

---

## Procurement Output
The `forecast_next7days.py` script generates:
- **Daily forecast per site** — how many litres predicted each day
- **7-day total per site** — total expected consumption
- **Procurement recommendation** — 7-day total + 10% safety buffer
- **Urgency rating** — Low / Medium / High / Critical based on average daily demand
