# 📈 Diesel Demand Forecasting — Telecom Sites (ML)

![Python](https://img.shields.io/badge/Python-3.10-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-Forecasting-orange) ![Scikit--learn](https://img.shields.io/badge/Scikit--learn-ML-red)

## Overview
Machine learning model predicting **7-day site-level diesel demand** across 50 telecom sites using rolling averages, lag features, calendar effects, and site metadata. Enables proactive procurement and eliminates reactive emergency orders.

## Models Evaluated
| Model | MAE | R² |
|-------|-----|-----|
| Baseline (7-day MA) | ~48L | 0.71 |
| Random Forest | ~21L | 0.93 |
| **XGBoost** | **~19L** | **0.95** |

## How to Run
```bash
pip install -r requirements.txt
python scripts/generate_data.py
python scripts/feature_engineering.py
python scripts/train_model.py
```

## Feature Engineering Highlights
- Rolling 7/14/30-day consumption averages
- Lag features at 1, 3, 7, 14 days
- Calendar: day of week, month, week of year, holiday flag
- Site: grid reliability score, technology type, tank capacity

## Skills Demonstrated
`Machine Learning` `XGBoost` `Feature Engineering` `Time-Series Forecasting` `Scikit-learn` `Model Evaluation` `Python`

---
*Kennedy Onuorah | [LinkedIn](https://www.linkedin.com/in/kennedy-onuorah-7a3793128)*
