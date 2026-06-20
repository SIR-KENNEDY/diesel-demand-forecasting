"""
train_model.py — Trains XGBoost (or Random Forest) demand forecast model.
Evaluates performance vs. baseline and saves feature importance chart.
"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle, os

BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC=os.path.join(BASE,"data","processed")
MODELS=os.path.join(BASE,"models"); os.makedirs(MODELS,exist_ok=True)
OUTPUTS=os.path.join(BASE,"outputs"); os.makedirs(OUTPUTS,exist_ok=True)

try:
    from xgboost import XGBRegressor; USE_XGB=True
except ImportError:
    print("XGBoost not found — using Random Forest"); USE_XGB=False

df=pd.read_csv(f"{PROC}/features.csv")
FEATURES=["ma_7d","ma_14d","ma_30d","lag_1d","lag_3d","lag_7d","lag_14d",
          "day_of_week","month","week_of_year","is_month_end","is_weekend","is_holiday",
          "grid_reliability_score","tank_capacity","tech_code"]
TARGET="daily_consumption_litres"
X=df[FEATURES]; y=df[TARGET]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Baseline
bl_preds=df.loc[X_test.index,"ma_7d"]
print(f"Baseline MAE: {mean_absolute_error(y_test,bl_preds):.2f}L  R²: {r2_score(y_test,bl_preds):.3f}")

model=(XGBRegressor(n_estimators=300,max_depth=5,learning_rate=0.05,random_state=42,verbosity=0)
       if USE_XGB else RandomForestRegressor(n_estimators=200,max_depth=8,random_state=42,n_jobs=-1))
model.fit(X_train,y_train)
preds=model.predict(X_test)

print(f"\n{'XGBoost' if USE_XGB else 'Random Forest'} Results:")
print(f"  MAE:  {mean_absolute_error(y_test,preds):.2f} litres")
print(f"  RMSE: {mean_squared_error(y_test,preds)**0.5:.2f} litres")
print(f"  R²:   {r2_score(y_test,preds):.3f}")

with open(f"{MODELS}/forecast_model.pkl","wb") as f: pickle.dump(model,f)
print(f"\nModel saved.")

if hasattr(model,"feature_importances_"):
    fi=pd.Series(model.feature_importances_,index=FEATURES).sort_values()
    fig,ax=plt.subplots(figsize=(8,6))
    fi.plot(kind="barh",ax=ax,color="steelblue")
    ax.set_title("Feature Importance"); ax.set_xlabel("Score")
    plt.tight_layout()
    plt.savefig(f"{OUTPUTS}/feature_importance.png",dpi=150,bbox_inches="tight")
    print("Feature importance chart saved.")
