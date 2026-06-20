"""feature_engineering.py — Rolling averages, lags, calendar features."""
import pandas as pd, os

BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW=os.path.join(BASE,"data","raw")
PROC=os.path.join(BASE,"data","processed"); os.makedirs(PROC,exist_ok=True)

df=pd.read_csv(f"{RAW}/fuel_readings.csv",parse_dates=["date"])
df=df.sort_values(["site_id","date"]).reset_index(drop=True)

for w in [7,14,30]:
    df[f"ma_{w}d"]=df.groupby("site_id")["daily_consumption_litres"].transform(lambda x:x.rolling(w,min_periods=1).mean())
for lag in [1,3,7,14]:
    df[f"lag_{lag}d"]=df.groupby("site_id")["daily_consumption_litres"].shift(lag)

df["day_of_week"]=df["date"].dt.dayofweek
df["month"]=df["date"].dt.month
df["week_of_year"]=df["date"].dt.isocalendar().week.astype(int)
df["is_month_end"]=df["date"].dt.is_month_end.astype(int)
df["is_weekend"]=(df["day_of_week"]>=5).astype(int)
df["tech_code"]=df["technology"].map({"2G":0,"3G":1,"4G":2})

df_clean=df.dropna()
df_clean.to_csv(f"{PROC}/features.csv",index=False)
print(f"Features saved: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} cols")
