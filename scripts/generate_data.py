"""generate_data.py — 18 months daily fuel readings for 50 sites."""
import pandas as pd, numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(55)
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW=os.path.join(BASE,"data","raw"); os.makedirs(RAW,exist_ok=True)
N_SITES=50; DAYS=548; START=datetime(2022,1,1)

sites=pd.DataFrame({
    "site_id":[f"SITE_{str(i).zfill(3)}" for i in range(1,N_SITES+1)],
    "technology":np.random.choice(["2G","3G","4G"],N_SITES,p=[0.2,0.3,0.5]),
    "grid_reliability_score":np.random.uniform(0.2,0.9,N_SITES).round(2),
    "tank_capacity":np.random.choice([2000,3000,5000],N_SITES),
})
records=[]
for _,s in sites.iterrows():
    base_cons=200*(1-s.grid_reliability_score*0.6)
    for d in range(DAYS):
        date=START+timedelta(days=d)
        is_holiday=int(date.weekday()==6 or np.random.random()<0.02)
        consumption=max(20, base_cons+20*np.sin(2*np.pi*d/365)+np.random.normal(0,15)+is_holiday*30)
        records.append({"site_id":s.site_id,"date":date.strftime("%Y-%m-%d"),
            "daily_consumption_litres":round(consumption,1),
            "grid_reliability_score":s.grid_reliability_score,
            "tank_capacity":s.tank_capacity,"technology":s.technology,"is_holiday":is_holiday})
pd.DataFrame(records).to_csv(f"{RAW}/fuel_readings.csv",index=False)
sites.to_csv(f"{RAW}/site_metadata.csv",index=False)
print(f"Generated {len(records):,} daily readings for {N_SITES} sites.")
