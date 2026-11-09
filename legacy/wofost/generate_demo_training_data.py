"""Create a DEMO-only simulated dataset for testing the ML pipeline.

This is NOT WOFOST output and must not be reported as scientific simulation data.
Use generate_simulated_dataset.py on a PC with PCSE + NASA POWER access for
actual WOFOST output.
"""
import numpy as np
import pandas as pd
from constants import STATE_DISTRICTS, CROPS, SEASONS, SOIL_TYPES

rng = np.random.default_rng(42)
rows=[]
for state,districts in STATE_DISTRICTS.items():
  for district in districts:
    for year in range(2015,2020):
      for crop in CROPS:
        for season in SEASONS:
          for soil in SOIL_TYPES:
            for irrigated in [False,True]:
              base={'wheat':3500,'rice':4200,'maize':3800}[crop]
              season_adj={'Kharif':0.95,'Rabi':1.05,'Zaid':0.9}[season]
              soil_adj={'Sandy':0.85,'Loam':1.0,'Clay':0.92}[soil]
              irr_adj=1.12 if irrigated else 0.9
              weather=rng.normal(1.0,0.10)
              twso=max(0,base*season_adj*soil_adj*irr_adj*weather)
              tagp=twso*rng.uniform(2.0,3.2)
              rows.append([state,district,year,crop,season,soil,irrigated,round(twso,2),round(tagp,2),'NON_ZERO','DEMO_SYNTHETIC'])
pd.DataFrame(rows,columns=['state','district','year','crop','season','soil_type','irrigated','TWSO_simulated','TAGP_simulated','status','data_status']).to_csv('demo_simulated_output_dataset.csv',index=False)
print(f'Saved {len(rows)} rows to demo_simulated_output_dataset.csv')
