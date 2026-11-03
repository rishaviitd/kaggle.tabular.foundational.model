"""Features and leakage-safe regional summaries for aggregated ZIP models."""
import numpy as np
import pandas as pd

Y = "AffordabilityPercentageTrue"

def feats(f):
 x=f.drop(columns=[Y,"Zip","Month","YearMonth"],errors="ignore").copy();z=f.Zip.astype(str).str.zfill(5)
 x["ZipNumeric"]=pd.to_numeric(z,errors="coerce")
 for k in (1,2,3):x[f"ZipPrefix{k}"]=z.str[:k]
 x["RentToIncome"]=x.RentIndex/(x.AnnualMedianHouseholdIncome/12)
 x["IncomeToRequired"]=x.AnnualMedianHouseholdIncome/(40*x.RentIndex)
 x["IncomePerPerson"]=x.AnnualMedianHouseholdIncome/x.AvgHouseholdSize.clip(lower=.1)
 x["RentXrenter"]=x.RentIndex*x.RenterPercent/100
 return x.replace([np.inf,-np.inf],np.nan)
def rkeys(f):
 z=f.Zip.astype(str).str.zfill(5);s=f.State.fillna("__MISSING__").astype(str);c=f.City.fillna("__MISSING__").astype(str)
 return pd.DataFrame({"Zip1":z.str[:1],"Zip2":z.str[:2],"Zip3":z.str[:3],"State":s,"StateCity":s+"|"+c,"Metro":f.Metro.fillna("__MISSING__").astype(str)})
def enc(ref,q,alpha=12):
 a,b=rkeys(ref),rkeys(q);y=ref[Y].to_numpy();gm=float(y.mean());o=np.zeros((len(q),12),np.float32);j=0
 for col in a:
  st=pd.DataFrame({"k":a[col].to_numpy(),"y":y}).groupby("k").y.agg(["sum","count"])
  n=b[col].map(st["count"]).fillna(0).to_numpy();ss=b[col].map(st["sum"]).fillna(0).to_numpy()
  o[:,j]=(ss+alpha*gm)/(n+alpha);o[:,j+1]=np.log1p(n);j+=2
 return o
