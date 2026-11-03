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


class RegionalEncoder:
 """Fit regional target summaries once for repeated serving requests."""
 def __init__(self, reference, alpha=12):
  keys=rkeys(reference);y=reference[Y].to_numpy()
  self.mean=float(y.mean());self.alpha=alpha
  self.stats={col:pd.DataFrame({'k':keys[col].to_numpy(),'y':y}).groupby('k').y.agg(['sum','count']) for col in keys}

 def transform(self, query):
  keys=rkeys(query);out=np.zeros((len(query),12),np.float32)
  for j,(col,stats) in enumerate(self.stats.items()):
   n=keys[col].map(stats['count']).fillna(0).to_numpy()
   total=keys[col].map(stats['sum']).fillna(0).to_numpy()
   out[:,2*j]=(total+self.alpha*self.mean)/(n+self.alpha)
   out[:,2*j+1]=np.log1p(n)
  return out
