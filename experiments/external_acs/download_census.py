import concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parents[2]/'artifacts/external_acs';root.mkdir(parents=True,exist_ok=True)
names=['b25106','b19001','b25077']
def fetch(name):
 url='https://www2.census.gov/programs-surveys/acs/summary_file/2024/table-based-SF/data/5YRData/acsdt5y2024-'+name+'.dat'
 dest=root/(name+'_zcta.dat');tmp=dest.with_suffix('.tmp');digest=hashlib.sha256();n=0
 with urllib.request.urlopen(url,timeout=90) as response, tmp.open('wb') as out:
  for line in response:
   digest.update(line)
   if line.startswith(b'GEO_ID|') or line.startswith(b'860Z200US'):
    out.write(line)
    if line.startswith(b'860Z200US'):n+=1
 if not n:raise ValueError('No ZCTAs for '+name)
 tmp.replace(dest)
 return {'table':name,'url':url,'sha256_original':digest.hexdigest(),'zcta_rows':n}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 results=list(pool.map(fetch,names))
(root/'sources.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
