"""Stream Realtor.com Economic Research ZIP listings; retain May/June 2026."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'artifacts/external_acs'
ROOT.mkdir(parents=True, exist_ok=True)
URL = 'https://econdata.s3-us-west-2.amazonaws.com/Reports/Core/RDC_Inventory_Core_Metrics_Zip_History.csv'
dest = ROOT / 'realtor_zip_2026may_june.csv'
tmp = dest.with_suffix('.tmp')
digest = hashlib.sha256()
count = 0
with urllib.request.urlopen(URL, timeout=180) as response, tmp.open('wb') as out:
    for index, line in enumerate(response):
        digest.update(line)
        if index == 0 or line.startswith((b'202605,', b'202606,')):
            out.write(line)
            count += index != 0
if not count:
    raise ValueError('No May/June 2026 Realtor ZIP rows')
tmp.replace(dest)
meta = {'source': 'Realtor.com Economic Research', 'url': URL,
        'sha256_original': digest.hexdigest(), 'filtered_rows': count}
(ROOT / 'sources_realtor.json').write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps(meta, indent=2))
