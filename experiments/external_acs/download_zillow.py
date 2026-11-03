"""Download Zillow Research ZIP-level home-value and rent series."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'artifacts/external_acs'
ROOT.mkdir(parents=True, exist_ok=True)
FILES = {
    'zillow_zhvi.csv': 'https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
    'zillow_zori.csv': 'https://files.zillowstatic.com/research/public_csvs/zori/Zip_zori_uc_sfrcondomfr_sm_month.csv',
    'zillow_zhvi_low.csv': 'https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfrcondo_tier_0.0_0.33_sm_sa_month.csv',
    'zillow_zhvi_high.csv': 'https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfrcondo_tier_0.67_1.0_sm_sa_month.csv',
    'zillow_zhvi_sfr.csv': 'https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv',
    'zillow_zhvi_condo.csv': 'https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_condo_tier_0.33_0.67_sm_sa_month.csv',
}
sources = []
for name, url in FILES.items():
    dest = ROOT / name
    tmp = dest.with_suffix('.tmp')
    with urllib.request.urlopen(url, timeout=180) as response, tmp.open('wb') as out:
        while block := response.read(1024 * 1024):
            out.write(block)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    tmp.replace(dest)
    sources.append({'name': name, 'url': url, 'sha256': digest})
(ROOT / 'sources_zillow.json').write_text(json.dumps(sources, indent=2) + '\n')
print(json.dumps(sources, indent=2))
