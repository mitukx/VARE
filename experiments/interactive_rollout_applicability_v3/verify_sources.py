"""Verify the pinned public source files; this does not execute upstream code."""
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,pathlib,urllib.request
root=pathlib.Path(__file__).resolve().parent
observations=json.loads((root/'source-contracts.json').read_text())['observations']
def verify(o):
    url=f"https://raw.githubusercontent.com/{o['repository']}/{o['commit']}/{o['path']}"
    body=urllib.request.urlopen(url,timeout=30).read()
    digest=hashlib.sha256(body).hexdigest()
    assert digest == o['source_sha256'], (o['path'],digest)
    assert len(body)==o['source_bytes']
    return {'repository':o['repository'],'commit':o['commit'],'path':o['path'],
            'sha256':digest,'matched':True}
with ThreadPoolExecutor(max_workers=4) as pool:
    results=list(pool.map(verify,observations))
(root/'source-verification.json').write_text(json.dumps(results,indent=2)+'\n')
print(f'Pinned source files matched: {len(results)}. No upstream code executed.')
