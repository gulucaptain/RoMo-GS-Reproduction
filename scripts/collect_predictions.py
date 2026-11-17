"""Retain all final model-rendered outputs, without original RGB-D inputs."""
import hashlib
import json
import shutil
from pathlib import Path


def main():
    out=Path('results/phase2/predictions');out.mkdir(exist_ok=True)
    rows=[]
    for folder in sorted(Path('results/phase2/runs').iterdir()):
        if not (folder/'summary.json').exists():continue
        cfg=json.loads((folder/'config.json').read_text())
        source=Path('results/phase2/local_data')/(folder.name+'_predictions.npz')
        target=out/source.name
        if source.exists():shutil.copyfile(source,target)
        elif not target.exists():raise FileNotFoundError('Rebuild model prediction outputs before collecting: '+str(source))
        rows.append({'run':folder.name,'path':str(target),'render_device':cfg['device'],
                     'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                     'contents':'Only model-rendered held-out RGB/depth/alpha and exposure-rendered RGB; no original input RGB-D'})
    assert len(rows)==28
    (out/'index.json').write_text(json.dumps(rows,indent=2));print('Retained28final model-rendered output archives')


if __name__=='__main__':main()
