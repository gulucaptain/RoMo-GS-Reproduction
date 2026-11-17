"""Final-model profiles must run only after the training queue has stopped."""
import json
import subprocess
import sys
from pathlib import Path


def main():
    runs=Path('results/phase2/runs')
    assert len(list(runs.glob('*/summary.json')))==28
    logs=Path('results/phase2/final_profile_logs');logs.mkdir(exist_ok=True)
    for folder in sorted(runs.iterdir()):
        cfg=json.loads((folder/'config.json').read_text())
        if cfg['seed']!=0:continue
        command=[sys.executable,'scripts/profile_final.py','--run',str(folder),'--device',cfg['device']]
        with open(logs/(folder.name+'.log'),'w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        print(json.dumps({'run':folder.name,'returncode':r.returncode}),flush=True)
        assert r.returncode==0,folder


if __name__=='__main__':main()
