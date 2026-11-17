"""Recorded phase-two schedule; benchmark cases are strictly sequential.

Training jobs may run on two CPU workers and two MPS workers. Their elapsed
times are operational logs; isolated profile cases establish speed/memory.
"""
import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path


def execute(command,log):
    with open(log,'w') as out:
        result=subprocess.run([sys.executable,*command],stdout=out,stderr=subprocess.STDOUT)
    print(json.dumps({'command':command,'log':str(log),'returncode':result.returncode}),flush=True)
    if result.returncode:raise RuntimeError(f'Failed: {log}')


def profiles():
    logs=Path('results/phase2/profile_logs');logs.mkdir(exist_ok=True)
    for device in ['cpu','mps']:
        for backend in ['dense','tiled','checkpoint']:
            for method in ['analytic','midpoint']:
                stem=f'{device}_{backend}_{method}'
                execute(['scripts/profile_tiled.py','--device',device,'--backend',backend,'--method',method],logs/f'{stem}.log')
    for device in ['cpu','mps']:
        for tile in [16,48]:
            for method in ['analytic','midpoint']:
                stem=f'{device}_checkpoint_{method}_tile{tile}'
                execute(['scripts/profile_tiled.py','--device',device,'--backend','checkpoint','--method',method,'--tile',str(tile)],logs/f'{stem}.log')


def train():
    logs=Path('results/phase2/training_logs');logs.mkdir(exist_ok=True)
    jobs=[]
    for device in ['cpu','mps']:
        for sequence in ['freiburg1_xyz','freiburg1_desk']:
            for track in ['native','controlled']:
                # Isolated profiles select MPS for the three-seed cohort.
                # CPU uses paired seed0 controlled cases for device checks.
                if device=='cpu' and track=='native':continue
                for seed in ([0,1,2] if device=='mps' else [0]):
                    for method in ['analytic','midpoint']:
                        jobs.append((device,sequence,track,seed,method))
    def job(row):
        device,sequence,track,seed,method=row
        stem=f'{sequence}_{track}_{method}_{device}_seed{seed}'
        command=['scripts/rgbd_experiment.py','--device',device,'--sequence',sequence,'--track',track,
                 '--method',method,'--seed',str(seed),'--steps','150']
        # Resume only a completed exact scheduled configuration.
        existing=Path('results/phase2/runs')/stem/'summary.json'
        if existing.exists():
            saved=json.loads(existing.read_text())
            assert saved['steps']==150
            print('Retained completed run '+stem,flush=True);return
        execute(command,logs/f'{stem}.log')
    cpu=[j for j in jobs if j[0]=='cpu'];mps=[j for j in jobs if j[0]=='mps']
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as outer:
        def cpu_queue():
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(job,cpu))
        def mps_queue():
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(job,mps))
        futures=[outer.submit(cpu_queue),outer.submit(mps_queue)]
        for f in futures:f.result()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['profiles','train']);a=p.parse_args()
    profiles() if a.stage=='profiles' else train()
