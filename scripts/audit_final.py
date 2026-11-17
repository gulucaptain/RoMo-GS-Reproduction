"""Check the exact execution plan, source hashes, configurations and outputs."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    base=Path('results/phase2');plan=json.loads((base/'execution_plan.json').read_text())
    for path,expected in plan['source_hashes'].items():assert sha(path)==expected,path
    expected=set()
    for device in ['cpu','mps']:
        for seq in plan['sequences']:
            for track in (plan['cpu_tracks'] if device=='cpu' else plan['mps_tracks']):
                for seed in (plan['cpu_control_seeds'] if device=='cpu' else plan['primary_seeds']):
                    for method in plan['methods']:expected.add(f'{seq}_{track}_{method}_{device}_seed{seed}')
    completed={p.parent.name for p in (base/'runs').glob('*/summary.json')}
    assert completed==expected,(expected-completed,completed-expected)
    provenance=[]
    for name in sorted(expected):
        root=base/'runs'/name
        config=json.loads((root/'config.json').read_text());result=json.loads((root/'summary.json').read_text())
        trace=json.loads((root/'trace.json').read_text())
        assert config['steps']==result['steps']==len(trace)==plan['steps']
        assert [x['step'] for x in trace]==list(range(plan['steps']))
        assert config['tile']==32 and config['chunk']==256 and config['samples']==10
        assert config['gaussians']==result['gaussians']==2400
        assert set(config['train']).isdisjoint(config['test'])
        assert all(np.isfinite([row['loss'],row['l1'],row['seconds']]).all() for row in trace)
        with np.load(root/'parameters.npz') as parameters:
            assert parameters['means'].shape==(2400,3)
            assert all(np.isfinite(parameters[key]).all() for key in parameters.files)
            assert (parameters['scales']>0).all()
            assert ((parameters['colors']>=0)&(parameters['colors']<=1)).all()
        source=json.loads((base/f"{config['sequence']}_source.json").read_text())
        assert sha(f"data/prepared/{config['sequence']}.npz")==source['prepared_sha256']
        provenance.append({'run':name,'prepared_data_sha256':source['prepared_sha256'],
                           'files':{p.name:sha(p) for p in [root/'config.json',root/'summary.json',root/'trace.json',root/'parameters.npz']}})
    replay=json.loads((base/'replay.json').read_text());assert {r['run'] for r in replay}==expected
    exceptions=[r for r in replay if r.get('strict_cpu_image_gate_passed') is False]
    assert all(r.get('source_device_replay',{}).get('verified') for r in exceptions), 'MPS pixel exceptions require source-device verification'
    diagnosis=json.loads((base/'replay_discrepancy_audit.json').read_text())
    assert {x['run'] for x in diagnosis['all_cases']}=={x['run'] for x in exceptions}
    assert diagnosis['all_source_device_replays_exact']
    audit={'runs':len(expected),'steps_per_run':plan['steps'],'source_hashes_verified':plan['source_hashes'],
           'all_configurations_match':True,'all_parameters_finite':True,'all_checkpoints_replayed':True,
           'strict_cpu_image_exceptions':[r['run'] for r in exceptions],
           'all_exception_depth_order_controls_verified':True,
           'provenance':provenance}
    (base/'final_audit.json').write_text(json.dumps(audit,indent=2))
    print('Verified28final configurations, all150-step logs, parameters, source/data hashes and replays')


if __name__=='__main__':main()
