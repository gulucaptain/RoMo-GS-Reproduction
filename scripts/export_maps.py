"""Export representative final Gaussian centers as colored PLY point clouds.

The complete floating-point Gaussian parameters remain in parameters.npz.
These PLY files are center/scale inspection artifacts, not surface meshes or
an assertion of a third-party Gaussian-viewer schema.
"""
import json
from pathlib import Path
import numpy as np


def main():
    target=Path('results/phase2/maps');target.mkdir(exist_ok=True)
    records=[]
    for folder in sorted(Path('results/phase2/runs').iterdir()):
        if not (folder/'summary.json').exists():continue
        config=json.loads((folder/'config.json').read_text())
        if config['device']!='mps' or config['seed']!=0:continue
        with np.load(folder/'parameters.npz') as p:
            means=p['means'];scales=p['scales'];colors=np.uint8(np.clip(p['colors']*255,0,255));opacity=p['opacity']
        path=target/(folder.name+'.ply')
        header=['ply','format ascii 1.0','comment Gaussian centers only; world coordinates in metres',
                'comment sigma_x/y/z are diagonal world covariance standard deviations; no surface mesh',
                f'element vertex {len(means)}','property float x','property float y','property float z',
                'property uchar red','property uchar green','property uchar blue','property float opacity',
                'property float sigma_x','property float sigma_y','property float sigma_z','end_header']
        with open(path,'w') as f:
            f.write('\n'.join(header)+'\n')
            for xyz,rgb,o,s in zip(means,colors,opacity,scales):
                f.write(' '.join([*(f'{v:.8g}' for v in xyz),*(str(int(v)) for v in rgb),f'{o:.8g}',*(f'{v:.8g}' for v in s)])+'\n')
        records.append({'run':folder.name,'path':str(path),'points':len(means),'scope':'Gaussian center point cloud with color/opacity/sigma; original full-precision arrays preserved separately'})
    assert len(records)==8
    (target/'index.json').write_text(json.dumps(records,indent=2))
    print('Exported8representative maps,2400Gaussian centers each')


if __name__=='__main__':main()
