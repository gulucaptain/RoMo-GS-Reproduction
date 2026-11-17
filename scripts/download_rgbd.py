"""Download the exact reviewed public TUM archives; no raw data in Git."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--sequences',nargs='+',choices=['freiburg1_xyz','freiburg1_desk'],default=['freiburg1_xyz','freiburg1_desk']);a=p.parse_args()
    folder=Path('data/downloads');folder.mkdir(parents=True,exist_ok=True)
    for seq in a.sequences:
        source=json.loads(Path(f'results/phase2/{seq}_source.json').read_text())
        path=folder/f'{seq}.tgz'
        if not path.exists():
            partial=path.with_suffix('.tgz.part')
            with urllib.request.urlopen(source['url'],timeout=60) as remote,open(partial,'wb') as out:
                while block:=remote.read(2**20):out.write(block)
            assert sha256(partial)==source['archive_sha256'],'Downloaded archive differs from reviewed source'
            partial.replace(path)
        assert sha256(path)==source['archive_sha256'],f'Existing cache differs: {path}'
        print(f'Verified {seq}: {path.stat().st_size} bytes',flush=True)


if __name__=='__main__':main()
