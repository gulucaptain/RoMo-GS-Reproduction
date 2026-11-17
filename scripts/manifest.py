"""Hash original code/configuration and recorded results; verify without rerunning."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--check",action="store_true");args=parser.parse_args()
    path=Path("results/manifest.json")
    if args.check:
        data=json.loads(path.read_text())
        for file,digest in data["files"].items():
            assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,file
        print(f"Verified {len(data['files'])} file hashes")
    else:
        roots=[Path(p) for p in ["romo_gs","scripts","tests","docs","results","paper/source.json","README.md","COVERAGE.md","requirements-lock.txt","pyproject.toml"]]
        files=[]
        for root in roots:
            files.extend([root] if root.is_file() else [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts and "local_data" not in p.parts])
        values={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files)) if p!=path}
        path.write_text(json.dumps({"algorithm":"SHA-256","files":values},indent=2)+"\n")
        print(f"Hashed {len(values)} files")


if __name__=="__main__":main()
