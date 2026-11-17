"""Verify the public snapshot without requiring datasets or ML dependencies."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
records=json.loads((root/'PUBLICATION_FILES.json').read_text())
failed=[f for f,h in records.items() if not (root/f).is_file() or hashlib.sha256((root/f).read_bytes()).hexdigest()!=h]
if failed: raise SystemExit('Integrity mismatch: '+', '.join(failed))
print(f'Verified {len(records)} public files.')
