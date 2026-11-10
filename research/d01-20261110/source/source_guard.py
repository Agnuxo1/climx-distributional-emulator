from pathlib import Path
import hashlib
import json

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def verify_source(root):
    root=Path(root)
    pin=json.loads((root/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    assert actual==set(pin['files'])|{'SOURCE_MANIFEST.json'}, 'source file set changed'
    for name,sha in pin['files'].items():
        assert digest(root/name)==sha, 'source identity changed: '+name
    return digest(root/'SOURCE_MANIFEST.json')
