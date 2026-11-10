"""Optional AWS checkpoint persistence. Hash-addressed payload, atomic pointer.
The caller supplies its own experiment-specific S3 prefix and protocol digest.
"""
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
from common import sha256


class S3CheckpointStore:
    def __init__(self, uri, fingerprint, client=None):
        parsed = urlparse(uri)
        if parsed.scheme != 's3' or not parsed.netloc or not parsed.path.strip('/'):
            raise ValueError('Use s3://bucket/experiment-specific-prefix')
        self.bucket = parsed.netloc; self.prefix = parsed.path.strip('/'); self.fingerprint = fingerprint
        if client is None:
            import boto3
            client = boto3.client('s3')
        self.client = client

    def pointer(self, name, kind):
        if Path(name).name != name or kind not in ('resume','final'): raise ValueError('Invalid checkpoint key')
        key = f'{self.prefix}/{name}/{kind}.json'
        try: value = json.loads(self.client.get_object(Bucket=self.bucket,Key=key)['Body'].read())
        except Exception as exc:
            if getattr(exc,'response',{}).get('Error',{}).get('Code') in ('NoSuchKey','404','NotFound'): return None
            raise
        if value['fingerprint'] != self.fingerprint: raise ValueError('Remote checkpoint protocol differs')
        expected = f'{self.prefix}/{name}/{value["sha256"]}.pt'
        if value['key'] != expected: raise ValueError('Remote pointer outside checkpoint namespace')
        return value

    def save(self, name, kind, path):
        digest = sha256(path); key = f'{self.prefix}/{name}/{digest}.pt'
        previous = self.pointer(name,kind)
        if previous and previous['sha256'] == digest: return
        self.client.upload_file(str(path), self.bucket, key)
        value = dict(key=key,sha256=digest,fingerprint=self.fingerprint)
        self.client.put_object(Bucket=self.bucket,Key=f'{self.prefix}/{name}/{kind}.json',Body=json.dumps(value).encode())
        # Retain hash-addressed payloads: no deletion races with readers. Apply
        # the account's lifecycle policy after the campaign, not mid-training.

    def restore(self, name, kind, path):
        value = self.pointer(name,kind)
        if value is None: return False
        path = Path(path); path.parent.mkdir(parents=True,exist_ok=True); temp = path.with_suffix('.download')
        self.client.download_file(self.bucket,value['key'],str(temp))
        if sha256(temp) != value['sha256']:
            temp.unlink(); raise ValueError('Remote checkpoint checksum failed')
        temp.replace(path); return True


def protocol_fingerprint(config, inputs):
    # Paths are metadata; hashes and config values identify the actual protocol.
    return hashlib.sha256(json.dumps(dict(config=config,inputs=inputs),sort_keys=True).encode()).hexdigest()
