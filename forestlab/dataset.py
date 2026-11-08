"""Original bytes, exact-feature grouping and fixed group-level partitions."""
import hashlib,base64
from importlib.resources import files
import numpy as np

SOURCE='https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'
CATALOG='https://archive.ics.uci.edu/dataset/267/banknote+authentication'
SHA256='d0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9'
FEATURES=['wavelet variance','wavelet skewness','wavelet curtosis','image entropy']
SEED=20261108

def load():
    raw=base64.b64decode(files('forestlab').joinpath('data/data_banknote_authentication.b64').read_text().strip(),validate=True)
    if hashlib.sha256(raw).hexdigest()!=SHA256:raise ValueError('original data hash mismatch')
    rows=np.array([[float(s) for s in line.split(',')] for line in raw.decode().splitlines() if line])
    if rows.shape!=(1372,5) or not np.isfinite(rows).all() or not np.isin(rows[:,4],[0,1]).all():raise ValueError('original schema mismatch')
    groups=[];index={}
    for source,row in enumerate(rows):
        key=tuple(row[:4])
        if key not in index:index[key]=len(groups);groups.append(dict(id=len(groups),x=list(map(float,key)),y=int(row[4]),source_rows=[]))
        group=groups[index[key]]
        if group['y']!=row[4]:raise ValueError('conflicting exact-copy labels')
        group['source_rows'].append(source)
    if len(groups)!=1348:raise ValueError('group count mismatch')
    order=np.random.default_rng(SEED).permutation(len(groups)).tolist()
    split={'train':order[:808],'validation':order[808:1078],'test':order[1078:]}
    return dict(source=SOURCE,catalog=CATALOG,source_sha256=SHA256,source_rows=1372,unit='one exact four-feature vector group',features=FEATURES,seed=SEED,groups=groups,split=split)

def arrays(data,part):
    ids=data['split'][part]
    return np.array([data['groups'][j]['x'] for j in ids]),np.array([data['groups'][j]['y'] for j in ids])
