"""Post-fit joint row-permutation diagnostics, not causal feature effects.

The original validation partition already selected the models. This is a
reused-validation diagnostic, not a new unbiased test. No fit or selection here.
"""
import hashlib,json,math,base64,gzip
from importlib.resources import files
from pathlib import Path
import numpy as np
from .core import design,metrics,feature_digest
from .controls import scores

SEED=2026110802
REPEATS=16
SCOPE='Reused-validation post-fit diagnostic. Joint block shuffles preserve within-block dependence, break its relation to remaining features/labels, and can create unrealistic rows. Not causal importance, a p-value, uncertainty interval, OOB estimate or fresh test performance. No refit or reselection.'


def permutation(n, repeats=REPEATS, seed=SEED):
    if type(n) is not int or n<2 or type(repeats) is not int or not 1<=repeats<=128 or type(seed) is not int or seed<0:
        raise ValueError('at least two rows, 1..128 repeats and nonnegative integer seed required')
    rng=np.random.default_rng(seed)
    return [rng.permutation(n).tolist() for _ in range(repeats)]


def hybrid(X, block, donor):
    X=design(X);n,d=X.shape
    if not isinstance(block,(list,tuple)) or not block or any(type(f) is not int or not 0<=f<d for f in block) or len(set(block))!=len(block):
        raise ValueError('distinct valid nonempty feature block required')
    if not isinstance(donor,list) or len(donor)!=n or any(type(i) is not int for i in donor) or sorted(donor)!=list(range(n)):
        raise ValueError('donor must be a complete row bijection')
    out=X.copy();out[:,list(block)]=X[np.array(donor)[:,None],np.array(block)]
    return out


def forest_counts(model,X):
    """Batch tree routing. Same lower-cut rule and hard votes as core.trace."""
    X=design(X)
    if X.shape[1]!=model['n_features']:raise ValueError('model width mismatch')
    result=np.zeros(len(X),dtype=int)
    for tree in model['trees']:
        todo=[(0,np.arange(len(X)))]
        while todo:
            idx,rows=todo.pop();node=tree['nodes'][idx]
            if node['leaf']:
                result[rows]+=node['prediction'];continue
            left=X[rows,node['feature']]<=node['cut']
            if left.any():todo.append((node['left'],rows[left]))
            if (~left).any():todo.append((node['right'],rows[~left]))
    return result


def chosen(report):
    a=next(c for c in report['candidates'] if c['id']==report['selected_id'])
    b=next(c for c in report['controls'] if c['id']==report['selected_control_id'])
    return a,b


def summarize(counts, denominator, y):
    if type(denominator) is not int or denominator<1:raise ValueError('positive integer denominator required')
    a=np.asarray(counts)
    if a.shape!=np.asarray(y).shape or a.dtype.kind not in 'iu' or (a<0).any() or (a>denominator).any():raise ValueError('bounded integer counts required')
    return metrics(y,a/denominator)


def compute(report=None):
    from .experiment import saved
    report=saved() if report is None else report
    a,b=chosen(report);data=report['data'];ids=data['split']['validation']
    X=np.array([data['groups'][i]['x'] for i in ids]);y=np.array([data['groups'][i]['y'] for i in ids])
    if X.shape[1]!=4 or b['model']['kind']!='nearest':raise ValueError('frozen four-feature nearest-control protocol required')
    den=[a['model']['n_trees'],b['model']['k']]
    def evaluate(Z):
        first=forest_counts(a['model'],Z).tolist()
        second=np.rint(scores(b['model'],Z)*den[1]).astype(int).tolist()
        return [first,second]
    base=evaluate(X);base_metrics=[summarize(v,d,y) for v,d in zip(base,den)]
    maps=permutation(len(X));blocks=[];known={tuple(row) for row in X.tolist()}
    for mask in range(1,16):
        features=[f for f in range(4) if mask&(1<<f)];runs=[]
        for donor in maps:
            Z=hybrid(X,features,donor);counts=evaluate(Z);ms=[summarize(v,d,y) for v,d in zip(counts,den)]
            runs.append(dict(counts=counts,metrics=ms,error_delta=[m['errors']-o['errors'] for m,o in zip(ms,base_metrics)],unchanged_rows=int(np.all(Z==X,axis=1).sum()),rows_in_validation_support=sum(tuple(z) in known for z in Z.tolist())))
        blocks.append(dict(mask=mask,features=features,runs=runs))
    binding=hashlib.sha256(json.dumps(dict(ids=[a['id'],b['id']],models=[a['model'],b['model']]),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return dict(schema=1,scope=SCOPE,seed=SEED,repeats=REPEATS,partition='validation',validation_ids=ids,features=data['features'],feature_sha256=feature_digest(X),label_sha256=hashlib.sha256(bytes(y.tolist())).hexdigest(),model_sha256=binding,model_ids=[a['id'],b['id']],denominators=den,baseline=dict(counts=base,metrics=base_metrics),donor_maps=maps,blocks=blocks)


def save(obj,path):
    Path(path).write_text(json.dumps(obj,separators=(',',':'),allow_nan=False)+'\n')


def saved():
    spec=json.loads(files('forestlab').joinpath('artifacts/permutation.json').read_text())
    if spec['encoding']!='gzip-base64':raise ValueError('unknown permutation encoding')
    raw=gzip.decompress(base64.b64decode(''.join(files('forestlab').joinpath('artifacts/'+name).read_text().strip() for name in spec['parts']),validate=True))
    if hashlib.sha256(raw).hexdigest()!=spec['sha256']:raise ValueError('permutation archive digest mismatch')
    return json.loads(raw)


def audit(obj=None,report=None):
    from .experiment import equivalent
    obj=saved() if obj is None else obj
    expected=compute(report)
    if not equivalent(obj,expected):raise ValueError('permutation provenance, bijection, fit binding or diagnostic mismatch')
    return dict(status='PASS',blocks=len(obj['blocks']),repeats=obj['repeats'],model_row_evaluations=len(obj['blocks'])*obj['repeats']*len(obj['validation_ids'])*2)


def pack(obj,root):
    root=Path(root);raw=(json.dumps(obj,separators=(',',':'),allow_nan=False)+'\n').encode()
    encoded=base64.b64encode(gzip.compress(raw,mtime=0)).decode();parts=[]
    for i,start in enumerate(range(0,len(encoded),180000)):
        name='permutation-'+str(i)+'.b64';(root/name).write_text(encoded[start:start+180000]+'\n');parts.append(name)
    spec=dict(encoding='gzip-base64',sha256=hashlib.sha256(raw).hexdigest(),parts=parts)
    (root/'permutation.json').write_text(json.dumps(spec,separators=(',',':'))+'\n')
    return spec


def export(root):
    # Verify the archive and forward its exact bytes without recompression.
    saved();root=Path(root);root.mkdir(parents=True,exist_ok=True)
    spec=files('forestlab').joinpath('artifacts/permutation.json').read_text().strip()
    manifest=json.loads(spec)
    for i,name in enumerate(manifest['parts']):
        chunk=files('forestlab').joinpath('artifacts/'+name).read_text().strip()
        (root/('permutation-archive-'+str(i)+'.js')).write_text('globalThis.PERM_PARTS=globalThis.PERM_PARTS||[];globalThis.PERM_PARTS.push('+json.dumps(chunk)+');\n')
    loader='globalThis.PERM_ARCHIVE='+spec+";\nglobalThis.PERM_READY=(async()=>{const raw=new Uint8Array(await new Response(new Blob([Uint8Array.from(atob(PERM_PARTS.join('')),c=>c.charCodeAt(0))]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',raw))).map(v=>v.toString(16).padStart(2,'0')).join('');if(digest!==PERM_ARCHIVE.sha256)throw Error('Permutation digest mismatch');return globalThis.PERM_DATA=JSON.parse(new TextDecoder().decode(raw));})();\n"
    (root/'permutation-data.js').write_text(loader)
    return dict(parts=len(manifest['parts']),sha256=manifest['sha256'])
