"""Frozen selection protocol. Test labels are read only after all selections."""
import json,hashlib,math
from importlib.resources import files
from pathlib import Path
import numpy as np
from .core import forest,tree,votes,predict_tree,oob,metrics,feature_digest
from .dataset import load,arrays
from .controls import logistic,nearest,scores,sigmoid

SCOPE='Historical exact-feature-group benchmark, not independent specimen validation, banknote authentication or calibrated confidence. Hard-vote fractions are not probabilities.'

def key(candidate):
    m=candidate['validation'];return m['errors'],m['brier'],candidate['id']

def run(data=None):
    data=load() if data is None else data
    X,y=arrays(data,'train');V,v=arrays(data,'validation');candidates=[]
    for depth in [4,None]:
        for leaf in [2,8]:
            for mtry in [2,4]:
                identifier='forest-depth-'+str(depth)+'-leaf-'+str(leaf)+'-features-'+str(mtry)
                model=forest(X,y,n_trees=31,seed=20261108,max_depth=depth,min_leaf=leaf,max_features=mtry)
                score=votes(model,V)['fraction_one'];out=oob(model,X);valid=[i for i,r in enumerate(out) if r['count']]
                candidates.append(dict(id=identifier,model=model,validation=metrics(v,score),oob=out,oob_metrics=metrics(y[valid],[out[i]['fraction_one'] for i in valid]),oob_missing=len(y)-len(valid),nodes=sum(len(t['nodes']) for t in model['trees'])))
    selected=min(candidates,key=key)['id']
    controls=[]
    for lam in [.001,.01,.1,1.]:
        model=logistic(X,y,lam);controls.append(dict(id='logistic-'+str(lam),model=model,validation=metrics(v,scores(model,V))))
    for k in [1,5]:
        model=nearest(X,y,k);controls.append(dict(id='nearest-'+str(k),model=model,validation=metrics(v,scores(model,V))))
    model=dict(kind='constant',fraction_one=float(y.mean()));controls.append(dict(id='training-frequency',model=model,validation=metrics(v,scores(model,V))))
    selected_control=min(controls,key=key)['id']
    singles=[]
    for depth in [4,None]:
        for leaf in [2,8]:
            model=tree(X,y,max_depth=depth,min_leaf=leaf,max_features=4,seed=20261108)
            singles.append(dict(id='single-depth-'+str(depth)+'-leaf-'+str(leaf),model=model,validation=metrics(v,predict_tree(model,V))))
    selected_single=min(singles,key=key)['id']
    # Freeze all selections BEFORE accessing the test matrix/labels.
    T,t=arrays(data,'test')
    for c in candidates:c['test']=metrics(t,votes(c['model'],T)['fraction_one'])
    for c in controls:c['test']=metrics(t,scores(c['model'],T))
    for c in singles:c['test']=metrics(t,predict_tree(c['model'],T))
    return dict(schema=1,scope=SCOPE,data=data,candidates=candidates,controls=controls,singles=singles,selected_id=selected,selected_control_id=selected_control,selected_single_id=selected_single)

def save(report,path):
    Path(path).write_text(json.dumps(report,separators=(',',':'),allow_nan=False)+'\n')

def saved():
    return json.loads(files('forestlab').joinpath('artifacts/report.json').read_text())

def audit_tree(model,X,y):
    nodes=model['nodes'];seen=[]
    def visit(n,indices):
        if n in seen:raise ValueError('tree cycle/shared node')
        seen.append(n);node=nodes[n];a=y[indices];counts=[int((a==0).sum()),int(a.sum())]
        if node['id']!=n or node['counts']!=counts or node['prediction']!=int(counts[1]>counts[0]) or node['fraction_one']!=counts[1]/sum(counts):raise ValueError('node count/prediction mismatch')
        if node['leaf']:return
        if len(set(node['searched_features']))!=model['max_features'] or len(node['searched_features'])!=model['max_features'] or any(type(f) is not int or not 0<=f<X.shape[1] for f in node['searched_features']) or not 0<=node['feature']<X.shape[1] or node['feature'] not in node['searched_features']:raise ValueError('split feature mismatch')
        mask=X[indices,node['feature']]<=node['cut'];left=indices[mask];right=indices[~mask]
        if min(len(left),len(right))<model['min_leaf'] or (model['max_depth'] is not None and node['depth']>=model['max_depth']):raise ValueError('split stopping constraint')
        from .core import split
        best=split(X[indices],y[indices],node['searched_features'],model['min_leaf'])
        if best is None or any(node[k]!=best[k] for k in best):raise ValueError('split witness/optimum mismatch')
        if not any(X[indices,node['feature']]==node['cut']):raise ValueError('cut is not observed lower value')
        for child in [node['left'],node['right']]:
            if nodes[child]['depth']!=node['depth']+1:raise ValueError('depth mismatch')
        visit(node['left'],left);visit(node['right'],right)
    visit(0,np.arange(len(y)))
    if len(seen)!=len(nodes):raise ValueError('unreachable node')
    return len(nodes)

def equivalent(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    if type(a) is float:return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=2e-12,abs_tol=2e-12)
    return a==b

def audit(report):
    original=load();data=report['data']
    if data!=original:raise ValueError('source/group/split provenance mismatch')
    X,y=arrays(data,'train');V,v=arrays(data,'validation');T,t=arrays(data,'test');results=[]
    expected={('forest-depth-'+str(d)+'-leaf-'+str(l)+'-features-'+str(f)):(d,l,f) for d in [4,None] for l in [2,8] for f in [2,4]}
    if len(report['candidates'])!=8 or {c['id'] for c in report['candidates']}!=set(expected):raise ValueError('frozen forest grid mismatch')
    if len(report['controls'])!=7 or len(report['singles'])!=4:raise ValueError('control grid size mismatch')
    for c in report['candidates']:
        m=c['model']
        if (m['max_depth'],m['min_leaf'],m['max_features'])!=expected[c['id']] or m['seed']!=20261108:raise ValueError('forest configuration mismatch')
        if m['training_feature_sha256']!=feature_digest(X) or m['n_train']!=len(y) or m['n_trees']!=31:raise ValueError('training binding mismatch')
        rng=np.random.default_rng(m['seed']);nodes=0
        for tree_,counts in zip(m['trees'],m['inbag']):
            indices=rng.integers(0,len(y),len(y));want=np.bincount(indices,minlength=len(y)).tolist()
            if counts!=want:raise ValueError('bootstrap membership mismatch')
            nodes+=audit_tree(tree_,X[indices],y[indices])
        if len(m['trees'])!=31 or len(m['inbag'])!=31 or nodes!=c['nodes']:raise ValueError('forest size mismatch')
        if not equivalent(c['validation'],metrics(v,votes(m,V)['fraction_one'])):raise ValueError('validation mismatch')
        if not equivalent(c['test'],metrics(t,votes(m,T)['fraction_one'])) or c['oob']!=oob(m,X):raise ValueError('test/OOB mismatch')
        valid=[i for i,r in enumerate(c['oob']) if r['count']]
        if not equivalent(c['oob_metrics'],metrics(y[valid],[c['oob'][i]['fraction_one'] for i in valid])) or c['oob_missing']!=len(y)-len(valid):raise ValueError('OOB metric mismatch')
        results.append(dict(id=c['id'],nodes=nodes))
    expected_controls={'logistic-'+str(v) for v in [.001,.01,.1,1.]}|{'nearest-1','nearest-5','training-frequency'}
    if {c['id'] for c in report['controls']}!=expected_controls:raise ValueError('control IDs mismatch')
    mean=X.mean(axis=0);scale=X.std(axis=0);Z=(X-mean)/scale
    for c in report['controls']:
        m=c['model']
        if m['kind']=='constant':
            if c['id']!='training-frequency' or m['fraction_one']!=float(y.mean()):raise ValueError('constant control binding mismatch')
        else:
            if not equivalent(m['mean'],mean.tolist()) or not equivalent(m['scale'],scale.tolist()):raise ValueError('preprocessing source mismatch')
            if m['kind']=='nearest':
                if c['id']!='nearest-'+str(m['k']) or m['k'] not in [1,5] or m['labels']!=y.tolist() or not equivalent(m['train'],Z.tolist()):raise ValueError('neighbor training binding mismatch')
            elif m['kind']=='logistic':
                if c['id']!='logistic-'+str(m['regularization']) or m['regularization'] not in [.001,.01,.1,1.]:raise ValueError('logistic configuration mismatch')
                A=np.column_stack([np.ones(len(y)),Z]);theta=np.asarray(m['theta']);penalty=np.array([0]+[m['regularization']]*4)
                g=A.T@(sigmoid(A@theta)-y)/len(y)+penalty*theta
                objective=float(np.mean(np.logaddexp(0,A@theta)-y*(A@theta))+.5*np.dot(penalty*theta,theta))
                if np.max(np.abs(g))>2e-9 or not equivalent(m['history'][-1]['gradient_inf'],float(np.max(np.abs(g)))) or not equivalent(m['history'][-1]['objective'],objective) or m['reason']!='gradient_tolerance' or any(a['objective']<b['objective']-1e-13 for a,b in zip(m['history'],m['history'][1:])):raise ValueError('logistic final optimality/history mismatch')
            else:raise ValueError('unknown control kind')
        if not equivalent(c['validation'],metrics(v,scores(c['model'],V))) or not equivalent(c['test'],metrics(t,scores(c['model'],T))):raise ValueError('control metric mismatch')
    for c in report['singles']:
        audit_tree(c['model'],X,y)
        if c['validation']!=metrics(v,predict_tree(c['model'],V)) or c['test']!=metrics(t,predict_tree(c['model'],T)):raise ValueError('single tree metrics mismatch')
    if report['selected_id']!=min(report['candidates'],key=key)['id'] or report['selected_control_id']!=min(report['controls'],key=key)['id'] or report['selected_single_id']!=min(report['singles'],key=key)['id']:raise ValueError('validation selection mismatch')
    return dict(forests=results,controls=len(report['controls']),singles=len(report['singles']),scope=SCOPE)
