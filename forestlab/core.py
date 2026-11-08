"""Binary CART partitions and hard-vote forests, with auditable bootstrap counts.

No estimator is imported. Gini costs are compared by integer cross products.
Cuts use the lower observed value, NOT midpoints. Votes are not calibrated
probabilities, and per-tree disagreement is not a confidence interval.
"""
import math
import hashlib,json
import numpy as np


def design(X, y=None):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not len(X) or not X.shape[1] or not np.isfinite(X).all():
        raise ValueError('finite nonempty 2D features required')
    if y is None:
        return X
    y = np.asarray(y)
    if y.shape != (len(X),) or not np.isin(y, [0, 1]).all():
        raise ValueError('one binary label per row required')
    return X, y.astype(int)


def positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name+' must be a positive integer')
    return value


def split(X, y, features, min_leaf):
    """Return global best admissible split over supplied features.

    Compare c=l0*l1/nL+r0*r1/nR exactly. Weighted child Gini is 2c/N.
    Equal costs choose smallest feature, then smallest observed lower cut.
    Counts include bootstrap duplicates and integer arithmetic has no overflow.
    """
    X, y = design(X, y)
    positive(min_leaf, 'min_leaf')
    if not features or any(type(f) is not int or not 0 <= f < X.shape[1] for f in features) or len(set(features)) != len(features):
        raise ValueError('distinct known features required')
    n = len(y); total = int(y.sum()); best = None
    for f in sorted(features):
        order = np.argsort(X[:, f], kind='stable')
        vals = X[order, f]; labels = y[order]; l1 = 0
        for k in range(1, n):
            l1 += int(labels[k-1])
            if k < min_leaf or n-k < min_leaf or vals[k-1] == vals[k]:
                continue
            l0 = k-l1; r1 = total-l1; r0 = n-k-r1
            numerator = l0*l1*(n-k)+r0*r1*k
            denominator = k*(n-k)
            if best is None or numerator*best['cost_denominator'] < best['cost_numerator']*denominator:
                best = dict(feature=f, cut=float(vals[k-1]), cost_numerator=numerator, cost_denominator=denominator, left_counts=[l0,l1], right_counts=[r0,r1])
    if best is None:
        return None
    # Parent impurity uses total*(n-total)/n before multiplying by 2/n.
    if best['cost_numerator']*n >= total*(n-total)*best['cost_denominator']:
        return None
    best['gain'] = 2*(total*(n-total)/n-best['cost_numerator']/best['cost_denominator'])/n
    return best


def tree(X, y, *, max_depth=None, min_leaf=1, max_features=None, seed=0):
    X, y = design(X, y); positive(min_leaf, 'min_leaf')
    if max_depth is not None:
        positive(max_depth, 'max_depth')
    if max_features is None:
        max_features = X.shape[1]
    positive(max_features, 'max_features')
    if max_features > X.shape[1] or type(seed) is not int or seed < 0:
        raise ValueError('invalid feature budget or seed')
    rng = np.random.default_rng(seed); nodes = []
    def grow(indices, depth):
        node_id = len(nodes); labels = y[indices]; n1 = int(labels.sum()); count = len(indices)
        node = dict(id=node_id, depth=depth, counts=[count-n1,n1], prediction=int(2*n1>count), fraction_one=n1/count, leaf=True)
        nodes.append(node)
        if n1 in (0,count) or count < 2*min_leaf or max_depth is not None and depth >= max_depth:
            return node_id
        features = sorted(map(int, rng.choice(X.shape[1], max_features, replace=False)))
        node['searched_features'] = features
        best = split(X[indices], labels, features, min_leaf)
        if best is None:
            return node_id
        node.update(best); node['leaf'] = False
        mask = X[indices, best['feature']] <= best['cut']
        node['left'] = grow(indices[mask], depth+1)
        node['right'] = grow(indices[~mask], depth+1)
        return node_id
    grow(np.arange(len(y)), 0)
    return dict(n_features=X.shape[1], max_depth=max_depth, min_leaf=min_leaf, max_features=max_features, seed=seed, nodes=nodes)


def trace(model, x):
    x = np.asarray(x, dtype=float)
    if x.shape != (model['n_features'],) or not np.isfinite(x).all():
        raise ValueError('one finite full-width query required')
    path = []; node = model['nodes'][0]
    while not node['leaf']:
        left = bool(x[node['feature']] <= node['cut'])
        path.append(dict(node=node['id'], feature=node['feature'], cut=node['cut'], value=float(x[node['feature']]), branch='left' if left else 'right'))
        node = model['nodes'][node['left'] if left else node['right']]
    return dict(path=path, leaf=node['id'], counts=node['counts'], fraction_one=node['fraction_one'], prediction=node['prediction'])


def predict_tree(model, X):
    X = design(X)
    return np.array([trace(model,x)['prediction'] for x in X], dtype=int)


def feature_digest(X):
    return hashlib.sha256(json.dumps(np.asarray(X,dtype=float).tolist(),separators=(',',':'),allow_nan=False).encode()).hexdigest()


def forest(X, y, *, n_trees=31, seed=0, max_depth=None, min_leaf=2, max_features=2):
    X, y = design(X,y); positive(n_trees,'n_trees')
    if type(seed) is not int or seed < 0:
        raise ValueError('nonnegative integer seed required')
    # Bootstrap and node RNGs are separate; paired feature/depth controls share
    # the same bootstrap draws even when their tree shapes differ.
    rng = np.random.default_rng(seed); trees = []; inbag = []
    for t in range(n_trees):
        indices = rng.integers(0,len(y),len(y)); counts = np.bincount(indices,minlength=len(y))
        inbag.append(counts.tolist())
        trees.append(tree(X[indices],y[indices],max_depth=max_depth,min_leaf=min_leaf,max_features=max_features,seed=seed+1000003*(t+1)))
    return dict(training_feature_sha256=feature_digest(X), n_features=X.shape[1], n_train=len(y), n_trees=n_trees, seed=seed, max_depth=max_depth, min_leaf=min_leaf, max_features=max_features, inbag=inbag, trees=trees)


def votes(model,X):
    X = design(X)
    if X.shape[1] != model['n_features']:
        raise ValueError('query feature width mismatch')
    matrix = np.array([predict_tree(t,X) for t in model['trees']],dtype=int)
    fraction = matrix.mean(axis=0)
    return dict(votes=matrix.T.tolist(),fraction_one=fraction.tolist(),prediction=(fraction>.5).astype(int).tolist())


def oob(model,X):
    X = design(X)
    if X.shape != (model['n_train'],model['n_features']) or feature_digest(X) != model['training_feature_sha256']:
        raise ValueError('the exact training feature order is required for OOB')
    # Callers must also bind X to the stored training digest/IDs in the report.
    full = np.array(votes(model,X)['votes']); bag = np.array(model['inbag']).T
    counts = (bag==0).sum(axis=1); sums = (full*(bag==0)).sum(axis=1)
    return [dict(trees=[int(t) for t in np.flatnonzero(bag[i]==0)],count=int(counts[i]),fraction_one=(float(sums[i]/counts[i]) if counts[i] else None),prediction=(int(2*sums[i]>counts[i]) if counts[i] else None)) for i in range(len(X))]


def metrics(y, fraction):
    y = np.asarray(y); p = np.asarray(fraction,dtype=float)
    if y.shape != p.shape or not len(y) or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('binary labels and finite bounded vote fractions required')
    pred = p>.5
    return dict(n=len(y),errors=int((pred!=y).sum()),error=float(np.mean(pred!=y)),brier=float(np.mean((p-y)**2)),confusion=[[int(np.sum((y==a)&(pred==b))) for b in (0,1)] for a in (0,1)])
