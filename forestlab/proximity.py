"""Frozen forest leaf collisions. Indexed OOB is not a kernel guarantee."""
import copy
import numpy as np
from .core import trace, design, feature_digest


def leaf_codes(model, features):
    X = design(features)
    if X.shape[1] != model['n_features'] or model['n_trees'] != len(model['trees']) or not model['trees']:
        raise ValueError('matching nonempty forest and feature width required')
    return [[trace(tree, row)['leaf'] for tree in model['trees']] for row in X]


def _codes(codes):
    if not isinstance(codes,list) or not codes or not isinstance(codes[0],list) or not codes[0]:
        raise ValueError('nonempty row by tree integer leaf codes required')
    if any(not isinstance(row,list) or len(row)!=len(codes[0]) or any(type(v) is not int or v<0 for v in row) for row in codes):
        raise ValueError('rectangular nonnegative integer leaf codes required')
    return codes


def geometry(codes, oob=None):
    """Integer counts first. None means undefined, never zero-filled OOB."""
    _codes(codes); n=len(codes); trees=len(codes[0])
    if oob is not None and (not isinstance(oob,list) or len(oob)!=n or any(row is not None and (not isinstance(row,list) or len(row)!=trees or any(type(v) is not bool for v in row)) for row in oob)):
        raise ValueError('one Boolean OOB tree mask or null per row required')
    counts=[[sum(a==b for a,b in zip(left,right)) for right in codes] for left in codes]
    result=dict(leaf_codes=copy.deepcopy(codes),trees=trees,collision_counts=counts,proximity=[[v/trees for v in row] for row in counts],distance_squared=[[2*(trees-v)/trees for v in row] for row in counts])
    if oob is not None:
        den=[];num=[];values=[]
        for i in range(n):
            d=[];s=[];p=[]
            for j in range(n):
                known=oob[i] is not None and oob[j] is not None
                eligible=[t for t in range(trees) if oob[i][t] and oob[j][t]] if known else []
                count=sum(codes[i][t]==codes[j][t] for t in eligible)
                d.append(len(eligible) if known else None);s.append(count if known else None);p.append(count/len(eligible) if eligible else None)
            den.append(d);num.append(s);values.append(p)
        result.update(oob_masks=copy.deepcopy(oob),oob_denominators=den,oob_collision_counts=num,oob_proximity=values)
    return result


def quadratic(codes, coefficients):
    """Exact integer quadratic numerator and leaf-bucket sum-of-squares."""
    _codes(codes)
    if not isinstance(coefficients,list) or len(coefficients)!=len(codes) or any(type(v) is not int for v in coefficients):
        raise ValueError('one integer coefficient per row required')
    counts=geometry(codes)['collision_counts'];direct=sum(coefficients[i]*coefficients[j]*counts[i][j] for i in range(len(codes)) for j in range(len(codes)))
    buckets=[]
    for t in range(len(codes[0])):
        leafs={}
        for row,c in zip(codes,coefficients):leafs[row[t]]=leafs.get(row[t],0)+c
        buckets.append([dict(leaf=leaf,coefficient_sum=value,square=value*value) for leaf,value in sorted(leafs.items())])
    square_sum=sum(b['square'] for row in buckets for b in row)
    if square_sum!=direct:raise ArithmeticError('Gram identity failed')
    return dict(coefficients=[str(v) for v in coefficients],numerator=str(direct),denominator=str(len(codes[0])),buckets=[[dict(leaf=b['leaf'],coefficient_sum=str(b['coefficient_sum']),square=str(b['square'])) for b in row] for row in buckets],sum_of_squares=str(square_sum))


def indexed(model,data,groups):
    if not isinstance(groups,list) or not 1<=len(groups)<=64 or any(type(i) is not int or not 0<=i<len(data['groups']) for i in groups) or len(set(groups))!=len(groups):
        raise ValueError('one to64 distinct known integer source groups required')
    train=data['split']['train'];X=[data['groups'][i]['x'] for i in train]
    if len(train)!=model['n_train'] or feature_digest(X)!=model['training_feature_sha256']:
        raise ValueError('exact stored training feature order required for OOB binding')
    if len(model['inbag'])!=model['n_trees'] or any(len(row)!=len(train) or any(type(v) is not int or v<0 for v in row) or sum(row)!=len(train) for row in model['inbag']):
        raise ValueError('complete bootstrap multiplicities required')
    locations={i:j for j,i in enumerate(train)}
    masks=[[bag[locations[i]]==0 for bag in model['inbag']] if i in locations else None for i in groups]
    out=geometry(leaf_codes(model,[data['groups'][i]['x'] for i in groups]),masks)
    out.update(groups=list(groups),features=[list(data['groups'][i]['x']) for i in groups],training_local_indices=[locations.get(i) for i in groups])
    return out


def pair(model,left,right):
    # Feature-only inputs deliberately have no source-row identity or OOB result.
    if any(not isinstance(row,list) or len(row)!=model['n_features'] or any(type(v) not in (int,float) for v in row) for row in [left,right]):raise ValueError('two finite numeric feature rows required')
    out=geometry(leaf_codes(model,[left,right]));out['features']=[list(left),list(right)];out['identity']='feature-only; no OOB identity'
    return out


def controls():
    codes=[[0,1,0],[0,0,1],[1,0,2]];masks=[[True,False,True],[True,True,False],[False,True,True]]
    g=geometry(codes,masks);v=[1,-2,1]
    # Every pair is jointly OOB once; diagonal twice. Thus all entries are exact integers.
    q=sum(v[i]*v[j]*g['oob_proximity'][i][j] for i in range(3) for j in range(3))
    return dict(scope='Constructed partition/eligibility tables, not a fitted-data result',indefinite=dict(**g,coefficients=v,oob_quadratic=int(q),all_tree_quadratic=quadratic(codes,v)),undefined=geometry([[0,0],[0,1]],[[True,False],[False,True]]),collision=geometry([[0,2],[0,2]]))
