"""Strong standardized controls. No external estimators or test-fit preprocessing."""
import numpy as np
from .core import design,positive

def standardize(X):
    X=design(X);mean=X.mean(axis=0);scale=X.std(axis=0);scale=np.where(scale>0,scale,1.)
    return mean,scale

def sigmoid(x):
    x=np.asarray(x,dtype=float);out=np.empty_like(x);p=x>=0;out[p]=1/(1+np.exp(-x[p]));z=np.exp(x[~p]);out[~p]=z/(1+z)
    return out

def logistic(X,y,regularization,max_steps=80,tolerance=1e-9):
    X,y=design(X,y)
    if not np.isfinite(regularization) or regularization<=0:raise ValueError('positive regularization required')
    mean,scale=standardize(X);A=np.column_stack([np.ones(len(y)),(X-mean)/scale]);theta=np.zeros(A.shape[1]);penalty=np.diag([0]+[regularization]*X.shape[1]);history=[]
    def objective(t):
        score=A@t
        return float(np.mean(np.logaddexp(0,score)-y*score)+.5*t@penalty@t)
    reason='budget_exhausted'
    for step in range(max_steps+1):
        p=sigmoid(A@theta);grad=A.T@(p-y)/len(y)+penalty@theta;norm=float(np.max(np.abs(grad)));value=objective(theta)
        history.append(dict(step=step,objective=value,gradient_inf=norm))
        if norm<=tolerance:reason='gradient_tolerance';break
        if step==max_steps:break
        H=(A.T*(p*(1-p)))@A/len(y)+penalty;direction=np.linalg.solve(H,grad);slope=float(grad@direction);rate=1.
        while rate>2**-40 and objective(theta-rate*direction)>value-1e-4*rate*slope:rate*=.5
        if rate<=2**-40:reason='line_search_stopped';break
        history[-1]['rate']=rate;theta-=rate*direction
    return dict(kind='logistic',regularization=regularization,mean=mean.tolist(),scale=scale.tolist(),theta=theta.tolist(),history=history,reason=reason)

def logistic_scores(model,X):
    X=design(X);Z=(X-np.array(model['mean']))/np.array(model['scale']);return sigmoid(model['theta'][0]+Z@np.array(model['theta'][1:]))

def nearest(X,y,k):
    X,y=design(X,y);positive(k,'k')
    if k>len(y):raise ValueError('k exceeds training size')
    mean,scale=standardize(X)
    return dict(kind='nearest',k=k,mean=mean.tolist(),scale=scale.tolist(),train=((X-mean)/scale).tolist(),labels=y.tolist())

def nearest_scores(model,X):
    X=design(X);Z=(X-np.array(model['mean']))/np.array(model['scale']);train=np.array(model['train']);y=np.array(model['labels'])
    return np.array([y[np.argsort(np.sum((train-z)**2,axis=1),kind='stable')[:model['k']]].mean() for z in Z])

def scores(model,X):
    if model['kind']=='logistic':return logistic_scores(model,X)
    if model['kind']=='nearest':return nearest_scores(model,X)
    if model['kind']=='constant':return np.full(len(X),model['fraction_one'])
    raise ValueError('unknown control')
