"""Independent small split search, using complete masks and Fraction Gini."""
from fractions import Fraction as F

def best(X,y,features,min_leaf):
    n=len(y);parent=F(2*sum(y)*(n-sum(y)),n*n);candidates=[]
    for feature in sorted(features):
        values=sorted(set(row[feature] for row in X))
        for cut in values[:-1]:
            left=[label for row,label in zip(X,y) if row[feature]<=cut]
            right=[label for row,label in zip(X,y) if row[feature]>cut]
            if min(len(left),len(right))<min_leaf:continue
            cost=sum((F(2*sum(z)*(len(z)-sum(z)),len(z)*n) for z in [left,right]),F(0))
            if cost<parent:candidates.append((cost,feature,cut))
    return min(candidates) if candidates else None
