import copy,json,math,unittest
import numpy as np
from oracle import best
from forestlab.core import split,tree,trace,predict_tree,forest,votes,oob,metrics,feature_digest

class CoreTests(unittest.TestCase):
    def test_sixty_exact_split_oracles(self):
        rng=np.random.default_rng(41211)
        for k in range(60):
            X=rng.integers(-3,4,(5+k%8,1+k%4)).tolist();y=rng.integers(0,2,len(X)).tolist();features=list(range(len(X[0])));leaf=1+k%3
            ref=best(X,y,features,leaf);actual=split(X,y,features,leaf)
            if ref is None:self.assertIsNone(actual)
            else:
                self.assertEqual((actual['feature'],actual['cut']),ref[1:]);self.assertAlmostEqual(actual['gain'],2*sum(y)*(len(y)-sum(y))/len(y)**2-float(ref[0]))
    def test_cuts_lower_observed_and_feature_ties(self):
        X=[[0,0],[2,2],[4,4],[8,8]];y=[0,0,1,1]
        m=tree(X,y);self.assertEqual(m['nodes'][0]['feature'],0);self.assertEqual(m['nodes'][0]['cut'],2)
        self.assertEqual(trace(m,[2,99])['prediction'],0);self.assertEqual(trace(m,[math.nextafter(2,math.inf),99])['prediction'],1)
        # A midpoint tree would classify3 as0; lower-observed cuts classify1.
        self.assertEqual(trace(m,[3,99])['prediction'],1)
    def test_no_positive_gain_is_leaf_and_tie_is_zero(self):
        m=tree([[1],[1]],[0,1]);self.assertTrue(m['nodes'][0]['leaf']);self.assertEqual(trace(m,[1])['prediction'],0)
        m=tree([[0],[1],[2],[3]],[0,1,0,1],min_leaf=2);self.assertTrue(m['nodes'][0]['leaf'])
    def test_recursive_partition_counts_and_leaf_constraints(self):
        rng=np.random.default_rng(23);X=rng.normal(size=(80,3));y=(X[:,0]*X[:,1]>0).astype(int);m=tree(X,y,max_depth=4,min_leaf=3)
        def audit(node_id,indices):
            node=m['nodes'][node_id];labels=y[indices];self.assertEqual(node['counts'],[int((labels==0).sum()),int(labels.sum())]);self.assertEqual(node['prediction'],int(labels.mean()>.5))
            if node['leaf']:return
            selected=X[indices,node['feature']]<=node['cut'];self.assertGreaterEqual(min(int(selected.sum()),int((~selected).sum())),3)
            ref=best(X[indices].tolist(),labels.tolist(),node['searched_features'],3);self.assertEqual((node['feature'],node['cut']),ref[1:]);audit(node['left'],indices[selected]);audit(node['right'],indices[~selected])
        audit(0,np.arange(len(y)));self.assertLessEqual(max(n['depth'] for n in m['nodes']),4)
    def test_bootstrap_exact_replay_and_paired_controls(self):
        X=np.arange(30).reshape((15,2));y=np.arange(15)%2
        a=forest(X,y,n_trees=5,seed=3,max_features=1);b=forest(X,y,n_trees=5,seed=3,max_features=2,max_depth=2)
        self.assertEqual(a['inbag'],b['inbag']);self.assertEqual(a,forest(X,y,n_trees=5,seed=3,max_features=1));self.assertEqual(a['training_feature_sha256'],feature_digest(X))
        rng=np.random.default_rng(3)
        for bag in a['inbag']:
            self.assertEqual(bag,np.bincount(rng.integers(0,len(y),len(y)),minlength=len(y)).tolist());self.assertEqual(sum(bag),len(y))
    def test_oob_exclusion_and_zero_votes(self):
        X=np.arange(30).reshape((15,2));y=np.arange(15)%2;m=forest(X,y,n_trees=5,seed=11,max_features=2)
        full=votes(m,X);result=oob(m,X)
        for i,r in enumerate(result):
            subset=[t for t in range(5) if not m['inbag'][t][i]];self.assertEqual(r['trees'],subset);self.assertEqual(r['count'],len(subset))
            if subset:self.assertEqual(r['fraction_one'],sum(full['votes'][i][t] for t in subset)/len(subset))
            else:self.assertIsNone(r['fraction_one']);self.assertIsNone(r['prediction'])
        single=forest([[0]],[1],n_trees=1,max_features=1);self.assertEqual(oob(single,[[0]])[0]['count'],0)
        for altered in [X[::-1],X+1]:
            with self.assertRaises(ValueError):oob(m,altered)
    def test_json_roundtrip_and_vote_aggregation(self):
        X=[[0],[1],[2],[3]];y=[0,1,1,0];m=forest(X,y,n_trees=7,max_features=1);a=votes(m,X);self.assertEqual(a,votes(json.loads(json.dumps(m)),X))
        for v,p,label in zip(a['votes'],a['fraction_one'],a['prediction']):self.assertEqual(p,sum(v)/7);self.assertEqual(label,int(2*sum(v)>7))
    def test_metrics_confusion(self):
        m=metrics([0,0,1,1],[.1,.8,.5,.9]);self.assertEqual(m['errors'],2);self.assertEqual(m['confusion'],[[1,1],[1,1]]);self.assertAlmostEqual(m['brier'],(.01+.64+.25+.01)/4)
    def test_invalid_inputs(self):
        for X,y in [([],[]),([[float('nan')]],[1]),([[1],[2]],[1]),([[1]],[2])]:
            with self.assertRaises(ValueError):tree(X,y)
        for kwargs in [dict(min_leaf=0),dict(min_leaf=True),dict(max_depth=0),dict(max_features=0),dict(max_features=2),dict(seed=-1)]:
            with self.assertRaises(ValueError):tree([[0],[1]],[0,1],**kwargs)
        with self.assertRaises(ValueError):trace(tree([[0],[1]],[0,1]),[1,2])
        with self.assertRaises(ValueError):split([[0],[1]],[0,1],[0,0],1)
        with self.assertRaises(ValueError):metrics([0,1],[.2,float('nan')])
if __name__=='__main__':unittest.main()
