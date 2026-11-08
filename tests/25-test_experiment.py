import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from forestlab.dataset import load,arrays,SHA256
from forestlab.experiment import run,saved,audit,equivalent
from forestlab.controls import logistic,logistic_scores,nearest,nearest_scores,sigmoid
from forestlab.core import votes,trace

class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=load();cls.report=saved()
    def test_original_bytes_groups_and_disjoint_splits(self):
        d=self.data;self.assertEqual(d['source_sha256'],SHA256);self.assertEqual(len(d['groups']),1348);self.assertEqual(sorted(r for g in d['groups'] for r in g['source_rows']),list(range(1372)))
        a,b,c=map(set,[d['split'][k] for k in ['train','validation','test']]);self.assertFalse(a&b or a&c or b&c);self.assertEqual(len(a|b|c),1348);self.assertEqual([len(a),len(b),len(c)],[808,270,270])
        self.assertEqual(len({tuple(g['x']) for g in d['groups']}),1348);self.assertEqual(sum(len(g['source_rows'])-1 for g in d['groups']),24)
    def test_all_fresh_fits_and_full_audit(self):
        fresh=run();self.assertTrue(equivalent(fresh,self.report));result=audit(fresh);self.assertEqual(len(result['forests']),8);self.assertEqual(result['controls'],7);self.assertEqual(result['singles'],4)
    def test_test_poison_does_not_change_training_or_selection(self):
        poisoned=copy.deepcopy(self.data)
        for group in poisoned['split']['test']:poisoned['groups'][group]['y']=1-poisoned['groups'][group]['y']
        other=run(poisoned)
        for name in ['selected_id','selected_control_id','selected_single_id']:self.assertEqual(other[name],self.report[name])
        for category in ['candidates','controls','singles']:
            for a,b in zip(other[category],self.report[category]):
                for k in a:
                    if k!='test':self.assertTrue(equivalent(a[k],b[k]),(a['id'],k))
    def test_stronger_nearest_control_retained(self):
        r=self.report;chosen=next(c for c in r['candidates'] if c['id']==r['selected_id']);control=next(c for c in r['controls'] if c['id']==r['selected_control_id']);self.assertEqual(control['id'],'nearest-5');self.assertEqual(control['test']['errors'],0);self.assertEqual(chosen['test']['errors'],3);self.assertLess(control['validation']['errors'],chosen['validation']['errors'])
        self.assertTrue(all(c['oob_missing']==0 for c in r['candidates']));self.assertTrue(all(c['model']['reason']=='gradient_tolerance' for c in r['controls'] if c['model']['kind']=='logistic'))
    def test_training_only_preprocessing_and_logistic_stationarity(self):
        X,y=arrays(self.data,'train');mean=X.mean(axis=0);scale=X.std(axis=0);A=np.column_stack([np.ones(len(y)),(X-mean)/scale])
        for c in self.report['controls']:
            m=c['model']
            if m['kind']=='constant':self.assertEqual(m['fraction_one'],float(y.mean()));continue
            self.assertEqual(m['mean'],mean.tolist());self.assertEqual(m['scale'],scale.tolist())
            if m['kind']=='nearest':self.assertEqual(m['train'],((X-mean)/scale).tolist());self.assertEqual(m['labels'],y.tolist());continue
            t=np.array(m['theta']);p=sigmoid(A@t);pen=np.array([0]+[m['regularization']]*4);grad=A.T@(p-y)/len(y)+pen*t
            self.assertLess(np.max(np.abs(grad)),1e-9);self.assertTrue(all(a['objective']>=b['objective']-1e-14 for a,b in zip(m['history'],m['history'][1:])))
            def f(z):return np.mean(np.logaddexp(0,A@z)-y*(A@z))+.5*np.dot(pen*z,z)
            probe=t+np.array([.2,-.1,.3,.2,-.3]);g=A.T@(sigmoid(A@probe)-y)/len(y)+pen*probe
            for j in range(5):
                plus=probe.copy();minus=probe.copy();plus[j]+=1e-5;minus[j]-=1e-5;self.assertAlmostEqual((f(plus)-f(minus))/2e-5,g[j],places=8)
    def test_independent_knn_and_ties(self):
        m=nearest([[0],[2],[4]],[1,0,1],1);self.assertEqual(nearest_scores(m,[[1]])[0],1)
        for k in [1,5]:
            X,y=arrays(self.data,'train');m=nearest(X,y,k);query=np.array(self.data['groups'][self.data['split']['test'][0]]['x']);z=(query-np.array(m['mean']))/np.array(m['scale']);dist=[(sum((a-b)**2 for a,b in zip(row,z)),i) for i,row in enumerate(m['train'])];ids=[i for _,i in sorted(dist)[:k]];self.assertEqual(nearest_scores(m,[query])[0],sum(int(y[i]) for i in ids)/k)
    def test_mutations_rejected(self):
        cases=[]
        a=copy.deepcopy(self.report);a['data']['split']['test'][0]=a['data']['split']['train'][0];cases.append(a)
        a=copy.deepcopy(self.report);a['candidates'][0]['model']['inbag'][0][0]+=1;cases.append(a)
        a=copy.deepcopy(self.report);a['candidates'][0]['model']['trees'][0]['nodes'][0]['gain']+=.1;cases.append(a)
        a=copy.deepcopy(self.report);a['candidates'][0]['test']['errors']+=1;cases.append(a)
        a=copy.deepcopy(self.report);a['selected_id']='changed';cases.append(a)
        a=copy.deepcopy(self.report);a['candidates'][0]['model']['max_features']=3;cases.append(a)
        for changed in cases:
            with self.assertRaises(ValueError):audit(changed)
if __name__=='__main__':unittest.main()
