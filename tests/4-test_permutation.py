import copy,itertools,json,subprocess,sys,tempfile,unittest
from fractions import Fraction
from pathlib import Path
import numpy as np
from forestlab import permutation as p
from forestlab.experiment import saved
from forestlab.core import votes

class PermutationTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.r=saved();cls.a=p.saved()
 def test_full_fresh_diagnostic_and_original_fit_unchanged(self):
  before=copy.deepcopy(self.r);out=p.compute(self.r);self.assertEqual(out,self.a);self.assertEqual(self.r,before);self.assertEqual(p.audit(out,self.r)['model_row_evaluations'],129600)
 def test_batch_routing_matches_scalar_all_original_rows(self):
  X=[g['x'] for g in self.r['data']['groups']];m=p.chosen(self.r)[0]['model'];self.assertEqual(p.forest_counts(m,X).tolist(),[sum(v) for v in votes(m,X)['votes']])
 def test_bijections_and_shared_masking_all240(self):
  X=np.arange(1080,dtype=float).reshape(270,4);self.assertEqual(self.a['donor_maps'],p.permutation(270));before=X.copy()
  for b in self.a['blocks']:
   for donor in self.a['donor_maps']:
    z=p.hybrid(X,b['features'],donor)
    for j in range(4):np.testing.assert_array_equal(z[:,j],X[donor,j] if j in b['features'] else X[:,j])
  np.testing.assert_array_equal(X,before)
 def test_exhaustive_three_row_oracle_and_full_block_support(self):
  X=np.array([[0.,0.],[1.,1.],[2.,2.]])
  for block in [[0],[1],[0,1]]:
   errors=[]
   for donor in itertools.permutations(range(3)):
    z=p.hybrid(X,block,list(donor));want=[[X[donor[i],j] if j in block else X[i,j] for j in range(2)] for i in range(3)]
    self.assertEqual(z.tolist(),want);errors.append(sum(a!=b for a,b in z.tolist()))
   mean=sum(map(Fraction,errors))/6
   self.assertEqual(mean,Fraction(0) if len(block)==2 else Fraction(2))
 def test_full_block_permutation_score_not_pointwise_invariant(self):
  b=self.a['blocks'][-1];self.assertTrue(all(r['rows_in_validation_support']==270 for r in b['runs']));self.assertGreater(min(r['error_delta'][0] for r in b['runs']),0)
  for run,donor in zip(b['runs'],self.a['donor_maps']):
   for m in range(2):self.assertEqual(run['counts'][m],[self.a['baseline']['counts'][m][j] for j in donor])
 def test_negative_repeats_kept_and_not_clipped(self):
  b=self.a['blocks'][7];self.assertLess(min(r['error_delta'][0] for r in b['runs']),0);self.assertEqual(len(b['runs']),16)
 def test_test_and_train_labels_not_read_by_diagnostic(self):
  changed=copy.deepcopy(self.r)
  for part in ['train','test']:
   for i in changed['data']['split'][part]:changed['data']['groups'][i]['y']=1-changed['data']['groups'][i]['y']
  self.assertEqual(p.compute(changed),self.a)
 def test_mutations_rejected(self):
  for change in [lambda a:a['donor_maps'][0].__setitem__(0,a['donor_maps'][0][1]),lambda a:a['blocks'][0]['runs'][0]['counts'][0].__setitem__(0,999),lambda a:a.__setitem__('model_sha256','wrong')]:
   a=copy.deepcopy(self.a);change(a)
   with self.assertRaises(ValueError):p.audit(a,self.r)
 def test_strict_validation_and_identity(self):
  for block,donor in [([], [0,1]),([0,0],[0,1]),([True],[0,1]),([2],[0,1]),([0],[1,1]),([0],[False,1]),([0],[0])]:
   with self.assertRaises(ValueError):p.hybrid([[0,1],[2,3]],block,donor)
  for args in [(1,16,0),(2,0,0),(2,129,0),(2,1,-1),(2,True,0)]:
   with self.assertRaises(ValueError):p.permutation(*args)
  for a,d in [([1.5,1],2),([3,1],2),([True,False],2),([1,0],0)]:
   with self.assertRaises(ValueError):p.summarize(a,d,[0,1])
  np.testing.assert_array_equal(p.hybrid([[0,1],[2,3]],[0],[0,1]),[[0,1],[2,3]])
 def test_all_metric_counts_exact_fraction_brier(self):
  y=[self.r['data']['groups'][i]['y'] for i in self.a['validation_ids']]
  for b in self.a['blocks']:
   for run in b['runs']:
    for j,d in enumerate(self.a['denominators']):
     counts=run['counts'][j];err=sum((2*c>d)!=v for c,v in zip(counts,y));brier=sum((Fraction(c,d)-v)**2 for c,v in zip(counts,y))/len(y)
     self.assertEqual(err,run['metrics'][j]['errors']);self.assertAlmostEqual(float(brier),run['metrics'][j]['brier'],places=14)

 def test_cli_fresh_audit_export_and_conflicts(self):
  def command(*args):return subprocess.run([sys.executable,'-m','forestlab','permutation',*args],capture_output=True,text=True)
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'fresh.json';out=command('--fresh','--output',str(path));self.assertEqual(out.returncode,0,out.stderr);self.assertEqual(json.loads(path.read_text()),self.a)
   out=command('--input',str(path),'--audit');self.assertEqual(out.returncode,0,out.stderr);self.assertEqual(json.loads(out.stdout)['model_row_evaluations'],129600)
   root=Path(tmp)/'web';out=command('--export',str(root));self.assertEqual(out.returncode,0,out.stderr)
   for name in ['permutation-data.js','permutation-archive-0.js']:self.assertEqual((root/name).read_bytes(),(Path(__file__).resolve().parents[1]/'docs'/name).read_bytes())
   for args in [('--fresh','--input',str(path)),('--input',str(path)),('--fresh','--export',str(root)),('--audit','--export',str(root))]:self.assertEqual(command(*args).returncode,2)
