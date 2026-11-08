import copy,itertools,json,subprocess,sys,tempfile,unittest
from fractions import Fraction
from pathlib import Path
import numpy as np
from forestlab import proximity as p
from forestlab.experiment import saved
from forestlab.proximity_export import build,export

class ProximityTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.r=saved();cls.data=cls.r['data']
 def test_all_partitions_small_exact_oracle_and_gram(self):
  # Exhaust every binary partition assignment of 3 points in3 trees.
  for flat in itertools.product(range(2),repeat=9):
   codes=[list(flat[i:i+3]) for i in range(0,9,3)];a=p.geometry(codes);B=np.zeros((3,6),dtype=int)
   for i,row in enumerate(codes):
    for t,leaf in enumerate(row):B[i,t*2+leaf]=1
   self.assertEqual(a['collision_counts'],(B@B.T).tolist())
   q=p.quadratic(codes,[1,-2,3]);want=sum(Fraction(c[i])*Fraction(c[j])*Fraction(a['collision_counts'][i][j],3) for c in [[1,-2,3]] for i in range(3) for j in range(3));self.assertEqual(Fraction(int(q['numerator']),int(q['denominator'])),want);self.assertGreaterEqual(int(q['numerator']),0)
 def test_exact_indefinite_and_undefined_controls(self):
  a=p.controls();g=a['indefinite'];self.assertEqual(g['oob_proximity'],[[1,1,0],[1,1,1],[0,1,1]]);self.assertEqual(g['oob_quadratic'],-2);self.assertEqual(Fraction(int(g['all_tree_quadratic']['numerator']),3),Fraction(10,3));self.assertIsNone(a['undefined']['oob_proximity'][0][1]);self.assertEqual(a['undefined']['oob_denominators'][0][1],0);self.assertEqual(a['collision']['distance_squared'][0][1],0)
 def test_all_eight_source_matrices_direct_paths_oob_and_no_labels(self):
  before=copy.deepcopy(self.r);ids=self.data['split']['train'][:24]+self.data['split']['validation'][:2]+self.data['split']['test'][:2]
  for c in self.r['candidates']:
   m=c['model'];g=p.indexed(m,self.data,ids)
   # Independent traversal and set intersections, no production trace/geometry.
   leaves=[];eligible=[]
   for i in ids:
    row=self.data['groups'][i]['x'];code=[]
    for tree in m['trees']:
     node=0
     while not tree['nodes'][node]['leaf']:
      n=tree['nodes'][node];node=n['left'] if row[n['feature']]<=n['cut'] else n['right']
     code.append(node)
    leaves.append(code);local=self.data['split']['train'].index(i) if i in self.data['split']['train'] else None
    eligible.append({t for t,b in enumerate(m['inbag']) if local is not None and b[local]==0} if local is not None else None)
   self.assertEqual(g['leaf_codes'],leaves)
   for i,j in itertools.product(range(len(ids)),repeat=2):
    collision={t for t in range(m['n_trees']) if leaves[i][t]==leaves[j][t]};self.assertEqual(g['collision_counts'][i][j],len(collision))
    if eligible[i] is None or eligible[j] is None:self.assertIsNone(g['oob_denominators'][i][j]);continue
    both=eligible[i]&eligible[j];self.assertEqual(g['oob_denominators'][i][j],len(both));self.assertEqual(g['oob_collision_counts'][i][j],len(both&collision))
    self.assertEqual(g['oob_proximity'][i][j],float(Fraction(len(both&collision),len(both))) if both else None)
  self.assertEqual(before,self.r);poisoned=copy.deepcopy(self.data)
  for row in poisoned['groups']:row['y']='not a label'
  self.assertEqual(p.indexed(self.r['candidates'][0]['model'],self.data,ids),p.indexed(self.r['candidates'][0]['model'],poisoned,ids))
 def test_binding_and_feature_identity_refusal(self):
  m=self.r['candidates'][0]['model'];d=copy.deepcopy(self.data);d['split']['train'][:2]=reversed(d['split']['train'][:2])
  with self.assertRaises(ValueError):p.indexed(m,d,[0,1])
  d=copy.deepcopy(self.data);d['groups'][d['split']['train'][0]]['x'][0]+=1
  with self.assertRaises(ValueError):p.indexed(m,d,[0,1])
  out=p.pair(m,self.data['groups'][0]['x'],self.data['groups'][0]['x']);self.assertNotIn('oob_proximity',out);self.assertEqual(out['proximity'][0][1],1)
 def test_order_relabel_duplicate_and_exact_large_integer(self):
  codes=[[0,4],[1,4],[0,9]];a=p.geometry(codes);b=p.geometry([[v+100 for v in row] for row in codes]);self.assertEqual(a['collision_counts'],b['collision_counts']);q=p.quadratic(codes,[10**35,-10**35,2]);self.assertEqual(q['numerator'],q['sum_of_squares']);self.assertGreater(int(q['numerator']),2**53)
  order=[2,0,1];self.assertEqual(p.geometry([codes[i] for i in order])['collision_counts'],[[a['collision_counts'][i][j] for j in order] for i in order])
 def test_refusals(self):
  for codes in [[],[[]],[[True]],[[1],[1,2]],[[1.2]],[[1],[-1]]]:
   with self.assertRaises(ValueError):p.geometry(codes)
  for mask in [[],[[0]],[[True,False]],[['yes']]]:
   with self.assertRaises(ValueError):p.geometry([[1]],mask)
  for ids in [[],[True],[0,0],[-1],[1348],[[0]]]:
   with self.assertRaises(ValueError):p.indexed(self.r['candidates'][0]['model'],self.data,ids)
  for row in [[True,0,0,0],['0',0,0,0],[float('nan'),0,0,0],[float('inf'),0,0,0]]:
   with self.assertRaises(ValueError):p.pair(self.r['candidates'][0]['model'],row,[0]*4)
 def test_complete_export_source_preservation(self):
  a=build();root=Path(__file__).resolve().parents[1]
  self.assertEqual(a,json.loads((root/'forestlab/artifacts/proximity-reference.json').read_text()))
  with tempfile.TemporaryDirectory() as tmp:
   self.assertEqual(export(tmp)['reference_pairs'],1152)
   for f in ['proximity-data.js','proximity-reference.json']:self.assertEqual((Path(tmp)/f).read_bytes(),(root/'docs'/f).read_bytes())
 def test_cli_source_feature_and_invalid(self):
  def run(*a):return subprocess.run([sys.executable,'-m','forestlab',*a],capture_output=True,text=True)
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'input.json'
   for obj in [{'groups':[0,1]},{'features':[[0]*4,[1]*4]}]:f.write_text(json.dumps(obj));self.assertEqual(run('proximity','--input',str(f)).returncode,0)
   for text in ['{"groups":[true]}','{"groups":[0],"groups":[1]}','{"features":[[true,0,0,0],[0,0,0,0]]}','{"groups":[0],"features":[]}']:
    f.write_text(text);self.assertEqual(run('proximity','--input',str(f)).returncode,2)
   self.assertEqual(run('proximity-controls').returncode,0);self.assertEqual(run('proximity-export','--output',d).returncode,0)
