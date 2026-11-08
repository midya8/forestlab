import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from forestlab.experiment import saved,equivalent
from forestlab.export import export

class CLITests(unittest.TestCase):
    def command(self,*args,check=True):
        return subprocess.run([sys.executable,'-m','forestlab',*args],capture_output=True,text=True,check=check)
    def test_report_audit_and_witness(self):
        r=json.loads(self.command('report').stdout);self.assertEqual(len(r['models']),19);self.assertEqual(len(json.loads(self.command('audit').stdout)['forests']),8)
        w=json.loads(self.command('witness','--group','0','--tree','3').stdout);self.assertEqual(w['group']['id'],0);self.assertEqual(w['tree'],3);self.assertIn(w['witness']['prediction'],[0,1])
    def test_fit_and_export_reproduction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fresh.json';self.command('fit','--output',str(path));self.assertTrue(equivalent(json.loads(path.read_text()),saved()));self.command('audit','--report',str(path))
            root=Path(tmp)/'docs';summary=json.loads(self.command('export','--output',str(root)).stdout);self.assertEqual(summary['group_evaluations'],10784);base=Path(__file__).resolve().parents[1]/'docs'
            for name in ['data.js','archive-0.js','archive-1.js','archive-2.js','archive-3.js']:self.assertEqual((root/name).read_bytes(),(base/name).read_bytes())
            actual=json.loads((root/'reference.json').read_text());expected=json.loads((base/'reference.json').read_text())
            self.assertEqual(actual.keys(),expected.keys());self.assertEqual(actual['predictions'],expected['predictions']);self.assertEqual(actual['paths'],expected['paths'])
            # Only native logistic/control arithmetic may vary in last bits.
            self.assertTrue(equivalent(actual['controls'],expected['controls']))
    def test_predict_and_strict_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'input.json';path.write_text(json.dumps({'features':[[0,0,0,0]]}));r=json.loads(self.command('predict','--input',str(path)).stdout);self.assertEqual(len(r['forest']['votes'][0]),31)
            for obj in [{'features':[]},{'features':[[True,0,0,0]]},{'features':[['1',0,0,0]]},{'features':[[float('nan'),0,0,0]]},{'features':[[0,0,0]]},{'features':[[0,0,0,0]],'extra':1}]:
                path.write_text(json.dumps(obj));self.assertNotEqual(self.command('predict','--input',str(path),check=False).returncode,0)
            self.assertNotEqual(self.command('witness','--tree','31',check=False).returncode,0)
    def test_cross_platform_comparison_is_not_vacuous(self):
        self.assertTrue(equivalent({'a':[1,.2]},{'a':[1,.2+1e-13]}));self.assertFalse(equivalent({'a':[1,.2]},{'a':[1,.2+1e-5]}));self.assertFalse(equivalent({'a':[1,float('nan')]},{'a':[1,float('nan')]}));self.assertFalse(equivalent({'a':1},{'a':True}));self.assertFalse(equivalent({'a':1},{'a':1.}))
if __name__=='__main__':unittest.main()
