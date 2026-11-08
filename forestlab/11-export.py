"""Export the full frozen report and native traversal references, no learned JS."""
import json
from pathlib import Path
from .experiment import saved
from .core import votes,trace
from .controls import scores

def export(root,report=None):
    r=saved() if report is None else report;root=Path(root);root.mkdir(parents=True,exist_ok=True)
    text='globalThis.FOREST_DATA='+json.dumps(r,separators=(',',':'),allow_nan=False)+';\n'
    (root/'data.js').write_text(text)
    X=[g['x'] for g in r['data']['groups']];pred=[]
    for c in r['candidates']:
        out=votes(c['model'],X);pred.append(dict(id=c['id'],fraction_one=out['fraction_one'],prediction=out['prediction']))
    selected=next(c for c in r['candidates'] if c['id']==r['selected_id'])['model']
    paths=[dict(group=i,tree=t,**trace(model,X[i])) for i in r['data']['split']['test'][:32] for t,model in enumerate(selected['trees'])]
    controls=[dict(id=c['id'],scores=scores(c['model'],X).tolist()) for c in r['controls'] if c['model']['kind']!='constant']
    refs=dict(predictions=pred,paths=paths,controls=controls)
    (root/'reference.json').write_text(json.dumps(refs,separators=(',',':'),allow_nan=False)+'\n')
    return dict(forests=len(pred),group_evaluations=len(pred)*len(X),paths=len(paths))
