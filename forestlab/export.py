"""Export the full frozen report and native traversal references, no learned JS."""
import json
from importlib.resources import files
from pathlib import Path
from .experiment import saved
from .core import votes,trace
from .controls import scores

def export(root):
    r=saved();root=Path(root);root.mkdir(parents=True,exist_ok=True)
    # Forward the source-checked archive exactly; never recompress it with a
    # potentially different platform zlib build during export.
    packed=files('forestlab').joinpath('artifacts/report.json').read_text().strip()
    for i,name in enumerate(json.loads(packed)['parts']):
        chunk=files('forestlab').joinpath('artifacts/'+name).read_text().strip()
        (root/('archive-'+str(i)+'.js')).write_text('globalThis.FOREST_ARCHIVE_PARTS=globalThis.FOREST_ARCHIVE_PARTS||[];globalThis.FOREST_ARCHIVE_PARTS.push('+json.dumps(chunk)+');\n')
    text='globalThis.FOREST_ARCHIVE='+packed+";\nglobalThis.FOREST_READY=(async()=>{const p=globalThis.FOREST_ARCHIVE;const bytes=Uint8Array.from(atob(globalThis.FOREST_ARCHIVE_PARTS.join('')),c=>c.charCodeAt(0));const decoded=new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',decoded))).map(v=>v.toString(16).padStart(2,'0')).join('');if(digest!==p.sha256)throw Error('Report digest mismatch');globalThis.FOREST_DATA=JSON.parse(new TextDecoder().decode(decoded));return globalThis.FOREST_DATA;})();\n"
    (root/'data.js').write_bytes(text.replace('\n','\r\n').encode())
    X=[g['x'] for g in r['data']['groups']];pred=[]
    for c in r['candidates']:
        out=votes(c['model'],X);pred.append(dict(id=c['id'],fraction_one=out['fraction_one'],prediction=out['prediction']))
    selected=next(c for c in r['candidates'] if c['id']==r['selected_id'])['model']
    paths=[dict(group=i,tree=t,**trace(model,X[i])) for i in r['data']['split']['test'][:32] for t,model in enumerate(selected['trees'])]
    controls=[dict(id=c['id'],scores=scores(c['model'],X).tolist()) for c in r['controls'] if c['model']['kind']!='constant']
    refs=dict(predictions=pred,paths=paths,controls=controls)
    (root/'reference.json').write_text(json.dumps(refs,separators=(',',':'),allow_nan=False)+'\n')
    return dict(forests=len(pred),group_evaluations=len(pred)*len(X),paths=len(paths))
