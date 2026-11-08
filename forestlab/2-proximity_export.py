"""Complete small reference matrices without changing original saved reports."""
import json
from pathlib import Path
from .experiment import saved
from .proximity import indexed,quadratic,controls

def build():
    r=saved();data=r['data'];ids=data['split']['train'][:8]+data['split']['validation'][:2]+data['split']['test'][:2]
    refs=[]
    for c in r['candidates']:
        result=indexed(c['model'],data,ids);coefficients=[(-1 if i%2 else 1)*(i%3+1) for i in range(len(ids))]
        refs.append(dict(id=c['id'],geometry=result,quadratic=quadratic(result['leaf_codes'],coefficients)))
    return dict(binding=dict(train_ids=list(data['split']['train']),train_features=[data['groups'][i]['x'] for i in data['split']['train']],training_feature_sha256=r['candidates'][0]['model']['training_feature_sha256']),schema='forestlab-proximity-v1',scope='Frozen supervised partitions; not semantic similarity, causal influence or fresh performance',references=refs,heatmap_references=[dict(id=c['id'],geometry=indexed(c['model'],data,data['split']['train'][:24])) for c in r['candidates']],heatmap_groups=data['split']['train'][:24],controls=controls())

def export(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);text=json.dumps(build(),separators=(',',':'),allow_nan=False)+'\n'
    (root/'proximity-reference.json').write_text(text)
    (root/'proximity-data.js').write_text('globalThis.FOREST_PROXIMITY_DATA='+text.rstrip()+';\n')
    return dict(models=8,reference_rows=12,reference_pairs=1152,leaf_coordinates=2976,heatmap_rows=24,heatmap_pairs=4608,heatmap_leaf_coordinates=5952)
