import argparse,json,sys
from .experiment import saved,run,save,audit
from .core import votes,trace
from .controls import scores
from .export import export

def main():
    p=argparse.ArgumentParser(description='Historical numeric classification forests, not banknote authentication or calibrated confidence.');s=p.add_subparsers(dest='command',required=True)
    s.add_parser('report');f=s.add_parser('fit');f.add_argument('--output',required=True);a=s.add_parser('audit');a.add_argument('--report');q=s.add_parser('predict');q.add_argument('--input',required=True);w=s.add_parser('witness');w.add_argument('--group',type=int,default=0);w.add_argument('--tree',type=int,default=0);d=s.add_parser('permutation');d.add_argument('--output');d.add_argument('--fresh',action='store_true');d.add_argument('--audit',action='store_true');d.add_argument('--input');d.add_argument('--export');e=s.add_parser('export');e.add_argument('--output',required=True);args=p.parse_args()
    try:
        r=run() if args.command=='fit' else saved()
        if args.command=='fit':save(r,args.output)
        if args.command in ['report','fit']:
            result={k:r[k] for k in ['scope','selected_id','selected_control_id','selected_single_id']}
            result['models']=[dict(id=c['id'],validation=c['validation'],test=c['test']) for category in ['candidates','controls','singles'] for c in r[category]]
        elif args.command=='audit':result=audit(json.load(open(args.report)) if args.report else r)
        elif args.command=='permutation':
            from . import permutation as perm
            if args.fresh and args.input:raise ValueError('--fresh and --input are mutually exclusive')
            if args.input and not args.audit:raise ValueError('--input requires --audit')
            if args.export and (args.fresh or args.input or args.audit or args.output):raise ValueError('--export cannot be combined with generation, input, audit or output')
            obj=perm.compute(r) if args.fresh else (json.load(open(args.input)) if args.input else perm.saved())
            if args.audit:result=perm.audit(obj,r)
            elif args.export:result=perm.export(args.export)
            else:result={k:obj[k] for k in ['scope','partition','repeats','model_ids','baseline']}
            if args.output:perm.save(obj,args.output)
        elif args.command=='export':result=export(args.output)
        elif args.command=='predict':
            obj=json.load(open(args.input));X=obj['features']
            if set(obj)!={'features'} or not isinstance(X,list) or not X or any(not isinstance(row,list) or len(row)!=4 or any(type(v) not in (float,int) for v in row) for row in X):raise ValueError('features must be nonempty arrays of four numbers')
            chosen=next(c for c in r['candidates'] if c['id']==r['selected_id']);control=next(c for c in r['controls'] if c['id']==r['selected_control_id'])
            result=dict(forest_id=chosen['id'],control_id=control['id'],forest=votes(chosen['model'],X),control_scores=scores(control['model'],X).tolist(),scope=r['scope'])
        else:
            m=next(c for c in r['candidates'] if c['id']==r['selected_id'])['model'];groups=r['data']['groups']
            if not 0<=args.group<len(groups) or not 0<=args.tree<len(m['trees']):raise ValueError('known group/tree index required')
            group=groups[args.group];train=r['data']['split']['train'];local=train.index(args.group) if args.group in train else None
            result=dict(group=group,tree=args.tree,witness=trace(m['trees'][args.tree],group['x']),inbag_count=(m['inbag'][args.tree][local] if local is not None else None),scope=r['scope'])
        print(json.dumps(result,allow_nan=False))
    except (ValueError,KeyError,TypeError,OSError,IndexError) as e:
        p.exit(2,'error: '+str(e)+'\n')
if __name__=='__main__':main()
