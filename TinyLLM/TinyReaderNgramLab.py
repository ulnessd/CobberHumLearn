#!/usr/bin/env python3
"""
TinyReaderNgramLab.py
Stage 2: train/evaluate transparent n-gram next-token models for the Tiny Reader corpus.
Run after TinyReaderCorpusInspector.py.
"""
from __future__ import annotations
import argparse, csv, json, math, random, re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
BOS='<BOS>'; EOS='<EOS>'; UNK='<UNK>'

def read_jsonl(path:Path)->List[Dict[str,Any]]:
    rows=[]
    with path.open('r',encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def tokenize(text:str)->List[str]: return re.findall(r"[A-Za-z']+|[.,!?;:]", text.lower())

def detok(toks:List[str])->str:
    s=' '.join(t for t in toks if t not in {BOS,EOS})
    s=re.sub(r'\s+([.,!?;:])', r'\1', s).strip()
    if not s: return s
    parts=re.split(r'([.!?]\s+)', s); out=[]; cap=True
    for p in parts:
        if not p: continue
        if cap and re.match(r'[a-z]',p): p=p[0].upper()+p[1:]
        out.append(p); cap=bool(re.match(r'[.!?]\s+',p))
    return ''.join(out).strip()

class NgramModel:
    def __init__(self,max_n=5,alpha=0.1):
        self.max_n=max_n; self.alpha=alpha
        self.ctx={n:defaultdict(Counter) for n in range(1,max_n+1)}
        self.vocab=set()
    def fit(self,texts:List[str]):
        for text in texts:
            toks=[BOS]*(self.max_n-1)+tokenize(text)+[EOS]
            self.vocab.update(toks)
            for i in range(self.max_n-1,len(toks)):
                target=toks[i]
                for n in range(1,self.max_n+1):
                    clen=n-1; context=tuple(toks[i-clen:i]) if clen else tuple()
                    self.ctx[n][context][target]+=1
        self.vocab.add(UNK)
    def best_counter(self,prefix_toks:List[str],order:int):
        order=max(1,min(order,self.max_n))
        for n in range(order,0,-1):
            clen=n-1; context=tuple(prefix_toks[-clen:]) if clen else tuple()
            c=self.ctx[n].get(context)
            if c: return n,context,c
        return 1,tuple(),self.ctx[1].get(tuple(),Counter())
    def predict_next(self,prefix:str,order=5,top_k=10):
        pref=[BOS]*(self.max_n-1)+tokenize(prefix)
        n,context,c=self.best_counter(pref,order); total=sum(c.values())
        if total==0: return []
        return [{'token':tok,'count':cnt,'probability':cnt/total,'used_order':n,'context':' '.join(context)} for tok,cnt in c.most_common(top_k)]
    def sample_next(self,context_toks:List[str],order:int,temp:float,rng:random.Random)->str:
        n,ctx,c=self.best_counter(context_toks,order)
        if not c: return EOS
        items=list(c.items()); toks=[x[0] for x in items]; counts=[float(x[1]) for x in items]
        if temp<=0: return toks[counts.index(max(counts))]
        total=sum(counts); probs=[x/total for x in counts]
        scaled=[p**(1.0/temp) for p in probs]; z=sum(scaled); scaled=[p/z for p in scaled]
        r=rng.random(); acc=0
        for tok,p in zip(toks,scaled):
            acc+=p
            if r<=acc: return tok
        return toks[-1]
    def generate(self,prefix:str,order=5,max_new_tokens=80,temperature=0.8,seed=1):
        rng=random.Random(seed); toks=[BOS]*(self.max_n-1)+tokenize(prefix)
        for _ in range(max_new_tokens):
            nxt=self.sample_next(toks,order,temperature,rng)
            if nxt==EOS: break
            toks.append(nxt)
        return detok(toks[self.max_n-1:])
    def evaluate(self,texts:List[str],order=5):
        total=correct=0; nll=0.0; vs=max(1,len(self.vocab))
        for text in texts:
            toks=[BOS]*(self.max_n-1)+tokenize(text)+[EOS]
            for i in range(self.max_n-1,len(toks)):
                actual=toks[i]; n,ctx,c=self.best_counter(toks[:i],order)
                denom=sum(c.values())+self.alpha*vs
                prob=(c.get(actual,0)+self.alpha)/denom if denom else 1/vs
                pred=c.most_common(1)[0][0] if c else EOS
                total+=1; correct+=int(pred==actual); nll += -math.log(prob)
        return {'order':order,'tokens':total,'top1_accuracy':correct/total if total else 0,'avg_nll':nll/total if total else 0,'perplexity':math.exp(nll/total) if total else float('inf')}
    def to_jsonable(self):
        return {'max_n':self.max_n,'alpha':self.alpha,'context_counts':{str(n):{'\t'.join(ctx):dict(c) for ctx,c in m.items()} for n,m in self.ctx.items()}}

def write_csv(path,rows,fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',default='tiny_reader_model_prep/corpus_train.jsonl'); ap.add_argument('--val',default='tiny_reader_model_prep/corpus_val.jsonl')
    ap.add_argument('--outdir',default='tiny_reader_ngram_output'); ap.add_argument('--max-n',type=int,default=5); ap.add_argument('--alpha',type=float,default=0.1)
    ap.add_argument('--predict',default=''); ap.add_argument('--generate',default=''); ap.add_argument('--order',type=int,default=5); ap.add_argument('--top-k',type=int,default=12)
    ap.add_argument('--num-generate',type=int,default=5); ap.add_argument('--temperature',type=float,default=0.8); ap.add_argument('--seed',type=int,default=7)
    args=ap.parse_args(); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    train=read_jsonl(Path(args.train)); val=read_jsonl(Path(args.val)) if Path(args.val).exists() else []
    m=NgramModel(args.max_n,args.alpha); m.fit([r['text'] for r in train if r.get('text')])
    eval_rows=[m.evaluate([r['text'] for r in (val or train[:500]) if r.get('text')],order=o) for o in range(1,args.max_n+1)]
    write_csv(out/'ngram_eval.csv',eval_rows,['order','tokens','top1_accuracy','avg_nll','perplexity'])
    prompts=['father has a blue','he drops the','father is sad because','mother helps him look for','mother finds the','father thanks','then they','jane gives him the','she waves to','tim goes to the']
    pred_rows=[]
    for p in prompts:
        for rank,row in enumerate(m.predict_next(p,args.order,args.top_k),1): pred_rows.append({'prompt':p,'rank':rank,**row})
    write_csv(out/'ngram_prompt_predictions.csv',pred_rows,['prompt','rank','token','count','probability','used_order','context'])
    sample_prefixes=['Father goes to the yard.','Jane goes to the park.','Mother helps him look for','Dick has a red','Then they']
    lines=[]
    for p in sample_prefixes:
        lines.append('PREFIX: '+p)
        for i in range(args.num_generate): lines += [f'--- sample {i+1}', m.generate(p,args.order,80,args.temperature,args.seed+i)]
        lines.append('')
    (out/'ngram_generated_samples.txt').write_text('\n'.join(lines),encoding='utf-8')
    (out/'ngram_model.json').write_text(json.dumps(m.to_jsonable(),ensure_ascii=False),encoding='utf-8')
    rep=['Tiny Reader N-gram Lab','======================',f'Train records: {len(train)}',f'Validation records: {len(val)}',f'Vocabulary size: {len(m.vocab)}',f'Max n: {args.max_n}','', 'Validation metrics:']
    for r in eval_rows: rep.append(f"  order {r['order']}: top1={r['top1_accuracy']:.3f}, ppl={r['perplexity']:.2f}, nll={r['avg_nll']:.3f}")
    rep.append('\nCanonical prompt predictions:')
    for p in prompts:
        preds=m.predict_next(p,args.order,5); rep.append(f"  {p!r} -> "+', '.join(f"{x['token']} ({x['probability']:.2f})" for x in preds))
    (out/'ngram_summary.txt').write_text('\n'.join(rep)+'\n',encoding='utf-8')
    if args.predict:
        print(f'Predictions for: {args.predict!r}')
        for r in m.predict_next(args.predict,args.order,args.top_k): print(f"  {r['token']:>12s} count={r['count']:5d} p={r['probability']:.3f} order={r['used_order']}")
    if args.generate:
        print(f'Generated continuations for: {args.generate!r}')
        for i in range(args.num_generate): print(f'\n--- sample {i+1}\n'+m.generate(args.generate,args.order,80,args.temperature,args.seed+i))
    print('\n'.join(rep[:14])); print(f'\nWrote outputs to: {out.resolve()}')
if __name__=='__main__': main()
