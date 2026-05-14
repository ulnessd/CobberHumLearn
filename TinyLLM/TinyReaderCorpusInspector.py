#!/usr/bin/env python3
"""
TinyReaderCorpusInspector.py
Stage 1: validate/summarize the polished-only Tiny Reader corpus and create train/val/test splits.
Default input: tiny_reader_sets/tiny_reader_polished_only.jsonl
Default output: tiny_reader_model_prep/
"""
from __future__ import annotations
import argparse, csv, json, random, re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

BOS='<BOS>'; EOS='<EOS>'

def read_jsonl(path: Path)->List[Dict[str,Any]]:
    rows=[]
    with path.open('r', encoding='utf-8') as f:
        for n,line in enumerate(f,1):
            line=line.strip()
            if not line: continue
            try: rec=json.loads(line)
            except Exception as e: raise ValueError(f'Bad JSONL line {n}: {e}')
            rows.append(rec if isinstance(rec,dict) else {'text':str(rec)})
    return rows

def write_jsonl(path: Path, rows: List[Dict[str,Any]]):
    with path.open('w', encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+'\n')

def tokens(text:str)->List[str]:
    return re.findall(r"[A-Za-z']+|[.,!?;:]", text.lower())

def words(text:str)->List[str]:
    return re.findall(r"[A-Za-z']+", text.lower())

def sentences(text:str)->List[str]:
    out=[]
    for line in text.splitlines():
        line=line.strip()
        if not line: continue
        out += [p.strip() for p in re.split(r'(?<=[.!?])\s+', line) if p.strip()]
    return out

def frame(sent:str)->str:
    names={'mother','father','jane','sally','dick','tim','spot','puff'}
    colors={'red','blue','green','yellow'}
    objects={'apple','ball','book','toy','hat','basket','kite'}
    places={'park','home','school','store','garden','street','yard','house'}
    out=[]
    for w in tokens(sent):
        if w in names: out.append('{NAME}')
        elif w in colors: out.append('{COLOR}')
        elif w in objects: out.append('{OBJECT}')
        elif w in places: out.append('{PLACE}')
        else: out.append(w)
    s=' '.join(out)
    return re.sub(r'\s+([.,!?;:])', r'\1', s)

def normalize(rec:Dict[str,Any], i:int)->Dict[str,Any]:
    sid=str(rec.get('story_id') or rec.get('id') or f'REC_{i:06d}')
    text=str(rec.get('text') or rec.get('story') or rec.get('story_text') or '').replace('\\n','\n').strip()
    try: level=int(rec.get('level',0))
    except Exception:
        m=re.search(r'\bL([1-5])[_-]', sid); level=int(m.group(1)) if m else 0
    toks=tokens(text); sents=sentences(text)
    return {
        'story_id':sid, 'level':level,
        'theme':str(rec.get('theme','unknown') or 'unknown'),
        'subtheme':str(rec.get('subtheme','') or ''),
        'status':str(rec.get('status','') or ''),
        'text':text, 'token_count':len(toks), 'word_count':len(words(text)),
        'sentence_count':len(sents), 'line_count':len([x for x in text.splitlines() if x.strip()])
    }

def stratified_split(rows, train_frac, val_frac, seed):
    rng=random.Random(seed); by=defaultdict(list)
    for r in rows: by[r['level']].append(r)
    train=[]; val=[]; test=[]
    for level, group in sorted(by.items()):
        rng.shuffle(group); n=len(group)
        nt=int(round(n*train_frac)); nv=int(round(n*val_frac))
        if nt+nv>n: nv=max(0,n-nt)
        train += group[:nt]; val += group[nt:nt+nv]; test += group[nt+nv:]
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train,val,test

def write_rows(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def write_counter(path, counter, key, max_rows=5000):
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.writer(f); w.writerow([key,'count'])
        for k,c in counter.most_common(max_rows): w.writerow([k,c])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', default='tiny_reader_sets/tiny_reader_polished_only.jsonl')
    ap.add_argument('--outdir', default='tiny_reader_model_prep')
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--train-frac', type=float, default=0.90)
    ap.add_argument('--val-frac', type=float, default=0.05)
    ap.add_argument('--test-frac', type=float, default=0.05)
    args=ap.parse_args()
    if abs(args.train_frac+args.val_frac+args.test_frac-1)>1e-6: raise SystemExit('Fractions must sum to 1.0')
    inp=Path(args.input); out=Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    rows=[normalize(r,i+1) for i,r in enumerate(read_jsonl(inp))]
    rows=[r for r in rows if r['text'].strip()]
    warnings=[]; seen=set()
    for r in rows:
        if r['story_id'] in seen: warnings.append({'story_id':r['story_id'],'warning':'duplicate_story_id'})
        seen.add(r['story_id'])
        if r['level'] not in {1,2,3,4,5}: warnings.append({'story_id':r['story_id'],'warning':f'unexpected_level_{r["level"]}'})
        if r['word_count']<10: warnings.append({'story_id':r['story_id'],'warning':f'very_short_{r["word_count"]}'})
    train,val,test=stratified_split(rows,args.train_frac,args.val_frac,args.seed)
    write_jsonl(out/'corpus_records_clean.jsonl', rows); write_jsonl(out/'corpus_train.jsonl', train)
    write_jsonl(out/'corpus_val.jsonl', val); write_jsonl(out/'corpus_test.jsonl', test)
    vocab=Counter(); starts=Counter(); frames=Counter(); length_rows=[]
    for r in rows:
        vocab.update(tokens(r['text']))
        length_rows.append({k:r[k] for k in ['story_id','level','token_count','word_count','sentence_count','line_count']})
        for s in sentences(r['text']):
            ws=words(s)
            if ws: starts[ws[0]]+=1
            frames[frame(s)]+=1
    level_rows=[]
    for level in sorted({r['level'] for r in rows}):
        g=[r for r in rows if r['level']==level]; ts=[r['token_count'] for r in g]; ws=[r['word_count'] for r in g]
        level_rows.append({'level':level,'stories':len(g),'mean_tokens':round(sum(ts)/len(ts),2),'min_tokens':min(ts),'max_tokens':max(ts),'mean_words':round(sum(ws)/len(ws),2)})
    write_rows(out/'corpus_level_summary.csv', level_rows, ['level','stories','mean_tokens','min_tokens','max_tokens','mean_words'])
    write_counter(out/'corpus_vocab.csv', vocab, 'token'); write_counter(out/'corpus_sentence_starts.csv', starts, 'sentence_start', 500)
    write_counter(out/'corpus_sentence_frames.csv', frames, 'sentence_frame', 1000)
    write_rows(out/'corpus_token_lengths.csv', length_rows, ['story_id','level','token_count','word_count','sentence_count','line_count'])
    write_rows(out/'corpus_warnings.csv', warnings, ['story_id','warning'])
    rep=['Tiny Reader Corpus Inspector','============================',f'Input: {inp}',f'Total usable records: {len(rows)}',f'Train/Val/Test: {len(train)} / {len(val)} / {len(test)}',f'Vocabulary size including punctuation: {len(vocab)}',f'Warnings: {len(warnings)}','', 'Stories by level:']
    for r in level_rows: rep.append(f"  L{r['level']}: {r['stories']} stories | mean tokens {r['mean_tokens']} | range {r['min_tokens']}-{r['max_tokens']}")
    rep.append('\nTop 25 tokens:'); rep += [f'  {k}: {c}' for k,c in vocab.most_common(25)]
    rep.append('\nTop 25 sentence frames:'); rep += [f'  {c:6d}  {k}' for k,c in frames.most_common(25)]
    (out/'corpus_summary.txt').write_text('\n'.join(rep)+'\n', encoding='utf-8')
    print('\n'.join(rep[:12])); print(f'\nWrote outputs to: {out.resolve()}')
if __name__=='__main__': main()
