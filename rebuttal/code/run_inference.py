#!/usr/bin/env python3
"""Checkpointed native-chat inference; no silent truncation or invented outputs."""
import argparse,json,os,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--tag',required=True);ap.add_argument('--max-model-len',type=int,default=4096);ap.add_argument('--batch-size',type=int,default=24);ap.add_argument('--limit',type=int);args=ap.parse_args()
    from vllm import LLM,SamplingParams
    path=ROOT/'rebuttal/data/inference_suite.jsonl';items=[json.loads(x) for x in path.read_text().splitlines()];items=items[:args.limit] if args.limit else items
    out=ROOT/'rebuttal/results'/f'{args.tag}__new_inference.jsonl'; done={}; suite_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r=json.loads(line)
                if r.get('suite_sha256')!=suite_hash or r.get('hf_repo')!=args.model:raise ValueError('Checkpoint provenance mismatch; select a new output tag')
                done[r['item_id']]=r
    llm=LLM(model=args.model,max_model_len=args.max_model_len,gpu_memory_utilization=.85,trust_remote_code=False,dtype='half',seed=20261108,enable_prefix_caching=False)
    tok=llm.get_tokenizer();sp=SamplingParams(temperature=0,max_tokens=192,seed=20261108)
    def render(messages):
        if not tok.chat_template:raise ValueError('Native chat template missing; do not silently substitute raw completion')
        kw={'enable_thinking':False} if 'Qwen3' in args.model else {}
        return tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**kw)
    todo=[i for i in items if i['item_id'] not in done]
    for start in range(0,len(todo),args.batch_size):
        batch=todo[start:start+args.batch_size];prompts=[render([{'role':'user','content':i['prompt']}]) for i in batch]
        for p in prompts:
            if len(tok.encode(p))+192>args.max_model_len:raise ValueError('Prompt exceeds context; no silent truncation')
        t=time.time();outs=llm.generate(prompts,sp,use_tqdm=False);records=[]
        for item,res in zip(batch,outs):
            r={**item,'model':args.tag,'hf_repo':args.model,'raw_response':res.outputs[0].text,'finish_reason':res.outputs[0].finish_reason,'generated_tokens':len(res.outputs[0].token_ids),'suite_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'protocol_version':'v2_no_answer_examples_prefix_cache_disabled'}
            if 'followup' in item:
                msgs=[{'role':'user','content':item['prompt']},{'role':'assistant','content':r['raw_response']},{'role':'user','content':item['followup']}]
                follow=render(msgs)
                if len(tok.encode(follow))+192>args.max_model_len:raise ValueError('Followup context overflow')
                sec=llm.generate([follow],sp,use_tqdm=False)[0].outputs[0];r['step2_response']=sec.text;r['step2_finish_reason']=sec.finish_reason
            records.append(r)
        with out.open('a') as f:
            for r in records:f.write(json.dumps(r)+'\n')
            f.flush();os.fsync(f.fileno())
        print(f'{args.tag}: {min(start+len(batch),len(todo))}/{len(todo)} pending items; {time.time()-t:.1f}s',flush=True)
        if (start//args.batch_size)%10==0 and not os.environ.get('SKIP_REFRESH'):
            from analyze_new import main as refresh_status
            refresh_status()
if __name__=='__main__':main()
