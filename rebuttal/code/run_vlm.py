#!/usr/bin/env python3
"""Vision-language runner (vLLM chat API with local images). Suite items carry an 'images' list of absolute paths."""
import argparse,json,time,hashlib
import os;os.environ.setdefault('VLLM_ATTENTION_BACKEND','XFORMERS');os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER','0')  # avoid flashinfer import failure (python3.10) on sm80+ GPUs
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--tag',required=True);ap.add_argument('--tp',type=int,default=1);ap.add_argument('--suite',default='rebuttal/data/mm_vlm_suite.jsonl');ap.add_argument('--out',default='rebuttal/results/study2');ap.add_argument('--suffix',default='mmvlm');ap.add_argument('--max-tokens',type=int,default=120);ap.add_argument('--batch',type=int,default=64);ap.add_argument('--max-pixels',type=int,default=401408)
a=ap.parse_args();from vllm import LLM,SamplingParams
llm=LLM(model=a.model,tensor_parallel_size=a.tp,dtype='bfloat16',max_model_len=8192,gpu_memory_utilization=.90,limit_mm_per_prompt={'image':3},allowed_local_media_path=str(ROOT),mm_processor_kwargs={'max_pixels':a.max_pixels},seed=20261108,trust_remote_code=False,enforce_eager=True)
P=ROOT/a.suite;items=[json.loads(l) for l in P.read_text().splitlines() if l.strip()];h=hashlib.sha256(P.read_bytes()).hexdigest();O=ROOT/a.out;O.mkdir(parents=True,exist_ok=True);f=O/f'{a.tag}__{a.suffix}.jsonl'
done=set(json.loads(l)['item_id'] for l in f.read_text().splitlines() if l.strip()) if f.exists() else set();todo=[i for i in items if i['item_id'] not in done];sp=SamplingParams(temperature=0,max_tokens=a.max_tokens,seed=20261108);t=time.time()
def msg(i):
    c=[{'type':'image_url','image_url':{'url':'file://'+p}} for p in i.get('images',[])]+[{'type':'text','text':i['prompt']}];return [{'role':'user','content':c}]
for s in range(0,len(todo),a.batch):
    b=todo[s:s+a.batch];res=llm.chat([msg(i) for i in b],sp,use_tqdm=False)
    with f.open('a') as fh:
        for i,r in zip(b,res):fh.write(json.dumps({**{k:v for k,v in i.items() if k!='prompt'},'model':a.tag,'hf_repo':a.model,'raw_response':r.outputs[0].text,'finish_reason':r.outputs[0].finish_reason,'suite_sha256':h,'protocol_version':'vlm_v1'})+'\n')
    print(a.tag,min(s+a.batch,len(todo)),'/',len(todo),f'{time.time()-t:.0f}s',flush=True)
