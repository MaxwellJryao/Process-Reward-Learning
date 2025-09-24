from vllm import LLM, SamplingParams
import argparse
import torch
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer
import verl.utils.reward_score.math_verify as math_verify
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-Math-1.5B')
parser.add_argument('--max_length', type=int, default=3072)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--n', type=int, default=8)
parser.add_argument('--data', type=str, default='math500,minerva_math,olympiad_bench,aime24,amc23')
parser.add_argument('--tensor_parallel_size', type=int, default=1)
parser.add_argument('--model_prefix', type=str, default='Qwen1.5B')
parser.add_argument('--num_gpu', type=int, default=8)
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()

model_name = args.model_prefix
os.makedirs(f'result/{model_name}', exist_ok=True)
os.makedirs(f'result/{model_name}/tmp_outputs', exist_ok=True)

llm = LLM(args.model_name_or_path, dtype=torch.bfloat16,
          tensor_parallel_size=args.tensor_parallel_size,
          gpu_memory_utilization=0.6)
sampling_params = SamplingParams(
    max_tokens=args.max_length,
    temperature=args.temperature,
    n=args.n
)

system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
inst = "Let\'s think step by step and output the final answer within \\boxed{}"

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

test_datasets = args.data.split(',')

res = {}

for test_dataset in test_datasets:
    ds = load_dataset('json', data_files=f'data/{test_dataset}.jsonl', split='train')

    if (not os.path.exists(f'result/{model_name}/{test_dataset}_outputs.json')) and (not os.path.exists(f'result/{model_name}/tmp_outputs/{test_dataset}_outputs_{args.local_rank}.json')):        
        print(f"Generating {test_dataset}")
        
        prompts = []
        for item in ds:
            conv = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': item['problem'] + f' {inst}'}
            ]
            conv_chat = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            prompts.append(conv_chat)

        batch_size = len(prompts) // args.num_gpu
        start = args.local_rank * batch_size
        end = start + batch_size
        if args.local_rank == args.num_gpu - 1:
            end = len(prompts)
        prompts_batch = prompts[start:end]
        outputs = llm.generate(prompts_batch, sampling_params)
        new_outputs = [[output.text for output in outputs[i].outputs] for i in range(len(outputs))]

        with open(f'result/{model_name}/tmp_outputs/{test_dataset}_outputs_{args.local_rank}.json', 'w', encoding='utf-8') as f:
            json.dump(new_outputs, f, indent=4, ensure_ascii=False)


