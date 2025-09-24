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

system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
inst = "Let\'s think step by step and output the final answer within \\boxed{}"

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

test_datasets = args.data.split(',')

res = {}

for test_dataset in test_datasets:
    ds = load_dataset('json', data_files=f'data/{test_dataset}.jsonl', split='train')
    new_outputs = []
    for i in range(args.num_gpu):
        with open(f'result/{model_name}/tmp_outputs/{test_dataset}_outputs_{i}.json', 'r', encoding='utf-8') as f:
            new_outputs.extend(json.load(f))

    this_res = {}
    scores = []
    preds = []
    for i, item in enumerate(tqdm(ds, desc='Scoring')):
        scores.append([])
        preds.append([])
        for output in new_outputs[i]:
            try:
                answer = item['answer']
                if isinstance(answer, list):
                    answer = answer[0]
                answer = str(answer)
                score, str_preds = math_verify.compute_score(output, answer, return_preds=True)
                if str_preds and str_preds[1]:
                    str_preds = str_preds[1][-1]
                else:
                    str_preds = ''
            except:
                score = 0.0
                str_preds = ''
            scores[-1].append(score)
            preds[-1].append(str_preds)

    # scores = [[math_verify.compute_score(output, item['answer']) for output in new_outputs[i]] for i, item in enumerate(ds)]
    scores = np.array(scores)
    acc = np.mean(np.max(scores, axis=1) > 0.5)
    avg_acc = np.mean(np.mean(scores, axis=1))
    print(f"Pass Accuracy: {acc}")
    print(f"Average Accuracy: {avg_acc}")
    res[f'{test_dataset}'] = avg_acc
    this_res[f'{test_dataset}'] = avg_acc
    # res[f'{test_dataset} avg'] = avg_acc
    # this_res[f'{test_dataset} avg'] = avg_acc
    this_df = pd.DataFrame(this_res.items(), columns=['dataset', 'accuracy']).round(4)
    this_df.to_csv(f'result/{model_name}/{test_dataset}_results.csv', index=False)
    save_data = []
    for i, item in enumerate(ds):
        save_data.append({
            'problem': item['problem'],
            'answer': item['answer'],
            'outputs': new_outputs[i],
            'scores': scores[i].tolist(),
            'preds': preds[i]
        })
    with open(f'result/{model_name}/{test_dataset}_outputs.json', 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)

    

print(res)
df = pd.DataFrame(res.items(), columns=['dataset', 'accuracy']).round(4)
df['3 average'] = df.iloc[:3, 1].mean(axis=0)
df['5 average'] = df.iloc[:5, 1].mean(axis=0)
print(df)
df.to_csv(f'result/{model_name}/results.csv', index=False)


