import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import verl.utils.reward_score.math_verify as math_verify

metric = 'average' # 'majority_vote' or 'average' or 'pass'
ds_names = ['math500', 'minerva_math', 'olympiad_bench', 'aime24', 'amc23']
df = pd.DataFrame(columns=ds_names)

for step in tqdm(range(1, 5, 1)):
    avg_acc = 0
    step_accs = []
    for ds in ds_names:
        rewrite = False
        file_name = f'result/Qwen2.5-Math-1.5B-raft-plusplus-numina_math_em-sample1n32-sample32-iter{step}-n8_t1.0/{ds}_outputs.json'
        try:
            with open(file_name) as f:
                res = json.load(f)
        except:
            continue

        acc = 0
        for i, item in enumerate(tqdm(res)):
            if metric == 'average':
                acc += np.mean(item['scores'])
            elif metric == 'pass':
                acc += 1 if np.max(item['scores']) > 0.5 else 0
            elif metric == 'majority_vote':
                if 'preds' not in item:
                    rewrite = True
                    preds = []
                    for output in item['outputs']:
                        answer = item['answer']
                        if isinstance(answer, list):
                            answer = answer[0]
                        _, str_preds = math_verify.compute_score(output, str(answer), return_preds=True)
                        if str_preds and str_preds[1]:
                            preds.append(str_preds[1][-1])
                    res[i]['preds'] = preds
                votes = {}
                for pred in item['preds']:
                    if pred not in votes:
                        votes[pred] = 1
                    else:
                        votes[pred] += 1
                sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
                vote_idx = item['preds'].index(sorted_votes[0][0])
                if vote_idx >= 0:
                    acc += 1 if item['scores'][vote_idx] > 0.5 else 0

        if rewrite:
            with open(file_name, 'w', encoding='utf-8') as f:
                json.dump(res, f, indent=4, ensure_ascii=False)

        step_accs.append(acc / len(res))

    try:
        df.loc[step] = step_accs
    except:
        continue

df['3 average'] = df.iloc[:, :3].mean(axis=1)
df['5 average'] = df.iloc[:, :5].mean(axis=1)
df = df[['math500', 'minerva_math', 'olympiad_bench', '3 average', 'aime24', 'amc23', '5 average']]
print(df)
df.to_csv('res.csv', index=False, float_format='%.4f', sep='\t')
