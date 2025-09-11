"""
Preprocess the Numia dataset to parquet format
"""

import os
import datasets
from transformers import AutoTokenizer

from verl.utils.hdfs_io import copy, makedirs
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/numina_math')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_start', type=int, default=0)
    parser.add_argument('--train_end', type=int, default=10000000)
    parser.add_argument('--test_start', type=int, default=0)
    parser.add_argument('--test_end', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--flat_n', type=int, default=0)

    args = parser.parse_args()

    data_source = f'ScaleML-RLHF/numina_math_15_all'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-1.5B')

    # dataset = dataset['train'].train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = dataset['train']
    # test_dataset = dataset['test']
    train_end = min(args.train_end, len(train_dataset))
    # args.test_end = min(args.test_end, len(test_dataset))
    if train_end > 0:
        print(args.train_start, train_end)
        train_dataset = train_dataset.select(range(args.train_start, train_end))
    # if args.test_end > 0:
    #     test_dataset = test_dataset.shuffle(seed=args.seed).select(range(args.test_start, args.test_end))

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('problem')

            question = question + ' ' + instruction_following

            # We set the data_source as MATH so that we can use the reward model designed for MATH dataset
            
            # reward_model =  example['reward_model']
            reward_model = {
                "style": "rule",
                "ground_truth": example['answer']
            }

            data = {
                "data_source": 'numina_math',
                "prompt": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                "ability": "math",
                "reward_model": reward_model,
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn
    
    def able_to_extract(example):
        if len(tokenizer.encode(example['problem'])) > 700:
            return False
        # if last_boxed_only_string(example["response"]):
        #     return True
        return True

    # train_dataset = train_dataset.filter(able_to_extract)
    # test_dataset = test_dataset.filter(able_to_extract)
    if args.flat_n > 0:
        new_ds = []
        for item in train_dataset:
            for _ in range(args.flat_n):
                new_ds.append(item)
        # train_dataset = [train_dataset] * args.flat_n
        # test_dataset = [test_dataset] * args.flat_n
        # train_dataset = datasets.concatenate_datasets(train_dataset)
        # test_dataset = datasets.concatenate_datasets(test_dataset)
        train_dataset = datasets.Dataset.from_list(new_ds)
    
    print(f"Train dataset size: {len(train_dataset)}")
    # print(f"Test dataset size: {len(test_dataset)}")

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    # train_dataset = train_dataset.shuffle(seed=args.seed)
    # test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    print(train_dataset[0])
    local_dir = f'{args.local_dir}_15_all'
    hdfs_dir = args.hdfs_dir
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    # test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
                                                
