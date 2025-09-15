# python scripts/legacy_model_merger.py merge \
#     --backend fsdp \
#     --local_dir checkpoints/prm/Qwen2.5-Math-7B-numina-grpo-n1/global_step_500/actor \
#     --target_dir checkpoints/prm/Qwen2.5-Math-7B-numina-grpo-n1/global_step_500/actor/huggingface \
#     --hf_model_path Qwen/Qwen2.5-Math-7B

hf upload-large-folder PRM-CoT/Qwen2.5-Math-7B-grpo-n1-step500 --repo-type model \
    checkpoints/prm/Qwen2.5-Math-7B-numina-grpo-n1/global_step_500/actor/huggingface --num-workers 16