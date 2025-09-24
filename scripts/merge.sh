steps=(500)

for step in ${steps[@]}; do
    local_dir=/taiga/illinois/eng/cs/tozhang/jyao4/prm-cot/prm-cot/checkpoints/prm/Qwen2.5-Math-7B-numina-prm-n5-eta100-stepLen16/global_step_${step}/actor
    target_dir=$local_dir/huggingface

    python scripts/legacy_model_merger.py merge \
        --backend fsdp \
        --local_dir $local_dir \
        --target_dir $target_dir \
        --hf_model_path Qwen/Qwen2.5-Math-7B

    hf upload-large-folder PRM-CoT/Qwen2.5-Math-7B-prm-n5-eta100-stepLen16-step${step} --repo-type model \
        $target_dir --num-workers 16
done