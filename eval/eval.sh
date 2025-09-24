GPUS=(0 1)
NUM_GPUS=${#GPUS[@]}
echo "NUM_GPUS: $NUM_GPUS"

model_name_or_path="PRM-CoT/Qwen2.5-Math-7B-prm-n1-eta100-stepLen256-step500"
model_prefix="Qwen2.5-Math-7B-prm-n1-eta100-stepLen256-step500"
data="math500,minerva_math,olympiad_bench,aime24,amc23"

for i in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} python gen.py --model_name_or_path=$model_name_or_path \
        --model_prefix=$model_prefix --data=$data --num_gpu=$NUM_GPUS --local_rank=$i &
done

wait

echo "Generation done"

python aggregate.py --model_prefix=$model_prefix --data=$data --num_gpu=$NUM_GPUS
