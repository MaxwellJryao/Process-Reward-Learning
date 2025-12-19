GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}
echo "NUM_GPUS: $NUM_GPUS"

model_name_or_path="/taiga/illinois/eng/cs/tozhang/jyao4/prm-cot/prm-cot/checkpoints/prm/Llama-3.2-3B-Instruct-numina-grpo-prm_advprm-n5-eta200-stepLen256-stepSplit-length/global_step_300/actor/huggingface"
model_prefix="Llama-3.2-3B-Instruct-numina-grpo-prm_advprm-n5-eta200-stepLen256-stepSplit-length-step300"
data="math500,minerva_math,olympiad_bench,amc23,aime24"

for i in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} python gen.py --model_name_or_path=$model_name_or_path \
        --model_prefix=$model_prefix --data=$data --num_gpu=$NUM_GPUS --local_rank=$i &
done

wait

echo "Generation done"

python aggregate.py --model_prefix=$model_prefix --data=$data --num_gpu=$NUM_GPUS
