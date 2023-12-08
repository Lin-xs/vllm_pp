WORLD_SIZE=1

max_tokens=(3)

for MAX_TOKENS in "${max_tokens[@]}"
do
    echo "当前 MAX_TOKENS 的值为: ${MAX_TOKENS}"
    # CUDA_VISIBLE_DEVICES=2 python run.py --max_tokens ${MAX_TOKENS}  > ./log/opt_rank0_ws${WORLD_SIZE}_maxtoken${MAX_TOKENS}.txt
    CUDA_VISIBLE_DEVICES=2 python run.py  --max_tokens ${MAX_TOKENS}
done