if [ -z "$1" ]; then
    echo "请输入要创建的log文件夹"
else
    export LOGDIR=~/vllm_pp/nsys-rep/pp_time_record/${1}
    mkdir $LOGDIR
    mv ./nsys-rep/time.json ./nsys-rep/num_seq.json $LOGDIR
    mv ./nsys-rep/pp_time_record/rank_*.log $LOGDIR

fi