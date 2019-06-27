TASK=${1}
SUPERGLUEDATA=${2:-data}
LOGPATH=${3:-logs}
SEED=${4:-111}
GPU=${5:-0}

if [ $1 == "MultiRC" ]; then
    METRIC="f1"
elif [ $1 == "CB" ]; then
    METRIC="accuracy_macro_f1"
else
    METRIC="accuracy"
fi

python run.py \
    --task ${TASK} \
    --data_dir ${SUPERGLUEDATA} \
    --log_root ${LOGPATH} \
    --seed ${SEED} \
    --device ${GPU} \
    --n_epochs 10 \
    --optimizer adam \
    --lr 1e-5 \
    --grad_clip 5.0 \
    --warmup_percentage 0.0 \
    --counter_unit epochs \
    --evaluation_freq 0.25 \
    --logging 1 \
    --checkpointing 1 \
    --checkpoint_metric ${TASK}/SuperGLUE/valid/${METRIC}:max \
    --bert_model bert-large-cased \
    --batch_size 4 \
    --max_sequence_length 256 \
    --dataparallel 0