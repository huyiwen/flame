#!/usr/bin/bash
cd /data/flame
set -x

########### CONFIGS ###########
export BATCH_SIZE=32
export SEQ_LEN=4096
export GAS=2
export LR=0.0004
export EULERFORMER=false
export NNODES=${NNODES:-1}
export EXTRA=".fp8"

MODEL_ARCH="yulanmini-16"
LOAD_CHECKPOINT=1

path=exp/$MODEL_ARCH/n${NNODES}.batch${BATCH_SIZE}.gas${GAS}.lr${LR}.seqlen${SEQ_LEN}.ef${EULERFORMER}${EXTRA}

export WARMUP_STEPS=1024
###############################

config_tmpl=${1:-"train.tpl.toml"}
config_file=${config_tmpl%.tpl.toml}.toml
if ! command -v envsubst &> /dev/null; then
  bash /data/install/setup.sh
fi
if [[ $MODEL_ARCH == "transformer" ]]; then
  export CONFIG=/data/flame/configs/transformer_340M.json
elif [[ $MODEL_ARCH == "yulanmini-16" ]]; then
  envsubst < /data/metadata/output/miniyulan-2B-final-stage16-nooptim/checkpoint-155587/config.tpl.json > /data/metadata/output/miniyulan-2B-final-stage16-nooptim/checkpoint-155587/config.json
  export CONFIG=/data/metadata/output/miniyulan-2B-final-stage16-nooptim/checkpoint-155587
elif [[ $MODEL_ARCH == "yulanmini-16-4a2" ]]; then
  export CONFIG=/data/metadata/output/miniyulan-2B-final-stage16-nooptim_moe_4A2/checkpoint-155587
elif [[ $MODEL_ARCH == "yulanmini-pub" ]]; then
  export CONFIG=/data/metadata/reference-models/YuLan-Mini-Pub
elif [[ $MODEL_ARCH == "yulanmini-25-4a2" ]]; then
  export CONFIG=/data/metadata/reference-models/YuLan-Mini-Before-Annealing_moe_4A2
fi
envsubst < $config_tmpl > $config_file

# if load_checkpoint is set and directory not exists, create a symlink
if [[ $LOAD_CHECKPOINT == 1 && ! -d $path/checkpoint/step-0 ]]; then
  mkdir -p $path/checkpoint
  if [[ $MODEL_ARCH == "yulanmini-16" ]]; then
    ln -s /data/metadata/output/miniyulan-2B-final-stage16-tt/step-0 $path/checkpoint
  elif [[ $MODEL_ARCH == "yulanmini-pub" ]]; then
    ln -s /data/metadata/output/YuLan-Mini-Pub-tt/step-0 $path/checkpoint
  elif [[ $MODEL_ARCH == "yulanmini-16-4a2" ]]; then
    ln -s /data/metadata/output/miniyulan-16-4A2/step-0 $path/checkpoint
  elif [[ $MODEL_ARCH == "yulanmini-25-4a2" ]]; then
    ln -s /data/metadata/output/miniyulan-25-4a2-tt/step-0 $path/checkpoint
  fi
fi


########### TRAINING ##########
NNODE=${NNODE:-"1"}
NGPU=${NGPU:-"8"}
LOG_RANK=${LOG_RANK:-0}
RANK=${RANK:-0}

if [[ -z "${MASTER_ADDR}" ]]    ; then
  export MASTER_ADDR="localhost"
fi
if [[ -z "${MASTER_PORT}" ]]; then
  export MASTER_PORT="0"
fi

echo "Launching training..."


########### LOGGING ###########
if [ "$JOB_ID" == "" ]; then
  date=$(date +%Y%m%d%H%M)
else
  date=$JOB_ID
fi

mkdir -p $path/artifacts-$date
# cp -r * $path
cp -r configs      $path/artifacts-$date
cp -r flame        $path/artifacts-$date
cp -r train.py     $path/artifacts-$date
cp -r $config_file $path/artifacts-$date
cp -r $config_tmpl $path/artifacts-$date
cp -r $0           $path/artifacts-$date

export WANDB_RESUME=allow
if [[ -z "${WANDB_PROJECT}" ]]; then
  export WANDB_PROJECT="fla"
fi
if [[ -z "${WANDB_NAME}" ]]; then
  export WANDB_NAME="$(basename $path)"
fi
if [[ -z "${WANDB_RUN_ID}" ]]; then
  export WANDB_RUN_ID="$WANDB_NAME-$date"
fi

######### ENVIRONMENT #########
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1
# export TORCHDYNAMO_VERBOSE=1
export HF_ENDPOINT="https://hf-mirror.com"

source /data/flame/.venv/bin/activate
###############################

if [[ $NNODES -gt 1 ]]; then
    distributed_args="--nproc_per_node $NGPU --nnodes $NNODES --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT --max_restarts 0 --rdzv_backend static"
else
    distributed_args="--nproc_per_node $NGPU --standalone"
fi

torchrun $distributed_args train.py \
  --job.config_file $config_file --job.dump_folder $path |& tee ${path}/artifacts-${date}/train-${RANK}.log

echo "RUNNING DONE!"
error=$(tail -100 ${path}/artifacts-${date}/train-${RANK}.log | grep Error | grep -v ChildFailedError | grep -v thread | tail -1)
curl -H "Content-Type: application/json" -X POST https://wxpusher.zjiecode.com/api/send/message --data "{\"appToken\": \"AT_6x1rUKLWJsd3DGyvm7NNxpI3GNr7bEN5\", \"content\": \"[$RANK/$NNODES $JOB_ID] $JOB_UNIQUE_NAME $error\", \"topicIds\": [37328]}"
first_word=$(echo $error | sed 's/\[rank.\]: //g' | awk '{print $1}')
echo $error > ${path}/artifacts-${date}/error-${RANK}-${first_word%:}.log
