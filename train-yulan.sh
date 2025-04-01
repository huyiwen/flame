#!/usr/bin/bash
FLAME_PATH=/cpfs01/user/wenxian.hyw/flame
ALLOW_UNCOMMIT=0
cd $FLAME_PATH
set -x

########### CONFIGS ###########
export BATCH_SIZE=32
export SEQ_LEN=4096
export GAS=1
export LR=0.0006
export NNODES=${NNODES:-1}
export NUM_LOCAL_EXPERTS=4
export NUM_EXPERTS_PER_TOK=2

MODEL_ARCH=${MODEL_ARCH:-"hyw_dense"}
# EXTRA=".grouped.${NUM_LOCAL_EXPERTS}a${NUM_EXPERTS_PER_TOK}"
EXTRA=""

path=exp/$MODEL_ARCH/n${NNODES}.batch${BATCH_SIZE}.gas${GAS}.lr${LR}.seqlen${SEQ_LEN}${EXTRA}

LOAD_CHECKPOINT=1
DELETE_ONE=1
export WARMUP_STEPS=1024
export LOAD_STEP=-1
export DECAY_RATIO=0
# export DATASET=/data/hf_dataset/myl_mix890_long_28k_final/26_20241211_015209
export DATASET=/cpfs01/user/wenxian.hyw/data/fineweb-edu/sample/100BT
export STEPS=1024
# export DATASET=HuggingFaceFW/fineweb-edu,opencsg/Fineweb-Edu-Chinese-V2.1,OpenCoder-LLM/opc-fineweb-code-corpus,math-ai/AutoMathText,EleutherAI/proof-pile-2,OpenCoder-LLM/opc-fineweb-math-corpus
###############################



config_tmpl=${1:-"train-yulan.tpl.toml"}
config_file=${config_tmpl%.tpl.toml}.toml
if ! command -v envsubst &> /dev/null; then
  bash /data/install/setup.sh
fi

# fla natively supported archs
if [[ $MODEL_ARCH == "transformer" ]]; then
    export CONFIG=$FLAME_PATH/configs/transformer_340M.json
elif [[ $MODEL_ARCH == "rwkv7" ]]; then
    export CONFIG=$FLAME_PATH/configs/rwkv7

# hyw custom archs
elif [[ $MODEL_ARCH == "fla-yulan-new" ]]; then
    export CONFIG=$FLAME_PATH/configs/yulan_new
elif [[ $MODEL_ARCH == "fla-yulan-moe" ]]; then
    export CONFIG=$FLAME_PATH/configs/yulan_moe
elif [[ $MODEL_ARCH == "hyw_nsa" ]]; then
    export CONFIG=$FLAME_PATH/configs/yulan_nsa
elif [[ $MODEL_ARCH == "hyw_dense" ]]; then
    export CONFIG=$FLAME_PATH/configs/hyw_dense

# legacy code, need to move to hyw and register
elif [[ $MODEL_ARCH == "yulanmini-16" ]]; then
    export CONFIG=/data/metadata/output/miniyulan-2B-final-stage16-nooptim/checkpoint-155587
elif [[ $MODEL_ARCH == "yulanmini-16-4a2" ]]; then
    export CONFIG=/data/metadata/output/miniyulan-2B-final-stage16-nooptim_moe_4A2/checkpoint-155587
elif [[ $MODEL_ARCH == "yulanmini-pub" ]]; then
    export CONFIG=/data/metadata/reference-models/YuLan-Mini-Pub
elif [[ $MODEL_ARCH == "yulanmini-25-4a2" ]]; then
    CANDIDATES=(modeling_yulanmini_mg.py.py:modeling_yulanmini.py moe_module_specs.py moe_layer.py configuration_yulanmini.py rope.py modeling_flash_attention_utils.py)
    export CONFIG=/data/metadata/reference-models/YuLan-Mini-Before-Annealing_moe_4A2
elif [[ $MODEL_ARCH == "scratch-8a2" ]]; then
    CANDIDATES=(modeling_yulanmini_mg.py:modeling_yulanmini.py moe_module_specs.py moe_layer.py configuration_yulanmini.py rope.py modeling_flash_attention_utils.py)
    export CONFIG=$FLAME_PATH/3rdparty/configs/scratch-8a2
elif [[ $MODEL_ARCH == "tianyu-8a2" ]]; then
    CANDIDATES=(modeling_yulanmini_tianyu.py:modeling_yulanmini.py configuration_yulanmini.py rope.py modeling_flash_attention_utils.py)
    export CONFIG=$FLAME_PATH/3rdparty/configs/tianyu-8a2
fi

# substitute the config file
if [[ -f "$CONFIG/config.tpl.json" ]]; then
    envsubst < $CONFIG/config.tpl.json > $CONFIG/config.json
fi
envsubst < $config_tmpl > $config_file

# legacy code, need to move to hyw and register
if [[ $CANDIDATE ]]; then
    for candidate in ${CANDIDATES[@]}; do
        # split by : if there is a :
        if [[ $candidate == *":"* ]]; then
            target=$(echo $candidate | cut -d ":" -f 2)
            candidate=$(echo $candidate | cut -d ":" -f 1)  # do not change the order
        else
            target=$candidate
        fi
        ln -sf $FLAME_PATH/3rdparty/modelings-candidates/$candidate $CONFIG/$target
        cp -r $candidate $path/artifacts-$date
    done
fi

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
    else
        echo "No checkpoint found for $MODEL_ARCH"
    fi
fi
if [[ $DELETE_ONE == 1 && -d $path/checkpoint/step-1 ]]; then
    rm -rf $path/checkpoint/step-1
fi


########### TRAINING ##########
NGPU=${NGPU:-"2"}
LOG_RANK=${LOG_RANK:-0}
RANK=${RANK:-0}

if [[ -z "${MASTER_ADDR}" ]]    ; then
  export MASTER_ADDR="localhost"
  if [[ -n $(git status --porcelain)  && "${ALLOW_UNCOMMIT}" != "1" ]]; then
    git status
    echo "检测到未提交的更改，脚本已终止。"
    exit 1
  fi
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
# cp -r configs      $path/artifacts-$date
# cp -r flame        $path/artifacts-$date
# cp -r torchtitan   $path/artifacts-$date
# cp -r fla          $path/artifacts-$date
# cp -r train.py     $path/artifacts-$date
# cp -r $config_file $path/artifacts-$date
# cp -r $config_tmpl $path/artifacts-$date
cp -r $0           $path/artifacts-$date

export WANDB_RESUME=allow
if [[ -z "${WANDB_PROJECT}" ]]; then
  export WANDB_PROJECT="fla"
fi
if [[ -z "${WANDB_NAME}" ]]; then
  export WANDB_NAME="${MODEL_ARCH}.$(basename $path)"
fi
if [[ -z "${WANDB_RUN_ID}" ]]; then
  export WANDB_RUN_ID="$WANDB_NAME-$date"
fi

######### ENVIRONMENT #########
export OMP_NUM_THREADS=4
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1
# export TORCHDYNAMO_VERBOSE=1
# export TORCH_LOGS="+dynamo"
export HF_ENDPOINT="https://hf-mirror.com"

source $FLAME_PATH/.venv/bin/activate
###############################

if [[ $NNODES -gt 1 ]]; then
    distributed_args="--nproc_per_node $NGPU --nnodes $NNODES --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT --max_restarts 0 --rdzv_backend static"
else
    distributed_args="--nproc_per_node $NGPU --standalone"
fi

torchrun $distributed_args train.py \
  --job.config_file $config_file --job.dump_folder $path |& tee ${path}/artifacts-${date}/train-${RANK}.log

echo "RUNNING DONE!"
error=$(tail -100 ${path}/artifacts-${date}/train-${RANK}.log | grep "Error\|timeout" | grep -v ChildFailedError | grep -v thread | tail -1)
# curl -H "Content-Type: application/json" -X POST https://wxpusher.zjiecode.com/api/send/message --data "{\"appToken\": \"AT_6x1rUKLWJsd3DGyvm7NNxpI3GNr7bEN5\", \"content\": \"[$RANK/$NNODES $JOB_ID] $JOB_UNIQUE_NAME $error\", \"topicIds\": [37328]}"
first_word=$(echo $error | sed 's/\[rank.\]: //g' | awk '{print $1}')
echo $error > ${path}/artifacts-${date}/error-${RANK}-${first_word%:}.log
