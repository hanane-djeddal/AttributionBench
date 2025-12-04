models=("google/t5_xxl_true_nli_mixture")
export HF_HOME=$WORK/.cache/huggingface
export WANDB_MODE=offline

export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


for model in "${models[@]}"; do
    # 32 1e-5
    # ***************** Set parameters here *****************
    dataset_version=attributionBench_contrastive1neg_mismatch #subset_balanced AttributionBench #
    template=base_c_e
    lr=1e-4
    weight=0.2
    temp=0.07
    num_train_epoches=8
    start_gpu_index=0
    master_port=11111
    per_device_train_batch_size=2
    gas=4
    nodes=4
    cche_dir=$WORK/.cache/huggingface
    # ***************** The followings are auto-calculated parameters *****************
    cuda_devices=$(seq -s ',' $start_gpu_index $(($start_gpu_index + $nodes - 1)))
    export CUDA_VISIBLE_DEVICES=$cuda_devices
    bs=$((gas * nodes))
    eval_bs=1 #$((per_device_train_batch_size * 2))
    setting=template-${template}-bs${bs}-lr${lr}-gas${gas}-temp${temp}-wght${weight}
    current_time=$(date +"%Y-%m-%d-%H:%M:%S")

    echo ${CUDA_VISIBLE_DEVICES}
    # make sure you want to do the deletion  rm -rf $OUTPUT_DIR
    # ************************************************************************************
    export OUTPUT_DIR=../models/attribution_models/cont-${model}-${dataset_version}-${setting}
    #rm -rf $OUTPUT_DIR
    # ************************************************************************************

    export WANDB_NAME=${model}_${setting}_dataset_${dataset_version}_${current_time}

    # # train         --evaluation_strategy "no" \ --cache_dir ${cche_dir}  --generation_max_length 128 \ --generation_num_beams 1 \
    torchrun --nproc_per_node ${nodes} --master-port ${master_port} ../src/train/autoais_train_contrastive.py \
        --model_name_or_path $model \
        --data_path AttributionBench \
        --template ${template} \
        --template_path ../src/prompts.json \
        --dataset_version ${dataset_version} \
        --output_dir $OUTPUT_DIR \
        --num_train_epochs $num_train_epoches \
        --per_device_train_batch_size $per_device_train_batch_size \
        --per_device_eval_batch_size ${eval_bs} \
        --gradient_accumulation_steps ${gas} \
        --save_total_limit 1 \
        --eval_strategy "no"\
        --save_strategy "no" \
        --save_only_model True \
        --logging_steps 10 \
        --learning_rate $lr \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --predict_with_generate True \
        --lr_scheduler_type "cosine" \
        --contrastive_weight $weight\
        --cont_temperature $temp\
        --bf16 True \
        --tf32 True \
        --report_to wandb \
        --fsdp 'full_shard auto_wrap' \
        --fsdp_transformer_layer_cls_to_wrap 'T5Block'\

    # inference
    python ../src/inference/run_inference.py \
        --method autoais \
        --data_path AttributionBench \
        --dataset_version ${dataset_version} \
        --template_path ../src/prompts.json \
        --model_name ${OUTPUT_DIR} \
        --bs 1 \
        --split test_ood test\
        --output_dir ../inference_results/${dataset_version} \
        --max_length 2048  \
        --max_new_tokens 6 \
        --template ${template}

    # zero-shot
    # python ../src/inference/run_inference.py \
    #     --method autoais \
    #     --data_path AttributionBench \
    #     --dataset_version ${dataset_version} \
    #     --template_path ../src/prompts.json \
    #     --model_name $model \
    #     --bs 4 \
    #     --split test_ood test \
    #     --output_dir ../inference_results/${dataset_version} \
    #     --max_length 2048  \
    #     --max_new_tokens 6 \
    #     --template ${template}
done
