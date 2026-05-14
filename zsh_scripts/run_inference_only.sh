MODEL_NAME=../models/attribution_models/T5-use_InbatchNegatives-attributionBench_augmentedQwen30B_allerrors_contrastive_extendalltrain-bs16-lr5e-5-gas4-contW1.0-classifW1.0-Ep4-nneg8-npos8 #../models/attribution_models/T5-use_InbatchNegatives-attributionBench_hardpos_augmentedQwen30B_allerrors_contrastive-bs32-lr5e-5-gas4-contW1.0-classifW1.0-Ep3-nneg5-npos5-tau1.0 #/lustre/fswork/projects/rech/fiz/udo61qq/Code/checkpoints/attribution_models/google/t5_xxl_true_nli_mixture-AttributionBench-template-base_c_e-bs16-lr1e-5-gas4 #
export HF_HOME=$WORK/.cache/huggingface
export WANDB_MODE=offline

export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


# ***************** Set parameters here *****************

dataset_version=augmentedTest
start_gpu_index=0
template=base_c_e
nodes=4
data_path=/lustre/fswork/projects/rech/fiz/udo61qq/Code/RAGnRoll/results/augmented_datasets/Qwen3-30B-A3B-Instruct-2507_allfewshot_alltest_v2/allQwen3-30B-A3B-Instruct-2507_allfewshot_alltest_v2_hardpos.json
# ***************** The followings are auto-calculated parameters *****************
cuda_devices=$(seq -s ',' $start_gpu_index $(($start_gpu_index + $nodes - 1)))
export CUDA_VISIBLE_DEVICES=$cuda_devices

current_time=$(date +"%Y-%m-%d-%H:%M:%S")

echo ${CUDA_VISIBLE_DEVICES}

python ../src/inference/run_inference.py \
    --method autoais \
    --custom_data ${data_path}\
    --template_path ../src/prompts.json \
    --model_name ${MODEL_NAME} \
    --bs 1 \
    --split augmentedtest\
    --output_dir ../inference_results/${dataset_version} \
    --max_length 2048  \
    --max_new_tokens 6 \

##zeroshot
# python ../src/inference/run_inference.py \
#     --method autoais \
#     --custom_data ${data_path}\
#     --template_path ../src/prompts.json \
#     --model_name google/t5_xxl_true_nli_mixture\
#     --bs 4 \
#     --split augmentedtest \
#     --output_dir ../inference_results/${dataset_version} \
#     --max_length 2048  \
#     --max_new_tokens 6 \
#     --template ${template}