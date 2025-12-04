export CUDA_VISIBLE_DEVICES=0
export HF_HOME=$WORK/.cache/huggingface
export WANDB_MODE=offline

template=base_c_e  # base_q_c_e, base_q_c_e_r, base_c_e_r
dataset_version=subset_balanced

# put the model dir here
export MODEL_DIR=Qwen/Qwen3-4B   #osunlp/attrscore-alpaca-7b

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1  python ../src/inference/run_inference.py \
--method attrbench \
--data_path AttributionBench \
--dataset_version ${dataset_version} \
--template_path ../src/prompts.json \
--model_name ${MODEL_DIR} \
--bs 4 \
--split test_ood test \
--output_dir ../inference_results/${dataset_version} \
--max_length 4096 \
--max_new_tokens 512 \
--template ${template}
