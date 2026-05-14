#!/bin/bash

#dataset_versions=("attributionBench_hardpos_augmentedQwen30B_error_trueOptimshuffled" "attributionBench_hardpos_augmentedQwen30B_allerrors_contrastive" "attributionBench_augmentedQwen30B_allerrors_contrastive_extendalltrainshuffled" "attributionBench_augmentedQwen30B_allerrors_contrastive_extendalltrain" "augmentedTest")
dataset_versions=("AttributionBench" "attributionBench_random_augmentation_hardpos" "attributionBench_random_augmentation_hardpos_mixedAll") #attributionBench_random_augmentation_hardpos") #_mixedAll") #augmentedTest")

#"attributionBench_hardpos_augmentedQwen30B_allerror_shuffled" "attributionBench_hardpos_augmentedQwen30B_allerror" "attributionBench_hardpos_augmentedQwen30B_allerror_mixed_alltrain")
#"attributionBench_augmentedv1_shuffled_mixedalltrainfiltered2" "AttributionBench" "attributionBench_contrastive1neg_mismatch")
# dataset_versions=("subset_balanced" "overall_balanced" "not_balanced" "full_data")

for dataset_version in "${dataset_versions[@]}"; do
    for file in ../inference_results/${dataset_version}/*; do
        if [ -f "$file" ]; then
            # Skip files that contain "analysis" in their name
            if [[ $file == *"analysis"* ]]; then
                continue
            fi

            # Set method based on file prefix
            if [[ $file == *"attrbench"* ]]; then
                method="attrbench"
            elif [[ $file == *"autoais"* ]]; then
                method="autoais"
            else
                # If file does not start with "attrbench" or "autoais", skip the file
                continue
            fi
            
            echo "Processing $file with method $method"
            python ../analysis_inference_results.py --data_path "$file" --method $method ##--error_type
        fi
    done
done
