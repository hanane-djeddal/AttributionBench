import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
from typing import Dict, Optional
from dataclasses import dataclass
import os 
import datasets 
from peft import LoraConfig, get_peft_model, TaskType

import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

class ContrastiveT5(T5ForConditionalGeneration):
    def __init__(self, config, contrastive_weight=0.3, temperature=0.07):
        super().__init__(config)
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        
        # 1. Inspect the Parent Class (T5)
        # We determine exactly what arguments T5 accepts so we never pass it 
        # something it doesn't understand (like 'original_labels' or 'num_items_in_batch').
        signature = inspect.signature(super().forward)
        self._valid_parent_args = set(signature.parameters.keys())

    def get_pooled_embedding(self, hidden_states, attention_mask):
        """Mean Pooling with division-by-zero safety."""
        input_mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        # --- YOUR CUSTOM ARGUMENTS ---
        # We capture them explicitly here so they are available in this function
        neg_input_ids=None,       
        neg_attention_mask=None,
        original_labels=None, 
        # -----------------------------
        **kwargs
    ):
        # 2. Filter arguments for the Parent Class
        # We start with the known arguments T5 definitely needs
        parent_args = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'output_hidden_states': True,
            'return_dict': True
        }

        # We check any other leftover args (like 'decoder_input_ids' or 'past_key_values')
        # against the valid signature we inspected in __init__.
        # This automatically drops 'num_items_in_batch' and 'original_labels' 
        # from being passed to super(), preventing the crash.
        for key, value in kwargs.items():
            if key in self._valid_parent_args:
                parent_args[key] = value

        # 3. Safe Call to Parent
        outputs = super().forward(**parent_args)
        
        # Return early if not training or no negatives provided
        if not self.training or neg_input_ids is None:
            return outputs

        # 4. Extract Anchor Embeddings
        anchor_embeds = self.get_pooled_embedding(
            outputs.encoder_last_hidden_state, 
            attention_mask
        )

        # 5. Compute Negative Embeddings
        if self.config.use_cache is False: 
             neg_inputs_embeds = self.shared(neg_input_ids)
             if self.training:
                 neg_inputs_embeds.requires_grad_(True) 
             
             neg_outputs = self.encoder(
                 inputs_embeds=neg_inputs_embeds,
                 attention_mask=neg_attention_mask,
                 return_dict=True
             )
        else:
             neg_outputs = self.encoder(
                 input_ids=neg_input_ids,
                 attention_mask=neg_attention_mask,
                 return_dict=True
             )

        neg_embeds = self.get_pooled_embedding(
            neg_outputs.last_hidden_state, 
            neg_attention_mask
        )

        # 6. Contrastive Loss Logic
        # (Optional: You can use 'original_labels' here if you only want to 
        # contrast samples that are actually grounded (Label 1))
        
        anchor_norm = F.normalize(anchor_embeds, p=2, dim=1)
        neg_norm = F.normalize(neg_embeds, p=2, dim=1)
        sim_neg = torch.sum(anchor_norm * neg_norm, dim=1)
        
        # Margin Loss
        margin = -0.1
        contrastive_loss = torch.mean(torch.clamp(sim_neg - margin, min=0))

        # 7. Final Loss
        final_loss = outputs.loss + (self.contrastive_weight * contrastive_loss)
        
        return Seq2SeqLMOutput(
            loss=final_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )  
def prepare_dataset_for_contrastive(examples, tokenizer, max_length=512):
    """
    Prepare dataset with anchor and negative examples.
    
    Expected input format:
    {
        'claim': list of claims,
        'reference': list of references,
        'label': list of labels (0 or 1),
        'negative_claim': list of distorted claims,
        'negative_reference': list of distorted references
    }
    """
    # Format as NLI: premise = reference, hypothesis = claim
    #print("examples", examples.keys())
    anchor_inputs = [
        f"premise: {' '.join(ref)} hypothesis: {claim}" 
        for claim, ref in zip(examples['claim'],examples['references'])
    ]
    
    negative_inputs = [
        f"premise: {' '.join(neg_ref)} hypothesis: {neg_claim}"
        for neg_claim, neg_ref in zip(examples['negative_claim'],examples['negative_references'])
    ]
    
    # Tokenize anchors
    anchor_encodings = tokenizer(
        anchor_inputs,
        max_length=max_length,
        padding='max_length',
        truncation=True,
    )
    
    # Tokenize negatives
    neg_encodings = tokenizer(
        negative_inputs,
        max_length=max_length,
        padding='max_length',
        truncation=True,
    )
    
    # Labels for T5: tokenize target labels
    # Assuming binary: "entailment" for grounded (1), "not_entailment" for not grounded (0)
    label_texts = ["1" if l == "attributable" else "0" for l in examples['attribution_label']]
    label_encodings = tokenizer(
        label_texts,
        max_length=8,
        padding='max_length',
        truncation=True,
    )

    binary_labels = [1 if l == "attributable" else 0 for l in examples['attribution_label']]
    
    return {
        'input_ids': anchor_encodings['input_ids'],
        'attention_mask': anchor_encodings['attention_mask'],
        'labels': label_encodings['input_ids'],
        'neg_input_ids': neg_encodings['input_ids'],
        'neg_attention_mask': neg_encodings['attention_mask'],
        'original_labels': binary_labels # Added for SupCon grouping
    }

def evaluate_model(model, test_dataset, tokenizer, device='cuda', batch_size=8):
    """
    Evaluate model on test set and return detailed metrics.
    
    Args:
        model: T5 model to evaluate
        test_dataset: Dataset with 'input_ids', 'attention_mask', 'labels'
        tokenizer: T5 tokenizer
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary with accuracy, F1, precision, recall, and predictions
    """
    from torch.utils.data import DataLoader
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np
    
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    model.eval()
    model.to(device)
    
    # Set dataset format to torch
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Create dataloader
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_labels = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move to device - now batch items are tensors
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Replace -100 with pad_token_id for decoding
            labels_for_decode = labels.clone()
            labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id
            
            # Generate predictions
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=8,
                num_beams=1,  # Greedy decoding for speed
            )
            
            # Decode predictions and labels
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            refs = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
            
            # Convert to binary
            pred_binary = [1 if '1' in p.lower() else 0 for p in preds]
            label_binary = [1 if '1' in l.lower() else 0 for l in refs]
            
            all_predictions.extend(pred_binary)
            all_labels.extend(label_binary)
            
            # Progress indicator
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size} samples...")
    
    # Reset format
    test_dataset.reset_format()
    
    # Compute metrics
    results = {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'f1': f1_score(all_labels, all_predictions, average='binary'),
        'precision': precision_score(all_labels, all_predictions, average='binary'),
        'recall': recall_score(all_labels, all_predictions, average='binary'),
        'predictions': all_predictions,
        'labels': all_labels,
    }
    
    # Print detailed results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, 
                                target_names=['Not Grounded', 'Grounded']))
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"                 Predicted")
    print(f"                 Not    Grounded")
    print(f"Actual Not      {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"       Grounded {cm[1][0]:4d}    {cm[1][1]:4d}")
    print("="*50 + "\n")
    
    return results
# Example usage
def setup_training_example():
    """
    Complete example of setting up training with minimal changes.
    Includes before/after evaluation on test set.
    """
    from transformers import (
        T5ForConditionalGeneration, 
        T5Tokenizer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq
    )
    from datasets import Dataset
    
    # Load your T5-XXL NLI model
    model_name = "google/t5_xxl_true_nli_mixture"  # Replace with actual model path
    model = ContrastiveT5.from_pretrained(
        model_name,
        contrastive_weight=0.3, # Pass your args here
        temperature=0.07
    )

    #model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1
    )

    # 3. Wrap Model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if True:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads() # <--- Don't forget this!
        model.config.use_cache = False

    # Prepare your data
    # Example format - replace with your actual data loading
    data_path=os.environ['WORK']+"/attributionBench_contrastive1neg_mismatch"
    data = datasets.load_from_disk(data_path)
    train_dataset=data["train"]#.select(range(100))
    #print("train_dataset",train_dataset[0])
    
    # Test data (without negatives for evaluation)
    data_path=os.environ['WORK']+"/AttributionBench"
    data = datasets.load_from_disk(data_path)
    dev_dataset=data["dev"]#.select(range(100))
    test_dataset = data["test"]#.select(range(100))

    
    # Preprocess training data (with negatives)
    tokenized_train = train_dataset.map(
        lambda x: prepare_dataset_for_contrastive(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    # Preprocess test data (without negatives)
    def prepare_test_data(examples, tokenizer, max_length=512):
        """Prepare test data without negatives."""
        inputs = [
            f"premise: {' '.join(ref)} hypothesis: {claim}" 
            for claim, ref in zip(examples['claim'], examples['references'])
        ]
        
        encodings = tokenizer(
            inputs,
            max_length=max_length,
            padding='max_length',
            truncation=True,
        )
        label_texts = ["1" if l == 'attributable' else "0" for l in examples['attribution_label']]
        label_encodings = tokenizer(
            label_texts,
            max_length=8,
            padding='max_length',
            truncation=True,
        )
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': label_encodings['input_ids'],
        }
    
    tokenized_test = test_dataset.map(
        lambda x: prepare_test_data(x, tokenizer),
        batched=True,
        remove_columns=test_dataset.column_names
    )

    tokenized_dev= dev_dataset.map(
        lambda x: prepare_test_data(x, tokenizer),
        batched=True,
        remove_columns=dev_dataset.column_names
    )
    
    
    # ===== EVALUATE BEFORE TRAINING =====
    print("\n" + "#"*60)
    print("# BEFORE TRAINING - Baseline Performance")
    print("#"*60)
    results_before = None #evaluate_model(model, tokenized_test, tokenizer)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./output",
        remove_unused_columns=False,
        num_train_epochs=3,
        per_device_train_batch_size=1,        # Reduce from 4 to 1
        per_device_eval_batch_size=4,
        learning_rate=3e-5,  # Lower for fine-tuning large models
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=1000,
        eval_steps=1000,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        generation_max_length=8,  # Short for binary classification
        fp16=True,  # Use mixed precision for T5-XXL
        gradient_accumulation_steps=16,  # Important for large models
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # Create trainer with contrastive learning
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # ===== TRAIN MODEL =====
    print("\n" + "#"*60)
    print("# TRAINING WITH CONTRASTIVE LEARNING")
    print("#"*60)
    trainer.train()
    
    # ===== EVALUATE AFTER TRAINING =====
    print("\n" + "#"*60)
    print("# AFTER TRAINING - Final Performance")
    print("#"*60)
    results_after = evaluate_model(model, tokenized_test, tokenizer)
    
    # ===== COMPARE RESULTS =====
    print("\n" + "="*60)
    print("IMPROVEMENT SUMMARY")
    print("="*60)
    print(f"{'Metric':<15} {'Before':<12} {'After':<12} {'Improvement':<12}")
    print("-"*60)
    
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    for metric in metrics:
        before = results_before[metric]
        after = results_after[metric]
        improvement = after - before
        print(f"{metric.capitalize():<15} {before:<12.4f} {after:<12.4f} {improvement:+.4f}")
    
    print("="*60)
    
    return trainer, results_before, results_after

if __name__ == "__main__":
    setup_training_example()

# For evaluation - standard approach
def compute_metrics(eval_preds, tokenizer):
    """
    Compute metrics for evaluation during training.
    """
    predictions, labels = eval_preds
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Convert to binary predictions
    pred_labels = [1 if 'entailment' in p.lower() else 0 for p in decoded_preds]
    true_labels = [1 if 'entailment' in l.lower() else 0 for l in decoded_labels]
    
    # Compute accuracy
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    return {
        'accuracy': accuracy_score(true_labels, pred_labels),
        'f1': f1_score(true_labels, pred_labels, average='binary'),
        'precision': precision_score(true_labels, pred_labels, average='binary'),
        'recall': recall_score(true_labels, pred_labels, average='binary'),
    }