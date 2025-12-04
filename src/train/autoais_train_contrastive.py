import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer
from datasets import load_dataset, Features, Value
import datasets
import json
import os
import inspect

import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput, ModelOutput

os.environ['HF_HOME'] = os.environ['WORK'] + '/.cache/huggingface'
os.environ['WANDB_MODE'] = 'offline'

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="google/flan-t5-base")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    dev_data_path: str = field(default=None, metadata={"help": "Path to the dev data."})
    dataset_version: str = field(
        default="v3.0",
        metadata={"help": "Dataset version"}
    )
    template: str = field(default="base_c_e")
    template_path: str = field(default="src/train/template.json")


@dataclass
class Seq2SeqTrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adafactor")  # Changed to adafactor for memory efficiency
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )
    # Contrastive learning parameters
    contrastive_weight: float = field(default=0.3)
    cont_temperature: float = field(default=0.07)
    pooling_method: str = field(default="mean")  # mean, last, max



def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels

class ContrastiveT5(T5ForConditionalGeneration):
    def __init__(self, config, contrastive_weight=0.3, cont_temperature=0.07):
        super().__init__(config)
        self.contrastive_weight = contrastive_weight
        self.cont_temperature = cont_temperature
        
        # MEMORY OPTIMIZATION 1: Enable Gradient Checkpointing
        # This drastically reduces VRAM usage by re-computing activations during backward pass
        self.gradient_checkpointing_enable()
        
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
        neg_input_ids=None,       
        neg_attention_mask=None,
        original_labels=None, 
        **kwargs
    ):
        # Filter arguments for the Parent Class
        parent_args = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'output_hidden_states': True,
            'return_dict': True
        }

        for key, value in kwargs.items():
            if key in self._valid_parent_args:
                parent_args[key] = value

        # 1. Main T5 Forward (Anchor -> Generation Loss)
        outputs = super().forward(**parent_args)
        
        # Return early if not training or no negatives provided
        if not self.training or neg_input_ids is None or len(neg_input_ids) == 0:
            return outputs

        # 2. Extract Anchor Embeddings
        anchor_embeds = self.get_pooled_embedding(
            outputs.encoder_last_hidden_state, 
            attention_mask
        )

        # 3. Compute Negative Embeddings (Simplified & Optimized)
        # MEMORY OPTIMIZATION 2: Removed redundant manual embedding extraction.
        # Just run the encoder directly. This avoids creating unnecessary tensor copies.
        neg_outputs = self.encoder(
            input_ids=neg_input_ids,
            attention_mask=neg_attention_mask,
            return_dict=True
        )

        neg_embeds = self.get_pooled_embedding(
            neg_outputs.last_hidden_state, 
            neg_attention_mask
        )

        # 4. Contrastive Loss Logic
        anchor_norm = F.normalize(anchor_embeds, p=2, dim=1)
        neg_norm = F.normalize(neg_embeds, p=2, dim=1)
        
        # Cosine similarity
        sim_neg = torch.sum(anchor_norm * neg_norm, dim=1)
        
        # Margin Loss
        margin = -0.1
        contrastive_loss = torch.mean(torch.clamp(sim_neg - margin, min=0))

        # 5. Final Loss
        # Ensure we don't accidentally detach gradients
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
# =====================================================================
# IMPROVED CONTRASTIVE T5 MODEL
# =====================================================================

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


# =============================================================================
# METRICS & PREPROCESSING
# =============================================================================


def compute_metrics(eval_preds):
    print("Computing Metric")
    logits = (
        eval_preds.predictions[0]
        if isinstance(eval_preds.predictions, tuple)
        else eval_preds.predictions
    )
    max_length = 128  # Adjust as needed
    logits = logits[:, :max_length, :]  # Truncate before argmax
    preds = np.argmax(logits, axis=-1)
    labels = eval_preds.label_ids

    # Needed for global access to tokenizer if not passed explicitly
    # Ideally pass tokenizer to compute_metrics via partial
    global tokenizer 

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = [int(p.startswith(l)) for p, l in zip(decoded_preds, decoded_labels)]
    return {"accuracy": sum(result) / len(result)}


def _tokenize_fn(s: str, tokenizer: transformers.PreTrainedTokenizer, is_target=False):
    if is_target:
        token_ids = tokenizer(
            text_target=s,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids[0]
        token_ids = torch.where(token_ids == tokenizer.pad_token_id, -100, token_ids)
    else:
        token_ids = tokenizer(
            s,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids[0]

    return token_ids


class SupervisedDataset(Dataset):
    def __init__(
        self, data_args: str, tokenizer: transformers.PreTrainedTokenizer, split="train"
    ):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.dataset_path = data_args.data_path
        
        # Load Raw Data
        features = Features({
            "question": Value("string"),
            "claim": Value("string"),
            "claim_raw_string": Value("string"),
            "response": Value("string"),
            "references": datasets.Sequence(Value("string")),
            "citation_links": datasets.Sequence(Value("string")),
            "webpage_references": datasets.Sequence(Value("string")),
            "attribution_label": Value("string"),
            "src_dataset": Value("string"),
            "id": Value("string"),
            # Ensure negative columns are expected (even if null for test)
            "negative_claim": Value("string"), 
            "negative_references": datasets.Sequence(Value("string"))
        })

        data_path = os.environ['WORK'] + "/AttributionBench"
        data = datasets.load_from_disk(data_path)
        
        if split in ["stanford_dev", "attributedqa_dev", "hagrid_dev", "expertqa_dev"]:
            dataset = data["dev"]
        elif split == "train":
            data_path = os.environ['WORK'] + "/" + data_args.dataset_version
            data = datasets.load_from_disk(data_path)
            dataset = data[split]#.select(range(3000))
        else:
            data_path = os.environ['WORK'] + "/AttributionBench"
            data = datasets.load_from_disk(data_path)
            dataset = data[split]

        # Tokenize
        tokenized_dataset = dataset.map(
            self.process_function, batched=False, num_proc=4 # Increased proc for speed
        )
        
        # Filter
        filtered_dataset = tokenized_dataset.filter(
            lambda example: any([_ != -100 for _ in example["labels"]]), num_proc=4
        )
        
        logging.info(f"We cut {len(tokenized_dataset)} - {len(filtered_dataset)} instances")
        
        # Store Data
        self.input_ids = [torch.tensor(d, dtype=torch.int64) for d in filtered_dataset["input_ids"]]
        self.labels = [torch.tensor(l, dtype=torch.int64) for l in filtered_dataset["labels"]]
        
        # NEW: Store Contrastive Data
        # We need to handle cases (like test sets) where negatives might not exist
        if "neg_input_ids" in filtered_dataset.column_names:
             self.neg_input_ids = [torch.tensor(d, dtype=torch.int64) for d in filtered_dataset["neg_input_ids"]]
             self.original_labels = [torch.tensor(l, dtype=torch.int64) for l in filtered_dataset["original_labels"]]
        else:
            # Fallback for dev/test sets that don't need contrastive processing
            self.neg_input_ids = [torch.tensor([], dtype=torch.int64) for _ in range(len(filtered_dataset))]
            self.original_labels = [torch.tensor(0, dtype=torch.int64) for _ in range(len(filtered_dataset))]

        if len(self.input_ids) > 0:
            logging.info(f"{self.tokenizer.decode(self.input_ids[0], skip_special_tokens=True)}")

    def _tokenize_fn(self, text: str, minus_len: int = 0) -> Dict:
        """Tokenize a list of strings."""
        tokenized = self.tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length - minus_len,
            truncation=True,
        )
        input_ids = tokenized.input_ids[0]
        return dict(input_ids=input_ids)

    def process_function(self, example):
        # Helper to generate prompt string based on template
        def generate_prompt_text(ex_dict):
            query = ex_dict.get("question", "") or ""
            answer = ex_dict.get("claim", "") or ""
            response = ex_dict.get("response", "") or ""
            
            # Handle list vs string for references
            refs = ex_dict.get("references", [])
            documents_concatenation = "\n\n\n".join(refs) if isinstance(refs, list) else refs

            # Logic to select fields based on template flags
            have_question = "q" in self.data_args.template
            have_response = "r" in self.data_args.template
            # (Simplified logic based on your code's string check)
            if "q_c_e_r" in self.data_args.template:
                have_question, have_response = True, True
            elif "q_c_e" in self.data_args.template:
                have_question, have_response = True, False
            elif "c_e_r" in self.data_args.template:
                have_question, have_response = False, True
            
            if have_question and have_response:
                input_template = "### Input:\nQuestion: {}\n\nClaim: {}\n\nResponse: {}\n\nReference: {}\n\n### Output:"
                return input_template.format(query, answer, response, documents_concatenation)
            elif have_question and not have_response:
                input_template = "premise: {} hypothesis: {}"
                return input_template.format(documents_concatenation, f"{query} {answer}")
            elif not have_question and have_response:
                input_template = "### Input:\nClaim: {}\n\nResponse: {}\n\nReference: {}\n\n### Output:"
                return input_template.format(answer, response, documents_concatenation)
            else:
                input_template = "premise: {} hypothesis: {}"
                return input_template.format(documents_concatenation, answer)

        # 1. Generate Anchor Input
        source_text = generate_prompt_text(example)
        
        # 2. Generate Target
        target_text = "1" if f"{example['attribution_label']}" == "attributable" else "0"
        
        # 3. Tokenize Anchor & Target
        source_tokenized = self._tokenize_fn(source_text)
        target_tokenized = self._tokenize_fn(target_text) # Re-using helper, but is_target logic is simple

        # 4. Generate & Tokenize Negative (If available)
        neg_input_ids = []
        if example.get("negative_claim") and example.get("negative_references"):
            # Create a temporary dict for the negative example
            neg_example = example.copy()
            neg_example['claim'] = example['negative_claim']
            neg_example['references'] = example['negative_references']
            
            neg_source_text = generate_prompt_text(neg_example)
            neg_tokenized = self._tokenize_fn(neg_source_text)
            neg_input_ids = neg_tokenized["input_ids"]

        # 5. Get Binary Label (Integer)
        binary_label = 1 if target_text == "1" else 0

        return {
            "input_ids": source_tokenized["input_ids"],
            "labels": torch.where(target_tokenized["input_ids"] == self.tokenizer.pad_token_id, -100, target_tokenized["input_ids"]),
            "neg_input_ids": neg_input_ids,
            "original_labels": binary_label
        }

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i], 
            labels=self.labels[i],
            neg_input_ids=self.neg_input_ids[i],
            original_labels=self.original_labels[i]
        )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning with contrastive support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Unpack instances
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        neg_input_ids = [instance["neg_input_ids"] for instance in instances]
        original_labels = [instance["original_labels"] for instance in instances]

        # Pad Anchor
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        # Pad Negative
        # Handle case where neg_input_ids might be empty (e.g. dev set)
        if len(neg_input_ids[0]) > 0:
            neg_input_ids = torch.nn.utils.rnn.pad_sequence(
                neg_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            neg_attention_mask = neg_input_ids.ne(self.tokenizer.pad_token_id)
        else:
            neg_input_ids = torch.tensor([])
            neg_attention_mask = torch.tensor([])

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            neg_input_ids=neg_input_ids,
            neg_attention_mask=neg_attention_mask,
            original_labels=torch.tensor(original_labels, dtype=torch.long)
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    split_train = "train"
    split_eval = "dev"
    split_eval_ood = "test_ood"

    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_args=data_args, split=split_train
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    eval_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_args=data_args, split=split_eval
    )
    eval_dataset_stanford = SupervisedDataset(
        tokenizer=tokenizer, data_args=data_args, split="stanford_dev"
    )
    eval_dataset_hagrid = SupervisedDataset(
        tokenizer=tokenizer, data_args=data_args, split="hagrid_dev"
    )
    eval_dataset_attributedqa = SupervisedDataset(
        tokenizer=tokenizer, data_args=data_args, split="attributedqa_dev"
    )
    eval_dataset_expertqa = SupervisedDataset(
        tokenizer=tokenizer, data_args=data_args, split="expertqa_dev"
    )
    eval_dataset_ood = SupervisedDataset(
        tokenizer=tokenizer, data_args=data_args, split=split_eval_ood
    )
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, Seq2SeqTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # Suppress wandb
    # training_args.report_to = "wandb" #[] 
    # print("Loading Model")
    # with open(data_args.template_path) as f:
    #     template = json.load(f)

    # model = ContrastiveT5.from_pretrained(
    #         model_args.model_name_or_path,
    #         cache_dir=training_args.cache_dir,
    #         contrastive_weight=0.3, # Pass your args here
    #         temperature=0.07
    #     )

    # # model = transformers.T5ForConditionalGeneration.from_pretrained(
    # #     model_args.model_name_or_path,
    # #     cache_dir=training_args.cache_dir,
    # # )

    # global tokenizer
    # tokenizer = transformers.T5Tokenizer.from_pretrained(
    #     model_args.model_name_or_path,
    #     cache_dir=training_args.cache_dir,
    #     model_max_length=training_args.model_max_length,
    #     padding_side="right",
    #     use_fast=False,
    # )

    # data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    # trainer = Seq2SeqTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     compute_metrics=compute_metrics,
    #     args=training_args,
    #     **data_module,
    # )
    # trainer.train()
    # print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    # trainer.save_state()
    # trainer.save_model(output_dir=training_args.output_dir)
    if training_args.gradient_checkpointing is False:
        print("Forcing gradient_checkpointing=True for memory optimization")
        training_args.gradient_checkpointing = True
        
    # Ensure FP16 is ON if CUDA is available (Halves memory usage)
    if torch.cuda.is_available():
        print("CUDA detected: Enabling FP16 for memory optimization")
        training_args.fp16 = True
    
    print("Loading Model")
    # Load template logic
    try:
        with open(data_args.template_path) as f:
            template = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Template file not found at {data_args.template_path}")

    model = ContrastiveT5.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            contrastive_weight=training_args.contrastive_weight, #0.3, # Pass your args here
            cont_temperature=training_args.cont_temperature,#0.07
        )

    # Global tokenizer (referenced in compute_metrics)
    global tokenizer
    tokenizer = transformers.T5Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    if torch.cuda.is_available():
        print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        args=training_args,
        **data_module,
    )
    
    trainer.train()
    
    if torch.cuda.is_available():
        print(f"Final GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()
