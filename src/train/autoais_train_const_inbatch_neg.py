import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer
from datasets import load_dataset, Features, Value
import datasets
import json
import os
import random

os.environ['HF_HOME'] = os.environ['WORK'] + '/.cache/huggingface'
os.environ['WANDB_MODE'] = 'offline'

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
SHOW_BATCH_SIZE= 0

T5_TYPES=["claim_erronous_change","claim_numerical_mismatch","modify_passage-add_relevant_to_claim","claim_combine_facts","claim_add_to_the_claim_contradicting_info","modify_passage-add_contradiction","modify_passage-add_conflicting_sources","claim_infer_claim","claim_over_infer_claim"]

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="google/t5_xxl_true_nli_mixture")

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
    error_type: str = field(default=None)
    template: str = field(default="base_c_e")
    template_path: str = field(default="src/train/template.json")
    use_contrastive: bool = field(default=True, metadata={"help": "Use contrastive learning"})
    num_positives: int = field(default=8, metadata={"help": "Number of positives to sample per anchor"})
    num_negatives: int = field(default=8, metadata={"help": "Number of negatives to sample per anchor"})
    include_anchor_only: bool = field(
        default=True, 
        metadata={"help": "Include examples with no positives/negatives (anchor-only)"}
    )    
    filter_error_types: bool = field(
        default=False, 
        metadata={"help": "Filter error types only leaving select ones"}
    )

@dataclass
class Seq2SeqTrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )
    contrastive_weight: float = field(
        default=0.5, 
        metadata={"help": "Weight for contrastive loss (lambda1)"}
    )
    classification_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for classification loss (lambda2)"}
    )
    contrastive_temperature: float = field(
        default=0.07,
        metadata={"help": "Temperature for contrastive loss"}
    )
    use_decoder_embedding: bool = field(
        default=True,
        metadata={"help": "Use decoder's hidden state (True) or encoder's first token (False) for contrastive learning"}
    )
    use_in_batch_negatives: bool = field(
        default=True,
        metadata={"help": "Use in-batch negatives when explicit negatives are missing"}
    )


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss for T5 hidden states.
    Pulls positives closer, pushes negatives away.
    Supports in-batch negatives when explicit negatives are missing.
    """
    def __init__(self, temperature=0.07, use_in_batch_negatives=True):
        super().__init__()
        self.temperature = temperature
        self.use_in_batch_negatives = use_in_batch_negatives
        print("Setting temp to:", self.temperature )
    
    def forward(self, embeds, labels):
        """
        SupCon loss with graceful fallback when no valid pairs exist.
        """
        batch_size = embeds.shape[0]
        print(f"Batch size {batch_size}")

        # Need at least 2 examples to form any pair
        if batch_size < 2:
            logging.info(f"Less than 2 examples in the batch {batch_size}")
            return torch.tensor(0.0, device=embeds.device, requires_grad=True)

        # L2 normalize
        embeds = F.normalize(embeds, dim=-1)

        # Cosine similarity matrix scaled by temperature
        sim_matrix = torch.matmul(embeds, embeds.T) / self.temperature  # [B, B]

        # Build masks
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=embeds.device)
        labels_row = labels.unsqueeze(1)
        labels_col = labels.unsqueeze(0)
        positive_mask = (labels_row == labels_col) & ~self_mask  # [B, B]
        negative_mask = (labels_row != labels_col)               # [B, B]

        # Check if any valid contrastive pairs exist at all
        # Need at least one anchor that has both a positive AND a negative
        valid_anchors = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        if not valid_anchors.any():
            # e.g. all-same-label batch, or batch_size==1
            return torch.tensor(0.0, device=embeds.device, requires_grad=True)

        # Numerical stability
        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True).values.detach()

        exp_sim = torch.exp(sim_matrix)

        # Denominator: sum over all non-self pairs
        denom = (exp_sim * ~self_mask).sum(dim=1, keepdim=True)  # [B, 1]

        log_prob = sim_matrix - torch.log(denom + 1e-8)          # [B, B]

        # Average log-prob over positives, only for valid anchors
        num_positives = positive_mask.sum(dim=1).float()          # [B]
        loss_per_anchor = -(log_prob * positive_mask).sum(dim=1)  # [B]

        loss = (loss_per_anchor[valid_anchors] / num_positives[valid_anchors]).mean()

        return loss


def get_decoder_embedding(model, input_ids, attention_mask):
    """
    Get sentence embedding from T5 decoder's first-step hidden state.
    This uses the exact representation the model uses for classification,
    ensuring no disruption to pre-trained knowledge.
    
    Args:
        model: T5 model
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
    Returns:
        embedding: [batch_size, hidden_dim]
    """
    # Encode input
    model_to_use = model.module if hasattr(model, "module") else model
    
    # Use model_to_use for the calls below encoder_outputs = model.encoder(
    encoder_outputs = model_to_use.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True
    )
    
    # Prepare decoder start token
    decoder_start_token_id = model.config.decoder_start_token_id
    decoder_input_ids = torch.full(
        (input_ids.shape[0], 1),
        decoder_start_token_id,
        dtype=torch.long,
        device=input_ids.device
    )
    
    # Get decoder's first-step hidden state
    decoder_outputs = model.decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=encoder_outputs.last_hidden_state,
        encoder_attention_mask=attention_mask,
        return_dict=True
    )
    
    # Return first token's hidden state (sentence representation)
    return decoder_outputs.last_hidden_state[:, 0, :]  # [B, D]


def get_encoder_first_token(hidden_states):
    """
    Alternative: Use encoder's first token as sentence embedding.
    Simpler but less aligned with classification task than decoder.
    
    Args:
        hidden_states: [batch_size, seq_len, hidden_dim]
    Returns:
        embedding: [batch_size, hidden_dim]
    """
    return hidden_states[:, 0, :]  # [B, D]


class ContrastiveDataset(Dataset):
    """
    Dataset that loads anchor + positives + negatives for contrastive learning.
    """
    def __init__(
        self, 
        data_args: DataArguments, 
        tokenizer: transformers.PreTrainedTokenizer, 
        split="train"
    ):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.split = split
        self.dataset_path = data_args.data_path

        self.use_contrastive = data_args.use_contrastive and split == "train"

        self.filter_error_types = data_args.filter_error_types 
        
        # Load dataset
        self.data = self.load_dataset(split, data_args)
        
        logging.info(f"Loaded {len(self.data)} examples for {split}")
        if self.use_contrastive:
            logging.info(f"Contrastive learning enabled: sampling {data_args.num_positives} pos, {data_args.num_negatives} neg")
    
    def load_dataset(self, split, data_args):
        features = Features(
            {
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
            }
        )
        # Load the dataset
        data_path=os.environ['WORK']+"/AttributionBench"
        data = datasets.load_from_disk(data_path)
        if split in ["stanford_dev", "attributedqa_dev", "hagrid_dev", "expertqa_dev"]:
            dataset=data["dev"] 
        elif split == "train":
            # Load augmented dataset with positives/negatives
            # Assuming format: {"anchor": {...}, "positives": [...], "negatives": [...]}
            data_path=os.environ['WORK']+"/"+ data_args.dataset_version #"/AttributionBench"
            data = datasets.load_from_disk(data_path)
            dataset=data[split]
        else:
            data_path=os.environ['WORK']+"/AttributionBench"
            data = datasets.load_from_disk(data_path)

            dataset=data[split] 
        return dataset



    def process_function(self, example):
        def format_prompt(
            example,
            have_question=False,
            have_response=False,
            prompt_name=self.data_args.template,
        ):
            query = (
                example["question"]
                if example["question"] and example["question"] not in ["nan", "", None]
                else ""
            )
            answer = (
                example["claim"]
                if example["claim"] and example["claim"] not in ["nan", "", None]
                else ""
            )
            response = (
                example["response"]
                if example["response"] and example["response"] not in ["nan", "", None]
                else ""
            )
            if "references" in example.keys() and len(example["references"]):
                documents_concatenation = "\n\n\n".join(example["references"])
            else:
                print("Empty references",example)

            if have_question and have_response:
                input_template = "### Input:\nQuestion: {}\n\nClaim: {}\n\nResponse: {}\n\nReference: {}\n\n### Output:"
                input = input_template.format(
                    query, answer, response, documents_concatenation
                )
            elif have_question and not have_response:
                input_template = "premise: {} hypothesis: {}"
                input = input_template.format(documents_concatenation, " ".join(query, answer))
                # input_template = "### Input:\nQuestion: {}\n\nClaim: {}\n\nReference: {}\n\n### Output:"
                # input = input_template.format(query, answer, documents_concatenation)
            elif not have_question and have_response:
                input_template = "### Input:\nClaim: {}\n\nResponse: {}\n\nReference: {}\n\n### Output:"
                input = input_template.format(answer, response, documents_concatenation)
            else:
                input_template = "premise: {} hypothesis: {}"
                input = input_template.format(documents_concatenation, answer)

            instructions = json.load(open(self.data_args.template_path))
            # formatted_prompt = "{}{}".format(instructions[prompt_name]["llama2"], input)
            formatted_prompt = input

            return formatted_prompt

        if "q_c_e_r" in self.data_args.template:
            have_question = True
            have_response = True
        elif "q_c_e" in self.data_args.template:
            have_question = True
            have_response = False
        elif "c_e_r" in self.data_args.template:
            have_question = False
            have_response = True
        else:
            have_question = False
            have_response = False

        source = format_prompt(
            example,
            have_question=have_question,
            have_response=have_response,
            prompt_name=self.data_args.template,
        )
        return source
    
    def tokenize_example(self, text, is_target=False):
        """Tokenize text."""
        if is_target:
            token_ids = self.tokenizer(
                text_target=text,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ).input_ids[0]
            token_ids = torch.where(token_ids == self.tokenizer.pad_token_id, -100, token_ids)
        else:
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            )
            token_ids = encoding.input_ids[0]
            attention_mask = encoding.attention_mask[0]
            return token_ids, attention_mask
        
        return token_ids
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        

        anchor = example['anchor'] if "anchor" in example else example
        anchor_text = self.process_function(anchor)
        input_ids, attention_mask = self.tokenize_example(anchor_text)
            
        label = "1" if str(anchor.get('attribution_label', '')) == "attributable" else "0"
        labels = self.tokenize_example(label, is_target=True)
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pos_input_ids": [],  # Empty positives
            "pos_attention_masks": [],
            "neg_input_ids": [],  # Empty negatives
            "neg_attention_masks": [],
            "is_anchor_only": True  # Flag for trainer
        }


@dataclass
class ContrastiveDataCollator:
    """Collator for contrastive learning batches. Handles variable-length pos/neg lists."""
    tokenizer: transformers.PreTrainedTokenizer
    use_contrastive: bool = True
    
    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        # Pad anchors
        anchor_input_ids = [inst["input_ids"] for inst in instances]
        anchor_attention_masks = [inst["attention_mask"] for inst in instances]
        anchor_labels = [inst["labels"] for inst in instances]
        
        anchor_input_ids = torch.nn.utils.rnn.pad_sequence(
            anchor_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        anchor_attention_masks = torch.nn.utils.rnn.pad_sequence(
            anchor_attention_masks, batch_first=True, padding_value=0
        )
        anchor_labels = torch.nn.utils.rnn.pad_sequence(
            anchor_labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        # If not using contrastive or no positives/negatives in batch, return simple batch
        if not self.use_contrastive:
            return {
                "input_ids": anchor_input_ids,
                "attention_mask": anchor_attention_masks,
                "labels": anchor_labels
            }
        
        # Check if any instance has contrastive data
        has_contrastive = any(
            len(inst.get("pos_input_ids", [])) > 0 or len(inst.get("neg_input_ids", [])) > 0
            for inst in instances
        )
        
        if not has_contrastive:
            # All anchor-only: return classification-only batch
            return {
                "input_ids": anchor_input_ids,
                "attention_mask": anchor_attention_masks,
                "labels": anchor_labels
            }
        

        pos_input_ids = None
        pos_attention_masks = None
        
  
        neg_input_ids = None
        neg_attention_masks = None
        
        return {
            "input_ids": anchor_input_ids,
            "attention_mask": anchor_attention_masks,
            "labels": anchor_labels,
            "pos_input_ids": pos_input_ids,
            "pos_attention_masks": pos_attention_masks,
            "neg_input_ids": neg_input_ids,
            "neg_attention_masks": neg_attention_masks,
        }



class ContrastiveTrainer(Seq2SeqTrainer):
    """
    Custom trainer that combines classification loss with contrastive loss.
    Uses decoder's hidden state for contrastive learning to preserve pre-trained knowledge.
    """
    def __init__(self, *args, contrastive_loss_fn=None, contrastive_weight=0.5, 
                 classification_weight=1.0, use_decoder_embedding=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_loss_fn = contrastive_loss_fn
        self.contrastive_weight = contrastive_weight
        self.classification_weight = classification_weight
        self.use_decoder_embedding = use_decoder_embedding
        self.custom_loss_tracker = {'classification': [], 'contrastive': []}
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute combined loss: classification + supervised contrastive (in-batch only).
        """
        # Regular forward pass for classification
        outputs = model(**inputs)
        classification_loss = outputs.loss

        # Decode labels to get 0/1 anchor labels
        labels_decoded = inputs["labels"].clone()
        labels_decoded[labels_decoded == -100] = self.tokenizer.pad_token_id
        labels_text = self.tokenizer.batch_decode(labels_decoded, skip_special_tokens=True)
        anchor_labels = torch.tensor(
            [int(l.strip() == "1") for l in labels_text],
            device=inputs["labels"].device
        )

        # Get embeddings for all examples in the batch
        print("Stats:   inputs[input_ids]",   inputs["input_ids"])
        if self.use_decoder_embedding:
            embeds = get_decoder_embedding(
                model,
                inputs["input_ids"],
                inputs["attention_mask"]
            )  # [B, D]
        else:
            encoder_outputs = model.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True
            )
            embeds = get_encoder_first_token(encoder_outputs.last_hidden_state)  # [B, D]

        # Compute supervised contrastive loss with in-batch negatives
        contrastive_loss = self.contrastive_loss_fn(embeds, anchor_labels)

        # Combined loss
        total_loss = (
            self.classification_weight * classification_loss +
            self.contrastive_weight * contrastive_loss
        )



        if self.state.global_step % 10 == 0:
            print(
                    f"Step {self.state.global_step}: "
                    f"Total={total_loss:.4f}, "
                    f"Classification={classification_loss:.4f}, "
                    f"Contrastive={contrastive_loss:.4f}"
            )
            # 2. Append the detached, scalar values (.item()) to avoid memory leaks
            self.custom_loss_tracker['classification'].append(classification_loss.item())
            self.custom_loss_tracker['contrastive'].append(contrastive_loss.item())
        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs: Dict[str, float]) -> None:
        """
        Override the log method to inject custom metrics into the standard output dictionary.
        """
        # 3. Calculate the average of the custom losses since the last log step
        if len(self.custom_loss_tracker['classification']) > 0:
            logs["class_loss"] = sum(self.custom_loss_tracker['classification']) / len(self.custom_loss_tracker['classification'])
            logs["cont_loss"] = sum(self.custom_loss_tracker['contrastive']) / len(self.custom_loss_tracker['contrastive'])
            
            # Clear the trackers for the next logging window
            self.custom_loss_tracker['classification'] = []
            self.custom_loss_tracker['contrastive'] = []

        # 4. Call the parent class to handle the actual printing/logging
        super().log(logs)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    print("Computing Metric")
    logits = (
        eval_preds.predictions[0]
        if isinstance(eval_preds.predictions, tuple)
        else eval_preds.predictions
    )
    max_length = 128
    logits = logits[:, :max_length, :]
    preds = np.argmax(logits, axis=-1)
    labels = eval_preds.label_ids
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = [int(p.startswith(l)) for p, l in zip(decoded_preds, decoded_labels)]
    return {"accuracy": sum(result) / len(result)}


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning with contrastive learning."""
    split_train = "train"
    split_eval = "dev"
    split_eval_ood = "test_ood"

    print("train preparation")
    train_dataset = ContrastiveDataset(
        tokenizer=tokenizer, data_args=data_args, split="train"
    )
    

    data_collator = ContrastiveDataCollator(
        tokenizer=tokenizer, 
        use_contrastive=data_args.use_contrastive
    )
    
    # Eval datasets use regular format (no contrastive)
    eval_data_args = copy.deepcopy(data_args)
    eval_data_args.use_contrastive = False
    
    print("dev preparation")
    eval_dataset = ContrastiveDataset(
        tokenizer=tokenizer, data_args=eval_data_args, split="dev"
    )
    
    eval_collator = ContrastiveDataCollator(
        tokenizer=tokenizer,
        use_contrastive=False
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
    training_args.report_to = "wandb" #[] 

    with open(data_args.template_path) as f:
        template = json.load(f)

    model = transformers.T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    
    global tokenizer
    tokenizer = transformers.T5Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # Initialize contrastive loss
    contrastive_loss_fn = SupervisedContrastiveLoss(
        temperature=training_args.contrastive_temperature
    )
    
    print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Use custom trainer
    trainer = ContrastiveTrainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        args=training_args,
        contrastive_loss_fn=contrastive_loss_fn,
        contrastive_weight=training_args.contrastive_weight,
        classification_weight=training_args.classification_weight,
        use_decoder_embedding=training_args.use_decoder_embedding,
        **data_module,
    )
    
    trainer.train()
    print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    print(f"Model Saved to : {training_args.output_dir}")


if __name__ == "__main__":
    train()
