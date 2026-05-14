
# class QwenContrastiveDataset(Dataset):
#     """
#     Dataset for contrastive learning with Qwen3-4B.
#     Uses the prompt format: CLAIM + DOCUMENT -> CLASSIFICATION
#     """
#     def __init__(
#         self, 
#         data_args: DataArguments,
#         tokenizer: transformers.PreTrainedTokenizer,
#         max_seq_length: int = 4096,
#         split="train"
#     ):
#         self.data_args = data_args
#         self.tokenizer = tokenizer
#         self.max_seq_length = max_seq_length
#         self.split = split
#         self.use_contrastive = data_args.use_contrastive and split == "train"
        
#         # Qwen3 prompt template
#         self.prompt_template = """You will be given a claim and a document. Determine whether the claim is 'GROUNDED' or 'NOT GROUNDED' based on the document.A 'GROUNDED' claim is fully supported by the information provided in the document. It should be directly verifiable from the document. Only return the classification as the answer: 1 for 'GROUNDED' or 0 for 'NOT GROUNDED' without any explanation.
# CLAIM: {} 
# DOCUMENT: {}
# CLASSIFICATION:"""
        
#         self.data = self.load_dataset(split, data_args)
#         logging.info(f"Loaded {len(self.data)} examples for {split}")
#         if self.use_contrastive:
#             logging.info(f"Contrastive enabled: {data_args.num_positives} pos, {data_args.num_negatives} neg")
    
#     def load_dataset(self, split, data_args):
#         """Load dataset from disk"""
#         data_path = os.environ.get('WORK', '.') + "/AttributionBench"
        
#         if split == "train" and self.use_contrastive:
#             # Load contrastive dataset with anchor/positives/negatives
#             data_path = os.environ.get('WORK', '.') + "/" + data_args.dataset_version
#             data = datasets.load_from_disk(data_path)
#             return data[split]
#         else:
#             # Load regular dataset
#             data = datasets.load_from_disk(data_path)
#             return data[split] if split in data else data["dev"]
    
#     def format_example(self, example):
#         """Format example using Qwen3 prompt template"""
#         claim = example.get("claim", "")
#         if claim in ["nan", "", None]:
#             claim = ""
        
#         # Concatenate references as document
#         documents = example.get("references", [])
#         document = "\n\n".join(documents) if documents else ""
        
#         # Format prompt (without classification for input)
#         prompt = self.prompt_template.format(claim, document)
        
#         # Get label
#         label = "1" if str(example.get('attribution_label', '')) == "attributable" else "0"
        
#         # Complete text with label for training
#         full_text = prompt + label
        
#         return prompt, label, full_text
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         example = self.data[idx]
        
#         if not self.use_contrastive:
#             # Regular mode: just return formatted example
#             prompt, label, full_text = self.format_example(example)
#             return {
#                 "text": full_text,
#                 "prompt": prompt,
#                 "label": label
#             }
        
#         # Check if anchor-only (no positives/negatives)
#         has_positives = "positives" in example and len(example["positives"]) > 0
#         has_negatives = "negatives" in example and len(example["negatives"]) > 0
#         is_anchor_only = not has_positives and not has_negatives
        
#         if is_anchor_only:
#             # Anchor-only: classification only
#             anchor = example.get('anchor', example)
#             prompt, label, full_text = self.format_example(anchor)
            
#             return {
#                 "text": full_text,
#                 "prompt": prompt,
#                 "label": label,
#                 "positives": [],
#                 "negatives": [],
#                 "is_anchor_only": True
#             }
        
#         # Contrastive mode: anchor + sampled positives + negatives
#         anchor = example.get("anchor", example)
#         positives = example["positives"]
#         negatives = example["negatives"]
        
#         # Sample subset
#         num_pos = min(self.data_args.num_positives, len(positives))
#         num_neg = min(self.data_args.num_negatives, len(negatives))
        
#         sampled_positives = random.sample(positives, num_pos) if len(positives) > 0 else []
#         sampled_negatives = random.sample(negatives, num_neg) if len(negatives) > 0 else []
        
#         # Format anchor
#         anchor_prompt, anchor_label, anchor_text = self.format_example(anchor)
        
#         # Format positives
#         pos_data = []
#         for pos in sampled_positives:
#             prompt, label, full_text = self.format_example(pos)
#             pos_data.append({"text": full_text, "prompt": prompt, "label": label})
        
#         # Format negatives
#         neg_data = []
#         for neg in sampled_negatives:
#             prompt, label, full_text = self.format_example(neg)
#             neg_data.append({"text": full_text, "prompt": prompt, "label": label})
        
#         return {
#             "text": anchor_text,
#             "prompt": anchor_prompt,
#             "label": anchor_label,
#             "positives": pos_data,
#             "negatives": neg_data,
#             "is_anchor_only": False
#         }


# class QwenContrastiveCollator:
#     """
#     Custom collator for Qwen3 contrastive learning.
#     Handles variable-length positives/negatives.
#     """
#     def __init__(self, tokenizer, max_seq_length=4096, use_contrastive=True):
#         self.tokenizer = tokenizer
#         self.max_seq_length = max_seq_length
#         self.use_contrastive = use_contrastive
    
#     def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
#         # Tokenize anchors
#         anchor_texts = [inst["text"] for inst in instances]
#         anchor_encodings = self.tokenizer(
#             anchor_texts,
#             padding=True,
#             truncation=True,
#             max_length=self.max_seq_length,
#             return_tensors="pt"
#         )
        
#         # Extract labels from the tokenized text
#         anchor_labels_text = [inst["label"] for inst in instances]
#         anchor_labels = torch.tensor([int(l) for l in anchor_labels_text])
        
#         if not self.use_contrastive:
#             return {
#                 "input_ids": anchor_encodings.input_ids,
#                 "attention_mask": anchor_encodings.attention_mask,
#                 "labels": anchor_encodings.input_ids.clone(),  # For language modeling
#                 "binary_labels": anchor_labels
#             }
        
#         # Check if batch has contrastive data
#         has_contrastive = any(
#             len(inst.get("positives", [])) > 0 or len(inst.get("negatives", [])) > 0
#             for inst in instances
#         )
        
#         if not has_contrastive:
#             return {
#                 "input_ids": anchor_encodings.input_ids,
#                 "attention_mask": anchor_encodings.attention_mask,
#                 "labels": anchor_encodings.input_ids.clone(),
#                 "binary_labels": anchor_labels
#             }
        
#         # Collect all positives
#         all_pos_texts = []
#         pos_counts = []
#         for inst in instances:
#             positives = inst.get("positives", [])
#             pos_counts.append(len(positives))
#             all_pos_texts.extend([p["text"] for p in positives])
        
#         max_pos = max(pos_counts) if pos_counts else 0
        
#         # Tokenize positives
#         if max_pos > 0 and len(all_pos_texts) > 0:
#             pos_encodings = self.tokenizer(
#                 all_pos_texts,
#                 padding=True,
#                 truncation=True,
#                 max_length=self.max_seq_length,
#                 return_tensors="pt"
#             )
            
#             # Reshape to [batch, max_pos, seq_len]
#             pos_input_ids_list = []
#             pos_attention_mask_list = []
#             idx = 0
#             for count in pos_counts:
#                 if count == 0:
#                     # Dummy positives
#                     dummy_ids = torch.zeros((max_pos, pos_encodings.input_ids.shape[1]), dtype=torch.long)
#                     dummy_mask = torch.zeros((max_pos, pos_encodings.attention_mask.shape[1]), dtype=torch.long)
#                     pos_input_ids_list.append(dummy_ids)
#                     pos_attention_mask_list.append(dummy_mask)
#                 else:
#                     batch_pos_ids = pos_encodings.input_ids[idx:idx+count]
#                     batch_pos_mask = pos_encodings.attention_mask[idx:idx+count]
                    
#                     if count < max_pos:
#                         pad_amount = max_pos - count
#                         pad_ids = torch.zeros((pad_amount, batch_pos_ids.shape[1]), dtype=torch.long)
#                         pad_mask = torch.zeros((pad_amount, batch_pos_mask.shape[1]), dtype=torch.long)
#                         batch_pos_ids = torch.cat([batch_pos_ids, pad_ids], dim=0)
#                         batch_pos_mask = torch.cat([batch_pos_mask, pad_mask], dim=0)
                    
#                     pos_input_ids_list.append(batch_pos_ids)
#                     pos_attention_mask_list.append(batch_pos_mask)
#                     idx += count
            
#             pos_input_ids = torch.stack(pos_input_ids_list)
#             pos_attention_mask = torch.stack(pos_attention_mask_list)
#         else:
#             pos_input_ids = None
#             pos_attention_mask = None
        
#         # Collect all negatives
#         all_neg_texts = []
#         neg_counts = []
#         for inst in instances:
#             negatives = inst.get("negatives", [])
#             neg_counts.append(len(negatives))
#             all_neg_texts.extend([n["text"] for n in negatives])
        
#         max_neg = max(neg_counts) if neg_counts else 0
        
#         # Tokenize negatives
#         if max_neg > 0 and len(all_neg_texts) > 0:
#             neg_encodings = self.tokenizer(
#                 all_neg_texts,
#                 padding=True,
#                 truncation=True,
#                 max_length=self.max_seq_length,
#                 return_tensors="pt"
#             )
            
#             neg_input_ids_list = []
#             neg_attention_mask_list = []
#             idx = 0
#             for count in neg_counts:
#                 if count == 0:
#                     dummy_ids = torch.zeros((max_neg, neg_encodings.input_ids.shape[1]), dtype=torch.long)
#                     dummy_mask = torch.zeros((max_neg, neg_encodings.attention_mask.shape[1]), dtype=torch.long)
#                     neg_input_ids_list.append(dummy_ids)
#                     neg_attention_mask_list.append(dummy_mask)
#                 else:
#                     batch_neg_ids = neg_encodings.input_ids[idx:idx+count]
#                     batch_neg_mask = neg_encodings.attention_mask[idx:idx+count]
                    
#                     if count < max_neg:
#                         pad_amount = max_neg - count
#                         pad_ids = torch.zeros((pad_amount, batch_neg_ids.shape[1]), dtype=torch.long)
#                         pad_mask = torch.zeros((pad_amount, batch_neg_mask.shape[1]), dtype=torch.long)
#                         batch_neg_ids = torch.cat([batch_neg_ids, pad_ids], dim=0)
#                         batch_neg_mask = torch.cat([batch_neg_mask, pad_mask], dim=0)
                    
#                     neg_input_ids_list.append(batch_neg_ids)
#                     neg_attention_mask_list.append(batch_neg_mask)
#                     idx += count
            
#             neg_input_ids = torch.stack(neg_input_ids_list)
#             neg_attention_mask = torch.stack(neg_attention_mask_list)
#         else:
#             neg_input_ids = None
#             neg_attention_mask = None
        
#         return {
#             "input_ids": anchor_encodings.input_ids,
#             "attention_mask": anchor_encodings.attention_mask,
#             "labels": anchor_encodings.input_ids.clone(),
#             "binary_labels": anchor_labels,
#             "pos_input_ids": pos_input_ids,
#             "pos_attention_mask": pos_attention_mask,
#             "neg_input_ids": neg_input_ids,
#             "neg_attention_mask": neg_attention_mask,
#         }


# class QwenContrastiveTrainer(SFTTrainer):
#     """
#     Custom trainer combining classification + contrastive learning for Qwen3.
#     Preserves pre-trained knowledge through careful loss balancing.
#     """
#     def __init__(
#         self, 
#         *args,
#         contrastive_loss_fn=None,
#         contrastive_weight=0.15,
#         classification_weight=1.0,
#         contrastive_warmup_steps=500,
#         **kwargs
#     ):
#         super().__init__(*args, **kwargs)
#         self.contrastive_loss_fn = contrastive_loss_fn
#         self.base_contrastive_weight = contrastive_weight
#         self.contrastive_weight = contrastive_weight
#         self.classification_weight = classification_weight
#         self.contrastive_warmup_steps = contrastive_warmup_steps
        
#         logging.info(f"Contrastive Trainer initialized:")
#         logging.info(f"  - Base contrastive weight: {self.base_contrastive_weight}")
#         logging.info(f"  - Classification weight: {self.classification_weight}")
#         logging.info(f"  - Warmup steps: {self.contrastive_warmup_steps}")
    
#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         """
#         Compute combined loss: classification + contrastive.
#         Implements warmup for contrastive weight.
#         """
#         # Apply contrastive warmup
#         if self.state.global_step < self.contrastive_warmup_steps:
#             warmup_factor = self.state.global_step / self.contrastive_warmup_steps
#             self.contrastive_weight = self.base_contrastive_weight * warmup_factor
#         else:
#             self.contrastive_weight = self.base_contrastive_weight
        
#         # Extract contrastive inputs
#         pos_input_ids = inputs.pop("pos_input_ids", None)
#         pos_attention_mask = inputs.pop("pos_attention_mask", None)
#         neg_input_ids = inputs.pop("neg_input_ids", None)
#         neg_attention_mask = inputs.pop("neg_attention_mask", None)
#         binary_labels = inputs.pop("binary_labels", None)
        
#         # Forward pass for classification
#         outputs = model(**inputs)
#         classification_loss = outputs.loss
        
#         # If no contrastive data, return only classification loss
#         if pos_input_ids is None or pos_input_ids.numel() == 0 or pos_input_ids.shape[1] == 0:
#             return (classification_loss, outputs) if return_outputs else classification_loss
        
#         # Extract hidden states for contrastive learning
#         anchor_hidden = get_qwen_hidden_state(
#             model,
#             inputs["input_ids"],
#             inputs["attention_mask"]
#         )
        
#         # Get positives hidden states
#         batch_size, num_pos, seq_len = pos_input_ids.shape
#         pos_input_ids_flat = pos_input_ids.view(-1, seq_len)
#         pos_attention_mask_flat = pos_attention_mask.view(-1, seq_len)
        
#         pos_hidden_flat = get_qwen_hidden_state(
#             model,
#             pos_input_ids_flat,
#             pos_attention_mask_flat
#         )
#         pos_hidden = pos_hidden_flat.view(batch_size, num_pos, -1)
        
#         # Get negatives hidden states (if available)
#         neg_hidden = None
#         if neg_input_ids is not None and neg_input_ids.numel() > 0 and neg_input_ids.shape[1] > 0:
#             batch_size, num_neg, seq_len = neg_input_ids.shape
#             neg_input_ids_flat = neg_input_ids.view(-1, seq_len)
#             neg_attention_mask_flat = neg_attention_mask.view(-1, seq_len)
            
#             neg_hidden_flat = get_qwen_hidden_state(
#                 model,
#                 neg_input_ids_flat,
#                 neg_attention_mask_flat
#             )
#             neg_hidden = neg_hidden_flat.view(batch_size, num_neg, -1)
        
#         # Compute contrastive loss
#         contrastive_loss = self.contrastive_loss_fn(
#             anchor_embeds=anchor_hidden,
#             positive_embeds=pos_hidden,
#             negative_embeds=neg_hidden,
#             anchor_labels=binary_labels,
#             all_batch_embeds=anchor_hidden,
#             all_batch_labels=binary_labels
#         )
        
#         # Combined loss
#         total_loss = (
#             self.classification_weight * classification_loss +
#             self.contrastive_weight * contrastive_loss
#         )
        
#         # Logging
#         if self.state.global_step % 10 == 0:
#             has_explicit_neg = neg_hidden is not None
#             logging.info(
#                 f"Step {self.state.global_step}: "
#                 f"Total={total_loss:.4f}, "
#                 f"Class={classification_loss:.4f}, "
#                 f"Contr={contrastive_loss:.4f} "
#                 f"(weight={self.contrastive_weight:.4f}, explicit_neg={has_explicit_neg})"
#             )
        
#         return (total_loss, outputs) if return_outputs else total_loss
