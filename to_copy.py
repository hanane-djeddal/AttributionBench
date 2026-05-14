 Traceback (most recent call last):
   File "../src/train/autoais_train_const_inbatch_neg.py", line 637, in <module>
     train()
   File "../src/train/autoais_train_const_inbatch_neg.py", line 629, in train
     trainer.train()
   File "Code/alignment-handbook/handbook/lib64/python3.11/site-packages/transformers/trainer.py", line 2328, in train
     return inner_training_loop(
            ^^^^^^^^^^^^^^^^^^^^
   File "Code/alignment-handbook/handbook/lib64/python3.11/site-packages/transformers/trainer.py", line 2672, in _inner_training_loop
     tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   File "Code/alignment-handbook/handbook/lib64/python3.11/site-packages/transformers/trainer.py", line 4009, in training_step
     loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   File "../src/train/autoais_train_const_inbatch_neg.py", line 484, in compute_loss
     embeds = get_decoder_embedding(
              ^^^^^^^^^^^^^^^^^^^^^^
   File "../src/train/autoais_train_const_inbatch_neg.py", line 165, in get_decoder_embedding
     encoder_outputs = model.encoder(
                       ^^^^^^^^^^^^^^
   File "Code/alignment-handbook/handbook/lib64/python3.11/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
     return self._call_impl(*args, **kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   File "Code/alignment-handbook/handbook/lib64/python3.11/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
     return forward_call(*args, **kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   File "Code/alignment-handbook/handbook/lib64/python3.11/site-packages/transformers/models/t5/modeling_t5.py", line 1002, in forward
     inputs_embeds = self.embed_tokens(input_ids)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   File "Code/alignment-handbook/handbook/lib64/python3.11/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
     return self._call_impl(*args, **kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   File "Code/alignment-handbook/handbook/lib64/python3.11/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
     return forward_call(*args, **kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   File "Code/alignment-handbook/handbook/lib64/python3.11/site-packages/torch/nn/modules/sparse.py", line 191, in forward
     return F.embedding(
            ^^^^^^^^^^^^
   File "Code/alignment-handbook/handbook/lib64/python3.11/site-packages/torch/nn/functional.py", line 2567, in embedding
     return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 RuntimeError: 'weight' must be 2-D