This code has been written by David Stephane (https://github.com/steph1793/Pointer_Generator_Summarizer).
We have used this code as our code base for our experiment.

This code is based on the paper : https://arxiv.org/pdf/1704.04368

To Run this code please run the below command with parameters
<br>
<br>

__python main.py \
--max_enc_len=400 \
--max_dec_len=100 \
--max_dec_steps=100 \
--min_dec_steps=30 \
--batch_size=4 \
--beam_size=4 \
--vocab_size=50000 \
--embed_size=128 \
--enc_units=256 \
--dec_units=256 \
--attn_units=512 \
--learning_rate=0.1 \
--adagrad_init_acc=0.1 \
--max_grad_norm=0.8 \
--mode="eval" \
--checkpoints_save_steps=5000 \
--max_steps=40000 \
--num_to_test=5 \
--max_num_to_eval=100 \
--vocab_path="vocab path created" \
--data_dir="Path to the training data" \
--model_path="path to the path to save the model" \
--checkpoint_dir="path to the path to save the model checkpoint details" \
--test_save_dir="Path where test file gets saved"__
