This is the extension work on the Base Model. The model RNN is GRU. Here we have changed the architechture to LSTM.


Python files Where CHanges have been made:

1. layers.py 
  * Class Encoder
    - __init__
    - call
    - initialize_hidden_state
  * Class Decoder
    - __init__
    - call
2. model.py
  * PGN
    - __init__
    - call_encoder
    - call
3. training_helper.py
    - train_step
    
4. batcher 
  * Vocab
    - output_to_words

To Run the code please run the below command at the console:

__python main.py \
--max_enc_len=400 \
--max_dec_len=100 \
--max_dec_steps=120 \
--min_dec_steps=30 \
--batch_size=4 \
--beam_size=4 \
--vocab_size=50000 \
--embed_size=128 \
--enc_units=256 \
--dec_units=256 \
--attn_units=512 \
--learning_rate=0.15 \
--adagrad_init_acc=0.1 \
--max_grad_norm=0.8 \
--mode="eval" \
--checkpoints_save_steps=5000 \
--max_steps=38000 \
--num_to_test=5 \
--max_num_to_eval=100 \
--vocab_path="../../Datasets/tfrecords_folder/tfrecords_folder/vocab" \
--data_dir="../../Datasets/tfrecords_folder/tfrecords_folder/val" \
--model_path="../pgn_model_dir/checkpoint/ckpt-37000" \
--checkpoint_dir="../pgn_model_dir/checkpoint" \
--test_save_dir="../pgn_model_dir/test_dir/"__
