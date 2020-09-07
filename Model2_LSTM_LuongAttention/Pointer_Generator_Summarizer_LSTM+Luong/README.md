This is the extension work on the Base Model. Here we have added LSTM as the RNN unit along with that implemented Global attention.


Python files Where Changes have been made:

1. layers.py 
  * Class Encoder
    - __init__
    - call
    - initialize_hidden_state
  * Class Decoder
    - __init__
    - call
  * LuongAttention
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
    
    
To Run this code please run the below command with parameters:


python main.py
--max_enc_len=400
--max_dec_len=100
--max_dec_steps=100
--min_dec_steps=30
--batch_size=4
--beam_size=4
--vocab_size=50000
--embed_size=128
--enc_units=256
--dec_units=256
--attn_units=512
--learning_rate=0.1
--adagrad_init_acc=0.1
--max_grad_norm=0.8
--mode="eval"
--checkpoints_save_steps=5000
--max_steps=40000
--num_to_test=5
--max_num_to_eval=100
--vocab_path="vocab path created"
--data_dir="Path to the training data"
--model_path="path to the path to save the model"
--checkpoint_dir="path to the path to save the model checkpoint details"
--test_save_dir="Path where test file gets saved"
