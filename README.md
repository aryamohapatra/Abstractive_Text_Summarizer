# Abstractive Text Summarization

Teps to Run the code:

Get the data
  -   Preprocessed data from : https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail
  or 
  -   Preprocess the data using : https://github.com/abisee/cnn-dailymail
  
After getting the binary files convert into TFRecords using : https://github.com/steph1793/CNN-DailyMail-Bin-To-TFRecords

4 Models to Run:

1. Base Model : GRU + Bahdanau Attention
2. Model 1 : LSTM + Bahdanau Attention
3. Model 2 : LSTM + Luong's global attention
4. Model 3 : GRU + Luong's global attention


Open the respective folders and run the main.py with appropriate parameters
