import os

current_directory = os.getcwd() #parameters

processed_datapath = os.path.join(current_directory,"data/processed") # Define the path of processed data
tokenized_data_savepath= os.path.join(current_directory,"data/tokenized")
bert_datapath = os.path.join(current_directory,"data/forBert") # Define the path of processed data
lipcot_model_savepath= os.path.join(current_directory,"models/lipcot_model")
tokenizer_path = os.path.join(current_directory,"models/modifiedBertTokenizer") # Define the path where you want to save the tokenizer
bert_modelpath=os.path.join(current_directory,"models/bert_model")
bert_finetune_modelpath=os.path.join(current_directory,"models/bert_finetune_model")
bert_finetune_no_pretrain_modelpath=os.path.join(current_directory,"models/bert_finetune_model_without_pretrain")
cnn_modelpath=os.path.join(current_directory,"models/cnn_model")
results_path=os.path.join(current_directory,"results")
