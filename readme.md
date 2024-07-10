# LiPCoT: Linear Predictive Coding based Tokenizer for Self-Supervised Learning of Time Series Data via BERT

## Dataset 
We use EEG dataset of 28 PD and 28 control participants.

## How to run
- If you want to run all steps:
    - Download raw CSV dataset and place them in the `data/raw` folder
    - Run Step 1-7
- If you want to run only the BERT models, run Step 4-6. No need to download raw data as the processed dataset is included in this repo.

## Steps
### 1. Data Preparation
First, the data must be processed. `data_processing` notebook loads raw data and prepares training,validation and test dataset.

### 2. Tokenization via LiPCoT
`data_tokenizer` notebook tokenizes the data using LiPCoT model

### 3. Prepare tokenized dataset for BERT
`data_prepare` notebook prepares datasets for BERT models. If you are downloading from GitHub, up to this step is done for you.

### 4. Self-supervised learning via BERT
`pretrain_bert` notebook conducts pretraining of BERT model.

__If you are running code with data from GitHub, start with this step.__

### 5. Classification task: with pretrained BERT
`finetune_bert` notebook conducts fine-tune of BERT model for binary classification

### 6. Classification task: without pretraining
`finetune_bert_without_pretrain` notebook uses a randomly initialized BERT model and fine tunes it for classification

### 7. Classification task: CNN
`cnn_classifier` notebook uses CNN model for the classification task