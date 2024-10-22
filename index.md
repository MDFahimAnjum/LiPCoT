# LiPCoT: Linear Predictive Coding based Tokenizer for Self-Supervised Learning of Time Series Data via BERT

LiPCoT (Linear Predictive Coding based Tokenizer for time series) is a novel tokenizer that encodes time series data into a sequence of tokens, enabling self-supervised learning of time series using existing Language model architectures such as BERT. 

**Main Article:  [LiPCoT: Linear Predictive Coding based Tokenizer for Self-supervised Learning of Time Series Data via Language Models](https://arxiv.org/abs/2408.07292)**

- Unlike traditional time series tokenizers that rely heavily on CNN encoder for time series feature generation, LiPCoT employs stochastic modeling through linear predictive coding to create a latent space for time series providing a compact yet rich representation of the inherent stochastic nature of the data. 
- LiPCoT is computationally efficient and can effectively handle time series data with varying sampling rates and lengths, overcoming common limitations of existing time series tokenizers. 

## Citation
If you use this dataset or code in your research, please cite the following paper:

        @misc{anjum2024lipcot,
            title={LiPCoT: Linear Predictive Coding based Tokenizer for Self-supervised Learning of Time Series Data via Language Models},
            author={Md Fahim Anjum},
            year={2024},
            eprint={2408.07292},
            archivePrefix={arXiv},
            primaryClass={cs.LG}
        }


## Dataset 
We use EEG dataset of 28 PD and 28 control participants.
- Original dataset can be found at [link](http://predict.cs.unm.edu/downloads). The data are in .mat formats and you need Matlab to load them. (No need for this unless you are interested into the original EEG data)
- Raw CSV dataset used for this repo can be found at [link](https://www.dropbox.com/scl/fi/xinqn33vof0bnb9rlvmdh/raw.zip?rlkey=jb4dyumh7v82vbj36wsb53x13&dl=0). Download this for running all steps in this repo.

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

### 7. Classification task: CNN-based architectures
1. `cnn_classifier` notebook uses CNN model as described in [Oh et. al. (2018)](https://link.springer.com/article/10.1007/s00521-018-3689-5)
2. `deepnet_classifier` notebook uses Deep Convolutional Network as described in [Schirrmeister et. al. (2017)](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)
3. `shallownet_classifier` notebook uses Shallow Convolutional Network as described in [Schirrmeister et. al. (2017)](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)
4. `eegnet_classifier` notebook uses EEGNet as described in [here](https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py)