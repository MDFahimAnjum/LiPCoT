
#%%
import torch
from torch.utils.data import Dataset
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score

#%% 

class BertDataset(Dataset):
    def __init__(self, texts, labels,tokenizer,max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length=max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, 
                                   add_special_tokens=True,return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

# Modify pre-trained BERT model for sequence classification
class BertForSequenceClassification(nn.Module):
    def __init__(self, pretrained_model, num_classes,device):
        super(BertForSequenceClassification, self).__init__()
        pretrained_model.to(device)
        self.bert = pretrained_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs['hidden_states']  # Get the hidden states
        cls_token_output = hidden_states[-1][:, 0, :]  # [][batch x words x hidden vector] Take the last layer's [CLS] token output
        pooled_output = self.dropout(cls_token_output)
        logits = self.classifier(pooled_output)
        return logits

# Prepare Dataset:
def prepare_bert_data(lkdataset):
    text=[]
    labels=[]
    for lkdata in lkdataset:
        text.append(lkdata.tokenized)
        if lkdata.label==0:
            newlabel=np.array([0,1])
        else:
            newlabel=np.array([1,0])
        labels.append(newlabel)
    return text,labels

def get_mask_error(testdata,fill_mask,tokenizer):
    mask_str=tokenizer.mask_token
    masked_tokens=[]
    predicted_tokens=[]
    error_count = 0
    random.seed(42)
    for sentence in testdata:
        for i in range(1):
            # Split the string into words
            words = sentence.split()
            # Choose a random index
            random_index = random.randint(0, len(words) - 1)
            masked_word=words[random_index]
            words[random_index]=mask_str
            output_string = ' '.join(words)
            fill_mask_results=fill_mask(output_string) 
            #predicted_word=fill_mask_results[0]['token_str']
            #if masked_word != predicted_word:
            #    error_count += 1
            all_predicted_word = [result['token_str'] for result in fill_mask_results]
            predicted_word=all_predicted_word
            if masked_word in all_predicted_word:
                error_count += 0
            else:
                error_count += 1
            masked_tokens=np.append(masked_tokens,(tokenizer.encode(masked_word))[1])
            predicted_tokens=np.append(predicted_tokens,(tokenizer.encode(predicted_word))[1])
        #print(masked_word+" vs "+predicted_word)

    error_rate=100*error_count/len(testdata)
    print(error_rate)
    plt.figure
    plt.hist(masked_tokens, bins=100, color='blue', alpha=0.5, label='true')
    plt.hist(predicted_tokens, bins=100, color='red', alpha=0.5, label='predicted')
    plt.title('Histogram of MLP')
    plt.xlabel('Token Value')
    plt.ylabel('Frequency')
    #plt.yscale('log')  # Set y-axis to log scale
    plt.grid(axis='y')
    plt.legend()
    plt.show()
    return error_rate, masked_tokens, predicted_tokens

def get_txt_from_dataset(tokenizer,dataset):
    txt_data=[]
    for i in range(len(dataset)):
        curr_data=dataset[i]['input_ids']
        curr_data_attention_mask=dataset[i]['attention_mask'].numpy().astype(bool)
        curr_data=curr_data[curr_data_attention_mask] # remove paddings
        curr_data=curr_data[1:-1]
        curr_str=tokenizer.decode(curr_data)
        txt_data.append(curr_str)
    return txt_data

def get_trainer_history(t_history,name):
    t_val =(t_history[name]).values
    #t_step=(t_history['step']).values
    t_step=(t_history['epoch']).values
    #t_step=np.linspace(0,len(t_val)-1,len(t_val))
    valid_i=~np.isnan(t_val)
    t_val=t_val[valid_i]
    t_step=t_step[valid_i]
    return t_val,t_step


# Assuming your model is already trained and you have a validation dataset and dataloader
# Define a function to evaluate binary accuracy
def evaluate_binary_accuracy(model, dataloader,device):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)  # Move input tensors to the device (e.g., GPU)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            logits = model(input_ids, attention_mask) 
            # Convert logits to probabilities using sigmoid activation
            probs = torch.sigmoid(logits)
            probs_normalized = probs / probs.sum(dim=1, keepdim=True)

            # Convert probabilities to binary predictions (0 or 1)
            preds = (probs_normalized > 0.5).int()

            all_scores.extend(probs_normalized.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    # Calculate binary accuracy using sklearn's accuracy_score
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy*100,    all_preds,  all_labels  ,all_scores

def perf_metrics(y_true, y_pred,y_score):
    # Extract the first elements using list comprehension
    y_true = [array[0] for array in y_true]
    y_pred = [array[0] for array in y_pred]
    y_score = [array[0] for array in y_score]
    
    cm = confusion_matrix(y_true, y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate sensitivity (recall)
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])

    # Calculate specificity
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

    # Calculate precision
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    
    # Calculate F1 score
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

    # Calculate AUC
    #auc = roc_auc_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)

    # Print the metrics
    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:", accuracy*100)
    print("Sensitivity (Recall):", sensitivity*100)
    print("Specificity:", specificity*100)
    print("F1 Score:", f1)
    print("AUC Score:", auc)
    return cm, sensitivity,specificity,f1,auc
