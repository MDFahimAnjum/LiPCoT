import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score
import torch
from tqdm import tqdm
import random

def prepare_cnn_data(lkdataset):
    signal_data=[]
    labels=[]
    for lkdata in lkdataset:
        signal=lkdata.data # (signal x chan) matrix
        signal=np.transpose(signal) #(Chan x Signal Length) matrix
        signal_tensor=torch.from_numpy(signal).float() # tensor
        signal_data.append(signal_tensor)
        if lkdata.label==0:
            newlabel=torch.tensor([0,1]).float()
        else:
            newlabel=torch.tensor([1,0]).float()
        labels.append(newlabel)
    return signal_data,labels


def evaluate_binary_accuracy(model,eval_loader):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in eval_loader:
            in_data=batch[0]
            in_label=batch[1]

            # Forward pass
            logits = model(in_data) 
            # Convert logits to probabilities using sigmoid activation
            probs = torch.sigmoid(logits)
            probs_normalized = probs / probs.sum(dim=1, keepdim=True)

            # Convert probabilities to binary predictions (0 or 1)
            preds = (probs_normalized > 0.5).int()

            all_scores.extend(probs_normalized.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(in_label.tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy*100,    all_preds,  all_labels  ,all_scores


def model_trainer(model,train_loader,val_loader,training_arg):
    
    modelfullpath=training_arg['modelfullpath']
    criterion=training_arg['criterion']
    optimizer=training_arg['optimizer']
    num_epochs=training_arg['epochs']
    patience=training_arg['patience']
    random_seed=training_arg['seed']
    
    # Set the random seed for reproducibility
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Training loop
    num_batches = len(train_loader)
    best_val_accuracy=-1
    average_epoch_loss_all=[]
    acc_train_all=[]
    acc_val_all=[]

    tr_est_temp=[]
    tr_labels_temp=[]
    tr_scores_temp=[]

    v_est_temp=[]
    v_labels_temp=[]
    v_scores_temp=[]

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        # Use tqdm to create a progress bar for the training loop
        for batch_data, batch_labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            optimizer.zero_grad()
            output = model(batch_data)
            probs = torch.sigmoid(output)
            probs_normalized = probs / probs.sum(dim=1, keepdim=True)
            loss = criterion(probs_normalized, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Calculate and print average epoch loss
        average_epoch_loss = total_loss / num_batches
        average_epoch_loss_all.append(average_epoch_loss)
        #calculate accuracy perf
        accuracy, tr_est,tr_labels,tr_scores= evaluate_binary_accuracy(model, train_loader)
        v_accuracy, val_est,val_labels, val_scores = evaluate_binary_accuracy(model, val_loader)
        # Switch back to training mode for the next epoch
        model.train()
        # collect eval results
        acc_train_all.append(accuracy)
        acc_val_all.append(v_accuracy)

        tr_est_temp.extend([tr_est])
        tr_labels_temp.extend([tr_labels])
        tr_scores_temp.extend([tr_scores])

        v_est_temp.extend([val_est])
        v_labels_temp.extend([val_labels])
        v_scores_temp.extend([val_scores])

        #print(f'Epoch {epoch + 1} - Average Loss: {average_epoch_loss:.4f} - Train acc: {accuracy:.2f} - Validation acc: {v_accuracy:.2f}')

        # Check if validation accuracy has improved
        if v_accuracy > best_val_accuracy:
            best_val_accuracy = v_accuracy
            torch.save(model.state_dict(),modelfullpath) # Save the model state
            print(f'model state saved with validation accuracy: {best_val_accuracy:.4f}')
            counter = 0  # Reset counter if accuracy improved
        else:
            counter += 1  # Increment counter if accuracy did not improve

        # Check if patience limit reached
        if counter >= patience:
            print(f'Early stopping at epoch {epoch + 1} as validation accuracy did not improve for {patience} epochs.')
            break
    train_results={'labels': tr_labels_temp, 'scores': tr_scores_temp, 'pred': tr_est_temp}
    val_results={'labels': v_labels_temp, 'scores': v_scores_temp, 'pred': v_est_temp}

    results={'epoch_loss':average_epoch_loss_all, 
             'training_acc': acc_train_all, 
             'validation_acc':acc_val_all,
             'train_results': train_results,
             'validation_results':val_results }
    return results