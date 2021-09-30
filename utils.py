#this is all original code
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from tqdm.notebook import tqdm  # tdqm is for the progress bar - you must pip install tdqm


def train(net,  trainloader, validloader, n_epochs, loss_function, optimizer, device):


    loss_history=[]
    valid_history=[]
    
    for epoch in tqdm(range(n_epochs)): # tdqm is for the progress bar - you must pip install tdqm
        loss_epoch=0

        for idx, X_batch,y_batch in trainloader:
            X_batch,y_batch  = X_batch.to(device), y_batch.to(device)

            for param in net.parameters():
                param.grad = None #remove the gradients for each loop
            output=net.forward(X_batch) # forward pass
            loss = loss_function(output, y_batch) #calculate loss
            loss.backward()#backward pass
            optimizer.step() 

            loss_epoch+=loss.detach().cpu().numpy()
        loss_history.append(loss_epoch/len(trainloader))#take the mean so we can compare with valid_loss

        with torch.no_grad(): # remove gradient calculation for validation loop
            val_loss_epoch=0
            for idx, X_val,y_val in validloader:
                X_val,y_val  = X_batch.to(device), y_batch.to(device)
                output = net.forward(X_val)
                val_loss = loss_function(output, y_val)
                val_loss_epoch+=val_loss.cpu().numpy()
            valid_history.append(val_loss_epoch/len(validloader))
            print('Epoch=',epoch, 'CrossEntropyLoss = ', loss_epoch/len(trainloader), 'Valid CrossEntropyLoss = ', val_loss_epoch/len(validloader))

    return loss_history, valid_history


def make_conf_matrix(net, testloader, device): 
    net.eval() # put net in to evaluation mode
    conf_matrix = np.zeros((7,7))
    for _, inputs, labels in tqdm(testloader):
        inputs = inputs.to(device)
        labels=labels.item()  
        y_pred = net(inputs).argmax().cpu().item()
        conf_matrix[y_pred, labels]+=1  #populate the confusion matrix
    net.train() # put net back into train mode
    return conf_matrix


def plot_conf_matrix(conf_matrix):
    ax = sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15}, fmt='.4g', linewidths=1, 
                     linecolor='black', xticklabels=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
                     yticklabels=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
                     cbar=False)#reduce the decimal digits displayed for greater visibility
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    ax.tick_params(left=False, bottom=False, top=False)
    plt.xlabel('Truth\n', size=20)
    plt.ylabel('Predicted\n', size=20)
    
    
def get_eval_scores(conf_matrix):
    correct_preds=conf_matrix.diagonal().sum()
    all_preds = conf_matrix.sum()
    accuracy = correct_preds / all_preds # formula for accuracy
    accuracy = accuracy.round(3)*100
    precision_array=np.zeros(7)
    recall_array=np.zeros(7)
    f1_score_array=np.zeros(7)
    for i in range(7):
        precision_array[i]=conf_matrix[i,i]/(conf_matrix[i].sum()) # formula for precision
        recall_array[i]=conf_matrix[i,i]/(conf_matrix[:,i].sum())# formula for recall
        f1_score_array[i]=2*((precision_array[i]*recall_array[i])/(precision_array[i]+recall_array[i])) # formula for F1 score
    precision_array = precision_array.round(3)
    recall_array = recall_array.round(3)
    f1_score_array = f1_score_array.round(3)
    eval_arr = np.vstack((f1_score_array,precision_array, recall_array))
    index_values = ['F1 Score', 'Precision', 'Recall']
    column_values = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    evaldf = pd.DataFrame(data = eval_arr, 
                  index = index_values, 
                  columns = column_values)
    evaldf.sort_values(by='F1 Score', axis=1, ascending=False, inplace=True)
    print ('Accuracy =',accuracy,'%')
    display(evaldf)