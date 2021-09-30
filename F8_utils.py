#this is all orginal code apart from the amended collate_batch function, which is adapted from https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
import torch
import numpy as np
from torchtext.data.utils import get_tokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_accuracy(conf_matrix_val):
    correct_preds=conf_matrix_val.diagonal().sum()
    all_preds = conf_matrix_val.sum()
    accuracy = correct_preds / all_preds # formula for accuracy
    accuracy = accuracy*100
    return accuracy


def make_conf_matrix(net, valloader, device): # quicker!
    # put net in to evaluation mode
    conf_matrix = np.zeros((13,13))
    with torch.no_grad():
        for labels, inputs, offsets in valloader:
            inputs = inputs.to(device)
            labels=labels.cpu().numpy()  #detach?
            y_pred = net(inputs, offsets).argmax(1).cpu().numpy() # take argmax on first (col) dimen
            for i, j in zip(y_pred, labels):
                conf_matrix[i, j]+=1  #populate the confusion matrix
    return conf_matrix

def label_pipeline(label):
    return label


tokenizer = get_tokenizer('basic_english')

def text_pipeline(X): 
        return [vocab[token] for token in tokenizer(X)]   
    
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (idx, text, label) in batch:
         label_list.append(label_pipeline(label))
         processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)    

def load_checkpoint(file_name, net, device):
    ckpt = torch.load(file_name, map_location=device)
    net.load_state_dict(ckpt)
    print("Model's pretrained weights loaded!")
    
def plot_conf_matrix(conf_matrix):
    xticklabels=['empty',
 'sadness',
 'enthusiasm',
 'neutral',
 'worry',
 'surprise',
 'love',
 'fun',
 'hate',
 'happiness',
 'boredom',
 'relief',
 'anger']
    yticklabels=['empty',
 'sadness',
 'enthusiasm',
 'neutral',
 'worry',
 'surprise',
 'love',
 'fun',
 'hate',
 'happiness',
 'boredom',
 'relief',
 'anger']
    plt.figure(figsize=(9,9))
    ax = sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15}, fmt='.4g', linewidths=1, 
                     linecolor='black', xticklabels=['empty',
 'sadness',
 'enthusiasm',
 'neutral',
 'worry',
 'surprise',
 'love',
 'fun',
 'hate',
 'happiness',
 'boredom',
 'relief',
 'anger'],
                     yticklabels=['empty',
 'sadness',
 'enthusiasm',
 'neutral',
 'worry',
 'surprise',
 'love',
 'fun',
 'hate',
 'happiness',
 'boredom',
 'relief',
 'anger'],
                     cbar=False)#reduce the decimal digits displayed for greater visibility
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    ax.set_xticklabels(xticklabels, rotation = 45, fontsize = 16)
    ax.set_yticklabels(yticklabels, fontsize = 16)
    ax.tick_params(left=False, bottom=False, top=False)
    plt.xlabel('Truth', size=20)
    plt.ylabel('Predicted', size=20)
#     plt.title('Testset Confusion Matrix', loc='left', fontsize = 14)
    
def get_eval_scores(conf_matrix):
    correct_preds=conf_matrix.diagonal().sum()
    all_preds = conf_matrix.sum()
    accuracy = correct_preds / all_preds # formula for accuracy
    accuracy = accuracy.round(3)*100
    precision_array=np.zeros(13)
    recall_array=np.zeros(13)
    f1_score_array=np.zeros(13)
    for i in range(13):
        precision_array[i]=conf_matrix[i,i]/(conf_matrix[i].sum()) # formula for precision
        recall_array[i]=conf_matrix[i,i]/(conf_matrix[:,i].sum())# formula for recall
        f1_score_array[i]=2*((precision_array[i]*recall_array[i])/(precision_array[i]+recall_array[i])) # formula for F1 score
    precision_array = precision_array.round(3)
    recall_array = recall_array.round(3)
    f1_score_array = f1_score_array.round(3)
    eval_arr = np.vstack((f1_score_array,precision_array, recall_array))
    index_values = ['F1 Score', 'Precision', 'Recall']
    column_values = ['empty',
 'sadness',
 'enthusiasm',
 'neutral',
 'worry',
 'surprise',
 'love',
 'fun',
 'hate',
 'happiness',
 'boredom',
 'relief',
 'anger']
    evaldf = pd.DataFrame(data = eval_arr, 
                  index = index_values, 
                  columns = column_values)
    evaldf.sort_values(by='F1 Score', axis=1, ascending=False, inplace=True)
    print ('Accuracy =',accuracy,'%')
    display(evaldf)