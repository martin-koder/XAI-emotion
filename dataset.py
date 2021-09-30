#this is all original code
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt



class FER2013Dataset(Dataset):
    """Face Expression Recognition Dataset"""
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.classes = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral') # Define the name of classes / expression
        
        with open(self.file_path) as f: 
            self.data = f.readlines() #read data
            
        self.total_images = len(self.data) - 1 #reduce 1 for row of column

    def __len__(self):  
        return self.total_images
    
    def __getitem__(self, idx):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize((129.47,), (65.02,))]) # resize, normalise, convert to tensor
        if torch.is_tensor(idx):
            idx = idx.tolist()
        with open(self.file_path) as f:
            emotion, usage, img = self.data[idx + 1].split(",") #plus 1 to skip first row (column name)
            
        emotion = np.array(int(emotion)) # just make sure it is int not str
        img = img.split(" ") # because the pixels are seperated by space
        img = np.array(img, 'float') # just make sure it is int not str 
        img = img.reshape(48,48,1) # change shape from 2304 
        img = transform(img)

        X = img.float().view(1,224,224)
        y = torch.from_numpy(emotion).long() 
        
        return idx, X, y
    
def display_rnd_img(dataset):
    idx, example, emotion = dataset[np.random.randint(len(dataset))]
    example=example.detach().numpy().reshape(224,224)
    plt.imshow(example)
    plt.tick_params(
    axis='both',          
    which='both',      
    bottom=False,      
    left=False,         
    labelbottom=False,
    labelleft=False,)
    plt.gray()
    plt.title(str(list(dataset.classes)[emotion.detach().numpy().item()])+'\n', size=15)