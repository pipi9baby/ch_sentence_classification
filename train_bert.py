import os
import math
import torch
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm_notebook as tqdm
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange
import json
from torch.nn import CrossEntropyLoss
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data.sampler import SubsetRandomSampler

import visualization as vl
from transformers import BertTokenizer, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(1)

model_version = 'bert-base-chinese'

def micro_f1_score(pred, label):
    
    TP = torch.mul(pred, label)[0]
    FP = (torch.mul(pred, (label-1)) != 0)[0]
    FN = (torch.mul(pred-1, label) != 0)[0]
    
    precision = TP.sum() / (TP.sum() + FP.sum())
    recall = TP.sum() / (TP.sum() + FN.sum())
    
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1, TP, FP, FN

class EmergencyDataset(Dataset):
    """Emergency Dataset"""
    
    def __init__(self, csv_file):
        self.categories = ["家中", "遊憩", "交通", "工作"]
        self.take = len(self.categories)
        self.dataframe = pd.read_csv(csv_file)
        print(self.dataframe.keys())
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        text = self.dataframe['主訴'][idx]

        target = []
        for l in self.categories:
            target.append(self.dataframe[l][idx])
        
        return text, torch.Tensor(target)

# activate function
class gelu(nn.Module):
    
    def __init__(self):
        super(gelu, self).__init__()

    def forward(self, x):
        cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        return x * cdf


class MutiLabelModel(nn.Module):
    
    def __init__(self, encoder, emb_size=1024, out_size=5, ce_size=23, hidden=256): # hidden=256
        super(MutiLabelModel, self).__init__()
        
        self.encoder = encoder
        self.fn_size = emb_size
        
        self.out_fn = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.fn_size, self.fn_size//2),
            gelu(),
            nn.Dropout(0.2),
            nn.Linear(self.fn_size//2, out_size),
        )
        
    def forward(self, inp, seg, cat_emb=None, cls_loc=0): # , inp_title, seg_inp_title, cls_loc=0):

        embs = self.encoder(inp, seg)[0] # [batch, seq, hidden]
        outputs = embs[:, cls_loc, :]
        outputs = self.out_fn(outputs)

        return outputs
    
    
if __name__ == "__main__":
    filepath = '/home/dl-ismp-mh/han/Trauma_Ontology/all_data.tsv'
    TMPMDPATH = '/home/dl-ismp-mh/han/Trauma_Ontology/mse_ck.pt'
    figpath = '4labelfig'
    predPath = "predict.txt"
    ansPath = "ans.txt"
    lr = 3e-6
    batch_size = 8
    epochs = 200
    
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42
    
    ## Prepare dataset    
    dataset = EmergencyDataset(filepath)

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    # save dataloader
    torch.save(train_loader, 'dataloader_train.pth')
    torch.save(train_loader, 'dataloader_eval.pth')
    
    take = dataset.take
    board = vl.SummaryWriter(log_dir = figpath)
    
    # load pre-trained bert
    print("\nloading bert...")
    tokenizer = BertTokenizer.from_pretrained(model_version)
    encoder = BertModel.from_pretrained(model_version, output_attentions=True)   
    model = MutiLabelModel(encoder, 768, take).to(device)
    
    # Define loss
    criterion = nn.MSELoss()

    # Define opt
    num_total_steps = np.ceil(dataset.__len__() / batch_size)*epochs
    num_warmup_steps = int(num_total_steps * 0.5)

    optim = AdamW(model.parameters(), lr=lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps, num_total_steps)

    loss = 0    
    stepoch = 0
    
    # Read previous save weight
    if os.path.isfile(TMPMDPATH):
        print("load privious training weight")
        state = torch.load(TMPMDPATH)
        model.load_state_dict(state['state'])
        stepoch = state['epoch']
  
    model.train()   
    for epoch in range(stepoch, epochs):
        print('Epoch: {}'.format(epoch))
        print('step\tloss')

        for i, (text, target) in enumerate(train_loader):
            
            inputs = tokenizer.batch_encode_plus(text, return_tensors='pt', add_special_tokens=True, pad_to_max_length=True)
            token_type_ids = inputs['token_type_ids'].to(device)
            input_ids = inputs['input_ids'].to(device)
            out = model(input_ids, token_type_ids)
            
            target = target.to(device)
            l = criterion(out, target)            
            
            board.add_scalar("Loss", l, epoch)
            
            optim.zero_grad()
            
            l.backward()           
            optim.step()
            scheduler.step()
            
            if i%100 == 0:
                print(str(i)+'\t'+str(l.item())[:7])
            
                
                #save temporary model
                state = {
                    'epoch': epoch,
                    'state': model.state_dict(),
                    'encoder':encoder.state_dict(),
                    'tokenizer':tokenizer.state_dict()
                }
                torch.save(state, TMPMDPATH)
               
    
    model.eval()
    pred = []
    ans = []
    total, correct = 0, 0
    for i, (text, target) in enumerate(train_loader):
            
        inputs = tokenizer.batch_encode_plus(text, return_tensors='pt', add_special_tokens=True, pad_to_max_length=True)
        token_type_ids = inputs['token_type_ids'].to(device)
        input_ids = inputs['input_ids'].to(device)
        out = model(input_ids, token_type_ids)
            
        target = target.to(device)
        out = torch.sigmoid(out)
        
        pred += out.tolist()
        ans += target.tolist()
        
        t = torch.Tensor([0.65]).to(device)  # threshold
        thout = (out > t).float() * 1
                
        total += target.size(0)*target.size(1)
        correct += (thout == target).sum()
        
        if i%100 == 0:
            accuracy = 100 * correct / total
            print(str(accuracy.item())+"%")
            
    # output result
    fwA = open(ansPath,"w")
    fwP = open(predPath,"w")
    fwA.write(",".join(dataset.categories)+"\n")
    fwP.write(",".join(dataset.categories)+"\n")
    for i in range(len(ans)):
        fwA.write(str(ans[i])[1:-1]+"\n")
        fwP.write(str(pred[i])[1:-1]+"\n")
    
    fwA.close()
    fwP.close()