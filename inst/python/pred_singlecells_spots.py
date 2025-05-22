### import packages
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import pandas as pd
import operator
import random
import math
import logging
import dill
import torch.utils.data as data
from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

### main function
### pred
def pred_tls(dat_pred,pretrained_model,sen_len,pretrained_model_path,save_checkpoint_path):

    #BERT Encoder to encode text data for prototypical networks
    class BERTEncoder(nn.Module):
        """Encoder indices of sentences in BERT last hidden states."""
        def __init__(self, model_shortcut_name,max_length):
            super(BERTEncoder, self).__init__()
            self.max_length = max_length
            self.bert = AutoModel.from_pretrained(model_shortcut_name)
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_shortcut_name)
            
        def forward(self, tokens, mask):
            """BERT encoder forward.
            Args:
                tokens: torch.Tensor, [-1, max_length]
                mask: torch.Tensor, [-1, max_length]
                
            Returns:
                sentence_embedding: torch.Tensor, [-1, hidden_size]"""
        
            last_hidden_state = self.bert(tokens, attention_mask=mask)
            return last_hidden_state[0][:, 0, :]  # The last hidden-state of <CLS>

        def tokenize(self, text):
            """BERT tokenizer.
            
            Args:
                text: str
            
            Returns:
                ids: list, [max_length]
                mask: list, [max_length]"""
            ids = self.bert_tokenizer.encode(text, add_special_tokens=True,truncation=True,
                                            max_length=self.max_length)
            # attention mask: 1 for tokens that are NOT MASKED, 0 for MASKED tokens.
            mask = [1] * len(ids)
            # Padding
            while len(ids) < self.max_length:
                ids.append(0)
                mask.append(0)
            # truncation
            ids = ids[:self.max_length]
            mask = mask[:self.max_length]
            return ids, mask
        
    if pretrained_model=="TLSformer_BERT":
        class MLPModel(nn.Module):
            '''After BERT Encoder, added MLP layers'''
            def __init__(self):
                super(MLPModel, self).__init__()
                self.linear1 = nn.Linear(768, 360)
                self.dropout = nn.Dropout(p=0.2)
                self.relu = nn.ReLU()
                self.linear2 = nn.Linear(360, 256)
                # self.linear3 = nn.Linear(256, 10)
                
            def forward(self, x):
                #out = self.flatten(x)
                out = self.linear1(x)
                out = self.dropout(out)
                out = self.relu(out)
                out = self.linear2(out)
                # out = self.relu(out)
                # out = self.linear3(out)
                return out
    elif pretrained_model=="geneformer":
        class MLPModel(nn.Module):
            '''After BERT Encoder, added MLP layers'''
            def __init__(self):
                super(MLPModel, self).__init__()
                self.linear1 = nn.Linear(512, 360)
                self.dropout = nn.Dropout(p=0.2)
                self.relu = nn.ReLU()
                self.linear2 = nn.Linear(360, 256)
                # self.linear3 = nn.Linear(256, 10)
                
            def forward(self, x):
                #out = self.flatten(x)
                out = self.linear1(x)
                out = self.dropout(out)
                out = self.relu(out)
                out = self.linear2(out)
                # out = self.relu(out)
                # out = self.linear3(out)
                return out

    # Calculates Euclidean distance between class protoypes and query samples
    class L2Distance(nn.Module):
        def __init__(self):
            super(L2Distance, self).__init__()

        def forward(self, support, query):
            """
            Args:
                support: torch.Tensor, [B, totalQ, N, D]
                query: torch.Tensor, [B, totalQ, N, D]
                
            Returns:
                relation_score: torch.Tensor, [B, totalQ, N]"""
            l2_distance = torch.pow(support - query, 2).sum(-1, keepdim=False)  # [B, totalQ, N]
            return F.softmax(-l2_distance, dim=-1),l2_distance
        
    #Model for prototypical network
    class PrototypeNetwork(nn.Module):
        def __init__(self, encoder, relation_module, hidden_size, max_length,
                    current_device=torch.device("cpu"),mlp = MLPModel()):
            super(PrototypeNetwork, self).__init__()
            self.encoder = encoder
            self.relation_module = relation_module
            self.hidden_size = hidden_size  # D
            self.max_length = max_length
            self.current_device = current_device
            self.mlp = mlp

        def loss(self, predict_proba, label):
            # CE loss
            N = predict_proba.size(-1)
            return F.cross_entropy(predict_proba.view(-1, N), label.view(-1))
        
        def mean_accuracy(self, predict_label, label):
            return torch.mean((predict_label.view(-1) == label.view(-1)).type(torch.FloatTensor))

        def forward(self, support, support_mask, query, query_mask):
            """Prototype Networks forward.
            Args:
                support: torch.Tensor, [-1, N, K, max_length]
                support_mask: torch.Tensor, [-1, N, K, max_length]
                query: torch.Tensor, [-1, totalQ, max_length]
                query_mask: torch.Tensor, [-1, totalQ, max_length]
                
            Returns:
                relation_score: torch.Tensor, [B, totalQ, N]
                predict_label: torch.Tensor, [B, totalQ]"""
            B, N, K = support.size()[:3]
            totalQ = query.size()[1]  # Number of query instances for each batch
            
            ## Encoder & MLP
            # 1.1 Encoder
            support = support.view(-1, self.max_length)  # [B * N * K, max_length]
            support_mask = support_mask.view(-1, self.max_length)
            query = query.view(-1, self.max_length)  # [B * totalQ, max_length]
            query_mask = query_mask.view(-1, self.max_length)

            support = self.encoder(support, support_mask)  # [B * N * K, D]
            query = self.encoder(query, query_mask)  # [B * totalQ, D]
            
            # 1.2 MLP
            support = self.mlp(support)
            query = self.mlp(query)
            
            support = support.view(-1, N, K, self.hidden_size)  # [B, N, K, D]
            query = query.view(-1, totalQ, self.hidden_size)  # [B, totalQ, D]
            
            # 2. Prototype
            support = support.mean(2, keepdim=False)  # [B, N, D]

            # 3. Relation
            support = support.unsqueeze(1).expand(-1, totalQ, -1, -1)  # [B, totalQ, N, D]
            query = query.unsqueeze(2).expand(-1, -1, N, -1)  # [B, totalQ, N, D]
            relation_score,relation_detail = self.relation_module(support, query)  # [B, totalQ, N]

            predict_label = relation_score.argmax(dim=-1, keepdims=False)  # [B, totalQ]

            return relation_score, predict_label,relation_detail
        
    data_name = "tls"
    max_length = sen_len+2

    if torch.cuda.is_available():
        current_cuda = True
        current_device = torch.device("cuda")
    else:
        current_cuda = False
        current_device = torch.device("cpu")

    encoder = BERTEncoder(pretrained_model_path,max_length)
    tokenizer = encoder.tokenize
    relation_module = L2Distance()

    model = PrototypeNetwork(
            encoder,
            relation_module,
            256,
            max_length,
            current_device=current_device
    )

    
    # load trained model
    model.load_state_dict(torch.load(save_checkpoint_path+"/trained_model.pt"))
        
    with open(save_checkpoint_path+"/train_support_region_nonregion_vec.pkl","rb") as f:
        support_region_nonregion = dill.load(f)
    
    # load model into gpu or cpu
    if current_cuda:
        model.cuda()
    else:
        model.cpu()
    
    # calculate reference vector
    print("Predict single cells or spots")
        
    if current_cuda:
        support_region_nonregion =  torch.tensor(np.array([[[support_region_nonregion.cpu()]]])).to("cuda")
    else:
        support_region_nonregion =  torch.tensor(np.array([[[support_region_nonregion.cpu()]]])).to("cpu")
        
    support = support_region_nonregion[0]
    
    # pred
    
    pred_dat = dat_pred
    if current_cuda:
        predict_label = torch.tensor(np.zeros(shape = (len(pred_dat)))).to("cuda")
        region_distance = torch.tensor(np.zeros(shape = (len(pred_dat),2))).to("cuda")
        query_train = torch.tensor(np.zeros(shape = (len(pred_dat),256))).to("cuda")
    else:
        predict_label = torch.tensor(np.zeros(shape = (len(pred_dat)))).to("cpu")
        region_distance = torch.tensor(np.zeros(shape = (len(pred_dat),2))).to("cpu")
        query_train = torch.tensor(np.zeros(shape = (len(pred_dat),256))).to("cpu")
    
    with torch.no_grad():
        for i in tqdm(range(0,len(pred_dat))):
            #print(i)
            query, query_mask = tokenizer(pred_dat["sentence"][i])
            query = np.array(query)
            query_mask = np.array(query_mask)
            query = torch.tensor(query, dtype=torch.long)
            query_mask = torch.tensor(query_mask, dtype=torch.long)
            if current_cuda:
                query = query.cuda()
                query_mask = query_mask.cuda()
            query = query.view(-1, max_length)  # [B * totalQ, max_length]
            query_mask = query_mask.view(-1, max_length)
            query = model.encoder(query, query_mask)  # [B * totalQ, D]
            query = model.mlp(query)
            query_train[i] = query
            query = query.view(-1, 1, 256) # [B, totalQ, D]
            query = query.unsqueeze(2).expand(-1, -1, 2, -1)  # [B, totalQ, N, D]
            relation_score,relation_detail_tmp = model.relation_module(support, query)  # [B, totalQ, N]
            predict_label_tmp = F.softmax(-relation_detail_tmp, dim=-1).argmax(dim=-1, keepdims=False)  # [B, totalQ]
            predict_label[i] = predict_label_tmp[0][0]
            region_distance[i] = relation_detail_tmp[0][0]
            
    region_distance = pd.DataFrame(np.array(region_distance.cpu().numpy()))
    region_distance.index = pred_dat.index
    region_distance.columns = ["Non-TLS Distance","TLS Distance"]
    pred_dat = pd.concat([pred_dat,region_distance],axis=1)
    pred_dat["pred_label"] = predict_label.cpu().tolist()
    pred_dat["relative_distance"] = pred_dat["TLS Distance"] - pred_dat["Non-TLS Distance"]
    
    return pred_dat

