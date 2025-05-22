### import packages
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import operator
import random
from tqdm import tqdm
import math
import logging
import dill
import torch.utils.data as data
from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

### main function
### train
def train_prototype_hypersphere(dat_train,pretrained_model,sen_len,pretrained_model_path,save_checkpoint_path,batch_size = 1,train_K = 8,train_Q = 8, train_episodes = 300,val_episodes = 30,val_steps = 10):
    # Load train and test dataset in the form of  N way K shot tasks. 
    class GeneralDataset(data.Dataset):
        """
        Returns:
            support: torch.Tensor, [N, K, max_length]
            support_mask: torch.Tensor, [N, K, max_length]
            query: torch.Tensor, [totalQ, max_length]
            query_mask: torch.Tensor, [totalQ, max_length]
            label: torch.Tensor, [totalQ]"""
        def __init__(self, train_df, tokenizer, N, K, Q):
            # file_path = os.path.join(path, name)
            # if not os.path.exists(file_path):
            #     raise Exception("File {} does not exist.".format(file_path))
            ### Attention when the class label > 2, this part must be modified
            self.train_df = train_df
            self.train_df_0 = self.train_df.loc[self.train_df.region==0]
            self.train_df_1 = self.train_df.loc[self.train_df.region==1]
            self.train_X_0 = self.train_df_0['sentence']
            self.train_X_1 = self.train_df_1['sentence']
            self.train_Y_0 = self.train_df_0['region']
            self.train_Y_1 = self.train_df_1['region']
            self.train_Y = self.train_df['region']
            self.classes = list(set(self.train_Y))
            self.nb_classes = len(self.classes)
            self.N, self.K, self.Q = N, K, Q
            self.tokenizer = tokenizer

        def __getitem__(self, index):
            support, support_mask = [], []
            query, query_mask = [], []
            query_label = []
            target_classes = self.classes#random.sample(self.classes, self.N)  # Sample N class name
            #print(target_classes)
            for class_idx, class_name in enumerate(target_classes):
                # Add [] for each class
                support.append([])
                support_mask.append([])
                if class_name == 1:
                    if (sum(self.train_Y_1)) < self.K + self.Q :
                        self.train_X_1 = self.train_df_1['sentence']
                        self.train_Y_1 = self.train_df_1['region']
                    dataset = self.train_X_1
                    samples_rows = random.sample(dataset.index.tolist(), self.K + self.Q)
                    dataset = dataset.loc[samples_rows]
                    samples = dataset.values.tolist()
                    self.train_X_1 = self.train_X_1.drop(samples_rows)
                    self.train_Y_1 = self.train_Y_1.drop(samples_rows)
                else:
                    if (len(self.train_Y_0)) < self.K + self.Q :
                        self.train_X_0 = self.train_df_0['sentence']
                        self.train_Y_0 = self.train_df_0['region']
                    dataset = self.train_X_0
                    samples_rows = random.sample(dataset.index.tolist(), self.K + self.Q)
                    dataset = dataset.loc[samples_rows]
                    samples = dataset.values.tolist()
                    self.train_X_0 = self.train_X_0.drop(samples_rows)
                    self.train_Y_0 = self.train_Y_0.drop(samples_rows)
                    
            #   print(samples)
                for idx, sample in enumerate(samples):
                    # Tokenize. Senquences to indices.
                    indices, mask = self.tokenizer(sample)
                    if idx < self.K:
                        support[class_idx].append(indices)
                        support_mask[class_idx].append(mask)
                    else:
                        query.append(indices)
                        query_mask.append(mask)
                query_label += [class_idx] * self.Q
            #print(support)
            return (torch.tensor(support, dtype=torch.long),
                    torch.tensor(support_mask, dtype=torch.long),
                    torch.tensor(query, dtype=torch.long),
                    torch.tensor(query_mask, dtype=torch.long),
                    torch.tensor(query_label, dtype=torch.long))
        def __len__(self):
            return 100000


    def get_general_data_loader(train_df, tokenizer, N, K, Q, batch_size,
                                num_workers=4, sampler=False):
        dataset = GeneralDataset(train_df, tokenizer, N, K, Q)
        if sampler:
            sampler = data.distributed.DistributedSampler(dataset)
        else:
            sampler = None
        data_loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=1,
            sampler=sampler
        )
        return iter(data_loader)

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
        
        
    # Trains the model
    def train_prototype(data_name, train_data, val_data, tokenizer, model, B, N, K, Q, optimizer,
            train_episodes, val_episodes, val_steps, grad_steps, lr,
            warmup, weight_decay, save_checkpoint, cuda=False, fp16=False):

        data_name == "tls"
        train_data_loader = get_general_data_loader(train_data,tokenizer, N, K, Q, B)
        

        parameter_list = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]  # Do not use weight decay.
        optimizer = optimizer(
            [
                {
                    "params": [param for name, param in parameter_list
                    if not any(nd in name for nd in no_decay)],
                    "weight_decay": weight_decay
                },
                {
                    "params": [param for name, param in parameter_list
                    if any(nd in name for nd in no_decay)],
                    "weight_decay": 0.0
                }
            ], lr=lr
        )

        # A schedule with a learning rate that decreases linearly after linearly
        # increasing during a warmup period.
        # https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(warmup * train_episodes),
            num_training_steps=train_episodes
        )
        

        if cuda:
            model.cuda()
        if fp16:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        model.train()  # Set model to train mode.

        # Add model graph to tensorboard.
        dummy_support, dummy_support_mask, dummy_query, dummy_query_mask, _ = next(train_data_loader)
        if cuda:
            dummy_support = dummy_support.cuda()
            dummy_support_mask = dummy_support_mask.cuda()
            dummy_query = dummy_query.cuda()
            dummy_query_mask = dummy_query_mask.cuda()


        total_samples = 0  # Count 'total' samples
        total_loss = 0.0
        total_acc_mean = 0.0
        best_val_acc = 0.0
        run_num = 0

        for episode in tqdm(range(1, train_episodes + 1)):
            support, support_mask, query, query_mask, label = next(train_data_loader)
            if cuda:
                support = support.cuda()
                support_mask = support_mask.cuda()
                query = query.cuda()
                query_mask = query_mask.cuda()
                label = label.cuda()

            relation_score, predict_label,_ = model(support, support_mask, query, query_mask)
            loss = model.loss(relation_score, label) / grad_steps
            acc_mean = model.mean_accuracy(predict_label, label)

            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            if episode % grad_steps == 0:
                # Update params
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            run_num = run_num + 1
            total_loss += loss.item()
            total_acc_mean += acc_mean.item()
            total_samples += 1
            #print("Total loss: " + str(total_loss/run_num))    
            if val_data is not None and episode % val_steps == 0:
                eval_loss, eval_acc = eval(data_name, val_data, tokenizer, model,
                                            B, N, K, Q, val_episodes, cuda=cuda)
                # print("Eval Loss: " + str(eval_loss))
                # print("Eval acc: " + str(eval_acc))
                model.train()  # Reset model to train mode.
                if eval_acc > best_val_acc:
                    # Save model
                    torch.save(model.state_dict(), save_checkpoint)
                    best_val_acc = eval_acc
                total_samples = 0
                total_loss = 0.0
                total_acc_mean = 0.0
        print(best_val_acc)    
        return best_val_acc

    #Evaluates the model
    def eval(data_name, val_data, tokenizer, model, B, N, K, Q, val_episodes,
            load_checkpoint=None, is_test=False, cuda=False):
            
        if cuda:
            model.cuda()

        model.eval()  # Set model to eval mode.

        if data_name == "tls":
            data_loader = get_general_data_loader(val_data, tokenizer, N, K, Q, B)
        
        out_mark = "Test" if is_test else "Val"

        total_loss = 0.0
        total_acc_mean = 0.0
        with torch.no_grad():
            for episode in range(1, val_episodes + 1):
                support, support_mask, query, query_mask, label = next(data_loader)
                if cuda:
                    support = support.cuda()
                    support_mask = support_mask.cuda()
                    query = query.cuda()
                    query_mask = query_mask.cuda()
                    label = label.cuda()

                relation_score, predict_label,_ = model(support, support_mask, query, query_mask)
                loss = model.loss(relation_score, label)
                acc_mean = model.mean_accuracy(predict_label, label)
                total_loss += loss.item()
                total_acc_mean += acc_mean.item()

        loss_mean, acc_mean = total_loss / val_episodes, 100 * total_acc_mean / val_episodes
        return loss_mean, acc_mean

    dat_train = dat_train

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

    optimizer = torch.optim.Adam
    val_data = dat_train  
    train_data = dat_train
    N=2
    batch_size = batch_size
    K=train_K
    Q=train_Q
    train_episodes = train_episodes
    val_episodes = val_episodes
    val_steps = val_steps
    grad_steps = 1
    lr = 1e-5
    warmup = 0.06
    weight_decay = 0.01
    save_checkpoint = save_checkpoint_path+"/trained_model.pt"
    
    print("Train model to divide prototype")
    best_val_acc = train_prototype(data_name, dat_train, val_data, tokenizer, model, batch_size, N, K, Q, optimizer, train_episodes, val_episodes,val_steps, grad_steps, lr, warmup, weight_decay,save_checkpoint=save_checkpoint, cuda=current_cuda,fp16=False)

    # load trained model
    model.load_state_dict(torch.load(save_checkpoint))
    
    # load model into gpu or cpu
    if current_cuda:
        model.cuda()
    else:
        model.cpu()
    
    # calculate reference vector
    print("Calculate support prototype vector")
    
    nonregion_ref = dat_train[dat_train.region == 0]
    region_ref = dat_train[dat_train.region == 1]
    
    while len(region_ref)<len(nonregion_ref):
        region_ref = pd.concat([region_ref, region_ref], axis=0) 
    region_ref = region_ref.iloc[range(0,len(nonregion_ref))]
    
    use_gpu = current_cuda
    if use_gpu:
        support_region_nonregion = torch.tensor(np.zeros(shape = (len(region_ref),1,1,2,256))).to("cuda")
    else:
        support_region_nonregion = torch.tensor(np.zeros(shape = (len(region_ref),1,1,2,256))).to("cpu")
    
    with torch.no_grad():
        for j in tqdm(range(0,len(region_ref))):
            #print(j)
            support_tls, support_tls_mask = tokenizer(region_ref.iloc[j].sentence)
            support_nontls, support_nontls_mask = tokenizer(nonregion_ref.iloc[j].sentence)
            support = []
            support.append([])
            support.append([])
            support_mask = []
            support_mask.append([])
            support_mask.append([])
            support[0] = np.array(support_nontls)
            support[1] = np.array(support_tls)
            support_mask[0] = np.array(support_nontls_mask)
            support_mask[1] = np.array(support_tls_mask)
            support = torch.tensor(support, dtype=torch.long)
            support_mask = torch.tensor(support_mask, dtype=torch.long)
            if use_gpu:
                support = support.cuda()
                support_mask = support_mask.cuda()
            support = support.view(-1, max_length)  # [B * N * K, max_length]
            support_mask = support_mask.view(-1, max_length)
            support = model.encoder(support, support_mask)  # [B * N * K, D]
            support = model.mlp(support)
            support = support.view(-1, 2, 1, 256)  # [B, N, K, D]
            support = support.mean(2, keepdim=False)
            support = support.unsqueeze(1).expand(-1, 1, -1, -1)  # [B, totalQ, N, D]
            support_region_nonregion[j] = support
            
    support_tls_mean = np.array([list(range(0,len(support_region_nonregion[0][0][0][0])))])
    for n in tqdm(range(0,sum(dat_train.region == 1))):
        tmp = np.array([support_region_nonregion[n][0][0][1].to("cpu")])
        support_tls_mean = np.concatenate((support_tls_mean,tmp),axis = 0)
    support_tls_mean = torch.tensor(support_tls_mean)
    support_tls_mean = support_tls_mean[1:].mean(0, keepdim=True)
    
    support_nontls_mean = np.array([list(range(0,len(support_region_nonregion[0][0][0][0])))])
    for n in tqdm(range(0,len(support_region_nonregion))):
        tmp = np.array([support_region_nonregion[n][0][0][0].to("cpu")])
        support_nontls_mean = np.concatenate((support_nontls_mean,tmp),axis = 0)
    support_nontls_mean = torch.tensor(support_nontls_mean)
    support_nontls_mean = support_nontls_mean[1:].mean(0, keepdim=True)
    
    support_region_nonregion = np.concatenate((support_nontls_mean,support_tls_mean),axis = 0)
    
    if use_gpu:
        support_region_nonregion =  torch.tensor(np.array([[[support_region_nonregion]]])).to("cuda")
    else:
        support_region_nonregion =  torch.tensor(np.array([[[support_region_nonregion]]])).to("cpu")
        
    with open(save_checkpoint_path+"/train_support_region_nonregion_vec.pkl","wb") as f:
        dill.dump(support_region_nonregion,f)
        
    if use_gpu:
        predict_label = torch.tensor(np.zeros(shape = (len(dat_train)))).to("cuda")
        region_distance = torch.tensor(np.zeros(shape = (len(dat_train),2))).to("cuda")
        query_train = torch.tensor(np.zeros(shape = (len(dat_train),256))).to("cuda")
    else:
        predict_label = torch.tensor(np.zeros(shape = (len(dat_train)))).to("cpu")
        region_distance = torch.tensor(np.zeros(shape = (len(dat_train),2))).to("cpu")
        query_train = torch.tensor(np.zeros(shape = (len(dat_train),256))).to("cpu")
        
    support = support_region_nonregion[0]
    
    # pred
    
    print("Predict single cells or spots")
    
    with torch.no_grad():
        for i in tqdm(range(0,len(dat_train))):
            #print(i)
            query, query_mask = tokenizer(dat_train["sentence"][i])
            query = np.array(query)
            query_mask = np.array(query_mask)
            query = torch.tensor(query, dtype=torch.long)
            query_mask = torch.tensor(query_mask, dtype=torch.long)
            if use_gpu:
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
    region_distance.index = dat_train.index
    region_distance.columns = ["Non-TLS Distance","TLS Distance"]
    dat_train = pd.concat([dat_train,region_distance],axis=1)
    dat_train["pred_label"] = predict_label.cpu().tolist()
    dat_train["relative_distance"] = dat_train["TLS Distance"] - dat_train["Non-TLS Distance"]
    
    return dat_train
