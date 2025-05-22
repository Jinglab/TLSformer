import warnings
warnings.filterwarnings("ignore")
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

def gpu_sort_index(df,sen_len):
    df_np = df.to_numpy()
    df_np_tensor = torch.Tensor(df_np)
    df_np_tensor = df_np_tensor.to('cpu')
    sorted, indices = torch.sort(df_np_tensor, dim=0, descending=True)
    sorted_df = pd.DataFrame(sorted)
    sorted_df.columns = df.columns
    indices_tonumpy = indices.to('cpu').numpy()
    indices_tonumpy = pd.DataFrame(indices_tonumpy)
    gene_listdf = pd.DataFrame(columns = ["sentences"],index=df.columns)
    indices_tonumpy.columns = df.columns
    for i in tqdm(df.columns):
        genes = list(df.index[indices_tonumpy[i][sorted_df[i]!=0]])
        if len(genes)>sen_len:
            gene_listdf.loc[i] = " ".join(genes[1:sen_len])
        else:
            gene_listdf.loc[i] = " ".join(genes)
    return gene_listdf