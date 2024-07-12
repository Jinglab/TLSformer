# TLSformer v1.0.0 </a>

# TLSformer: a scalable method for identifying individual cells in tertiary lymphoid structures by knowledge transfer of spatial transcriptomic data

### Xiaokai Fan<sup></sup>,  Mei Xie<sup></sup>, Xinying Xue*, Ying Jing*

TLSformer is a computational tool designed to assist immune researchers in identifying TLS cells/spots and calculating the probability that a cell/spot belongs to TLS regions. Unlike previous methods, TLSformer leverages bidirectional encoder representation from transformers (BERT) and meta-learning to acquire general knowledge from the limited available TLS-related information. This knowledge is then transferred to identify single cells or spots within TLS regions from scRNA-seq or spatial transcriptomics data.

## Requirements and Installation
This toolkit is written in both R and Python programming languages. The core BERT and meta-learning algorithm are implemented in Python, while the initial data preparation and functional usage are written in R.

### Installation of TLSformer

To use TLSformer, first use the [yaml file](https://github.com/Jinglab/TLSformer/blob/main/tlsformer_env.yml) to create TLSformer Conda environment.

    conda env create -f tlsformer_env.yml

After successfully creating the environment, the Python path of this environment can be found like this, which will be used in subsequent workflow steps.

    /home/xfan/miniconda3/envs/TLSformer_env/bin/python
    
Install TLSformer by devtools in R
[![R >4.0](https://img.shields.io/badge/R-%3E%3D4.0-brightgreen)](https://www.r-project.org/)

    devtools::install_github("Jinglab/TLSformer")
    
Alternatively, you can download the [TLSformer_1.0.tar.gz](https://github.com/Jinglab/TLSformer/blob/main/TLSformer_1.0.tar.gz) package file from this GitHub repository and install it locally.

    install.packages("home/xfan/MLTLS_package/TLSformer_1.0.tar.gz")

## Quick Start

### Run TLSformer under a TLS knowledge transfer scenario
To use TLSformer, no complex preprocessing is needed; we only require the counts from a Seurat object as input.
Download the 10x Visium breast cancer pre-trained BERT and demo data from Google Cloud. The saved location of this pre-trained BERT will be utilized in the subsequent workflow steps.
- [breast cancer pre-trained gene word encoder](https://drive.google.com/drive/folders/1qLsl22T3IU2EEyXYM3z52_8MLNsFDyjO?usp=drive_link)
- [demo data](https://drive.google.com/drive/folders/1DZJ-f_RjpnRUszXNKm_KRGXpbHcwsEBK?usp=drive_link)


1.Load package and demo data

    library(Seurat)
    library(TLSformer)
    library(reticulate)
    library(tidyverse)
    st_dat_train <- readRDS("home/xfan/MLTLS_package/demo_data/bc_st_demo_data.rds")
    st_dat_pred <- readRDS("home/xfan/MLTLS_package/demo_data/melanoma_st_demo_data.rds")

2.Set parameters

    sen_len = 260
    save_inseu = TRUE
    genes_representor = "home/xfan/MLTLS_package/demo_data/pretrained_models_rank260/genelist.txt"
    envir_path = "/home/xfan/miniconda3/envs/TLSformer_env/bin/python"
    pretrained_model = "TLSformer_BERT"
    pretrained_model_path = "home/xfan/MLTLS_package/demo_data/pretrained_models_rank260/"
    save_checkpoint_path = "home/xfan/MLTLS_package/demo_data/"
    batch_size = 1 # depend on your GPU memory to change
    train_K = 2 # depend on your GPU memory to change
    train_Q = 2 # depend on your GPU memory to change
    train_episodes = 600

3.Generate sentences
    
    # Training data
    st_dat_train <- generate_sentences(
      seu_obj = st_dat_train,
      sen_len = sen_len,
      region_info = st_dat_train@meta.data$region,
      save_inseu = save_inseu,
      genes_representor = genes_representor,
      envir_path = envir_path
    )
    
    # Predicton data
    st_dat_pred <- generate_sentences(
      seu_obj = st_dat_pred,
      sen_len = sen_len,
      region_info = st_dat_pred@meta.data$region,
      save_inseu = save_inseu,
      genes_representor = genes_representor,
      envir_path = envir_path
    )

4.Training TLSformer
    
    # Training
    st_dat_train <- run_tlsformer_train(
        seu_obj = st_dat_train,
        pretrained_model = pretrained_model,
        sen_len = sen_len,
        pretrained_model_path = pretrained_model_path,
        save_checkpoint_path = save_checkpoint_path,
        batch_size = batch_size,
        train_K = train_K,
        train_Q = train_Q,
        train_episodes = train_episodes,
        envir_path = envir_path
    )

5.Use trained TLSformer to predict

    # Prediction
    st_dat_pred <- run_tlsformer_pred(
                        seu_obj = st_dat_pred,
                        pretrained_model_path = pretrained_model_path,
                        save_checkpoint_path = save_checkpoint_path,
                        envir_path = envir_path,
                        pretrained_model = pretrained_model,
                        sen_len=sen_len)
    # Normalization -- 0-1 scale
    st_dat_pred$relative_distance <- 1- (st_dat_pred$relative_distance - min(st_dat_pred$relative_distance))/(max(st_dat_pred$relative_distance)-min(st_dat_pred$relative_distance))
    SpatialFeaturePlot(st_dat_pred,features = c("region","relative_distance"))

## Built With
  - [Python](https://www.python.org/)
  - [PyTorch](https://pytorch.org/)
  - [R](https://www.contributor-covenant.org/](https://www.r-project.org/about.html))


## Lab website

  - **YingJing Lab** - *Guangzhou, China* - https://www.yingjinglab.com
