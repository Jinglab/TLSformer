#' @title Train TLSformer on spatial transcriptomics data
#' @description Train the TLSformer that is based on metrics learing few-shot learning to deal with the unblance problem. 
#' Beacuse The number of TLSs spots for training is generally much smaller that non-TLSs spots.    
#' 
#' @param seu_obj The seurat object which will be used for training.
#' @param pretrained_model Default is TLSformer_BERT. TLSformer_BERT or geneformer
#' @param sen_len Default is 260. The sentence length, the generated sentences length will be minus or equal this parameter. If the gene expression level is zero, the gene will not be invovled.
#' @param pretrained_model_path The pre-trained model saved path.
#' @param save_checkpoint_path The save path of model.
#' @param batch_size Batch size of training process.
#' @param train_K support set numbers.
#' @param train_Q query set numbers.
#' @param train_episodes training episodes.
#' @param envir_path The python env path.
#'
#'
#' @return The trained TLSformer model and support vector.
#' @export 

run_tlsformer_train <- function(seu_obj,pretrained_model = "TLSformer_BERT",
                               sen_len=260,pretrained_model_path,save_checkpoint_path,batch_size,
                               train_K,train_Q,train_episodes,envir_path){
    reticulate::use_condaenv(envir_path, required = TRUE)
    reticulate::source_python(system.file("python", "train_prototype_vector.py", package = "TLSformer"))
    dat_train <- train_prototype_hypersphere(dat_train = seu_obj@meta.data,
                                pretrained_model = pretrained_model,
                                sen_len = as.integer(sen_len),
                                pretrained_model_path = pretrained_model_path,
                                save_checkpoint_path = save_checkpoint_path,
                                batch_size = as.integer(batch_size),
                                train_K = as.integer(train_K),
                                train_Q = as.integer(train_Q), 
                                train_episodes = as.integer(train_episodes),
                                val_episodes = as.integer(30),
                                val_steps = as.integer(10))
    rownames(dat_train) <- dat_train$cell_barcode
    seu_obj@meta.data <- cbind(seu_obj@meta.data,dat_train[,c("Non-TLS Distance","TLS Distance", "relative_distance","pred_label")])
    return(seu_obj)
}