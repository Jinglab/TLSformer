#' @title Predict single cell or spots whether belong to TLS region
#' @description Run TLSformer trained model to calculate relative distance of single cells or spot with reference TLSs or Non-TLSs spots and infer whether ir belong to TLS region. 
#' 
#' @param seu_obj The seurat object which will be used for predicting.
#' @param pretrained_model_path The pre-trained gene word encoder model saved path.
#' @param save_checkpoint_path The save path of TLSformer trained model.
#' @param envir_path The python env path.
#' @param pretrained_model Default is TLSPredictor_BERT. TLSformer_BERT or geneformer.
#' @param sen_len Default is 260. The sentence length, the generated sentences length will be minus or equal this parameter. If the gene expression level is zero, the gene will not be invovled.
#'
#'
#' @return The relative distance of predicted single cells or spots with TLS prototype and non-TLS prototype, the prediction label of whether a single cell or spot belong to TLS region.
#' @export 

run_tlsformer_pred <- function(seu_obj,pretrained_model_path,save_checkpoint_path,envir_path,
                                  pretrained_model = "TLSformer_BERT", sen_len=260){
    reticulate::use_condaenv(envir_path, required = TRUE)
    reticulate::source_python(system.file("python", "pred_singlecells_spots.py", package = "TLSformer"))
    dat_pred <- pred_tls(dat_pred = seu_obj@meta.data,
                                pretrained_model = pretrained_model,
                                sen_len = as.integer(sen_len),
                                pretrained_model_path = pretrained_model_path,
                                save_checkpoint_path = save_checkpoint_path)
    rownames(dat_pred) <- dat_pred$cell_barcode
    seu_obj@meta.data <- cbind(seu_obj@meta.data,dat_pred[,c("Non-TLS Distance","TLS Distance", "relative_distance","pred_label")])
    return(seu_obj)
}