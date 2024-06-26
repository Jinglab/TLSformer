% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/run_model_pred.r
\name{run_tlsformer_pred}
\alias{run_tlsformer_pred}
\title{Predict single cell or spots whether belong to TLS region}
\usage{
run_tlsformer_pred(
  seu_obj,
  pretrained_model_path,
  save_checkpoint_path,
  envir_path,
  pretrained_model = "TLSformer_BERT",
  sen_len = 260
)
}
\arguments{
\item{seu_obj}{The seurat object which will be used for predicting.}

\item{pretrained_model_path}{The pre-trained gene word encoder model saved path.}

\item{save_checkpoint_path}{The save path of TLSformer trained model.}

\item{envir_path}{The python env path.}

\item{pretrained_model}{Default is TLSPredictor_BERT. TLSformer_BERT or geneformer.}

\item{sen_len}{Default is 260. The sentence length, the generated sentences length will be minus or equal this parameter. If the gene expression level is zero, the gene will not be invovled.}
}
\value{
The relative distance of predicted single cells or spots with TLS prototype and non-TLS prototype, the prediction label of whether a single cell or spot belong to TLS region.
}
\description{
Run TLSformer trained model to calculate relative distance of single cells or spot with reference TLSs or Non-TLSs spots and infer whether ir belong to TLS region.
}
