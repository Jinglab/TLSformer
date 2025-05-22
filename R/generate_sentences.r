#' @title Generate sentences
#' @description Based on the expression levels of each gene to generate sentence of each single cell or spot. 
#' 
#' @param seu_obj The seurat object. The gene expression counts matrix will be extracted from this object.
#' @param sen_len Default is 260. The length of gene word sentence, the generated sentences length will be minus or equal this parameter. Genes with zero expression levels will not be included in the sentences.
#' @param region_info The region information of TLS. If there lack or not contain region information, this parameter can be set as NULL. The region information 1 is represeted tha target region, 0 is represeted the other region
#' @param save_inseu Default is TRUE. Whether save sentences in the seurat meta.data. If set FALSE, the sentences will be save in a .txt file, which include the sentences and meta.data information. 
#' @param save_path If the save_inseu was setted as FALSE, the save path must be provided, this parameter is the sentences storage file path.
#' @param genes_representor The file path of genes were used in the pre-training the gene word encoder.
#' @param envir_path The python env path.
#'
#'
#' @return The sentences of single cells or spots which are saved in the meta.data of seurat object or saved in the single txt file.
#' @export 

generate_sentences <- function(seu_obj,sen_len=260,region_info,save_inseu=TRUE,save_path,genes_representor,envir_path){
    reticulate::use_condaenv(envir_path, required = TRUE)
    reticulate::source_python(system.file("python", "GPU_sort_index_makesentences.py", package = "TLSformer"))
    sc_dat_inputdf <- as.data.frame(seu_obj@assays[[1]]@counts)
    model_used_genes <- data.table::fread(genes_representor) %>%
        as.data.frame()
    model_used_genes <- model_used_genes$Gene
    input_gene = intersect(rownames(sc_dat_inputdf),model_used_genes)
    ### Input df: count matrix
    ### Save name: file path and prefix 
    ### Sen len: sentences length/the numbers of words
    ### Model gene list: the trained BERT used genes
    ### Region: the region of your interesting
    if(!is.null(input_gene)){
        mat <- sc_dat_inputdf[input_gene,]
        mat <- mat[!str_detect(rownames(mat),"^MT-"),]
        mat <- mat[!str_detect(rownames(mat),"^RP[SL]"),]
        mat <- mat[!str_detect(rownames(mat),"^MIR"),]
        mat <- mat[!str_detect(rownames(mat),"-AS1]"),]
        mat <- mat[!str_detect(rownames(mat),"^LINC]"),]
    }else{
        mat <- sc_dat_inputdf[!str_detect(rownames(sc_dat_inputdf),"\\."),]
        mat <- mat[!str_detect(rownames(mat),"^MT-"),]
        mat <- mat[!str_detect(rownames(mat),"^RP[SL]"),]
        mat <- mat[!str_detect(rownames(mat),"^MIR"),]
        mat <- mat[!str_detect(rownames(mat),"-AS1]"),]
        mat <- mat[!str_detect(rownames(mat),"^LINC]"),]
    }
    gsentence_df <- gpu_sort_index(mat,sen_len=as.integer(sen_len)) ## .py file
    gsentence_df <- as.data.frame(gsentence_df) %>%
        rownames_to_column()
    colnames(gsentence_df) <- c("cell_barcode","sentence")
    if(!is.null(region_info)){
        gsentence_df$region = region_info
    }
    if(save_inseu){
        seu_obj@meta.data$sentence <- gsentence_df$sentence
        return(seu_obj)
    }else{
        write.table(gsentence_df,file = paste0(save_path,"_sentence_rank",sen_len,".txt"),quote = F,row.names = F,col.names = T,sep = "\t")
    }
}

