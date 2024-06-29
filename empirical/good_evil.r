library(xtable)

make_analysis <- function( df ) {

    ## from orginal paper

    #####DEFINE FUNCTIONS##########
    #Calculate norm of vector#
    norm_vec <- function(x) sqrt(sum(x^2))

    #Dot product#
    dot <- function(x,y) (sum(x*y))

    #Cosine Similarity#
    cos <- function(x,y) dot(x,y)/norm_vec(x)/norm_vec(y)

    #Normalize vector#
    nrm <- function(x) x/norm_vec(x)

    #Calculate semantic dimension from antonym pair#
    dimension<-function(x,y) nrm(nrm(x)-nrm(y))

    ###STORE EMBEDDING AS MATRIX, NORMALIZE WORD VECTORS###
    cdfm<-as.matrix(data.frame(df))
    cdfmn<-t(apply(cdfm,1,nrm))


    ###IMPORT LISTS OF TERMS TO PROJECT AND ANTONYM PAIRS#####
    ant_pairs_good <- read.csv("word_pairs/good_evil.csv",header=FALSE, stringsAsFactor=F)

    word_dims<-matrix(NA,nrow(ant_pairs_good),ncol( df ))


    ###SETUP "make_dim" FUNCTION, INPUT EMBEDDING AND ANTONYM PAIR LIST#######
    ###OUTPUT AVERAGE SEMANTIC DIMENSION###

    make_dim<-function(embedding,pairs){
        word_dims<-data.frame(matrix(NA,nrow(pairs),ncol( df )))
        for (j in 1:nrow(pairs)){
            rp_word1<-pairs[j,1]
            rp_word2<-pairs[j,2]
            tryCatch(
                word_dims[j,]<-dimension(embedding[rp_word1,],embedding[rp_word2,]),
                error=function(e){}
            )
        }
        dim_ave<-colMeans(word_dims, na.rm = TRUE)
        dim_ave_n<-nrm(dim_ave)
        return(dim_ave_n)
    }

    good_dim<-make_dim(df,ant_pairs_good)

    good_proj<-cdfmn%*%good_dim
    projections_df<-cbind(good_proj)
    colnames(projections_df)<-c("good_proj")

    wlist=c("politician", "worker", "banker", "consultant", "marketer", "farmer", "priest", "teacher", "capitalist", "communist")

    df <- data.frame(projections_df[wlist,])
    df$occupation <- row.names( df )

    return( df )

}

df <-read.csv(file="marxist-embedding.csv", header=FALSE, sep=",")

## custom code to modify the CSV for a working format
df <- df[ ! is.na( df$V1 ), ]
df$V1 <- gsub( "▁", "", df$V1 )
df <- df[!duplicated(df[ , c("V1")]),]
row.names( df ) <- df$V1
df$V1 <- NULL

df.marxist <- make_analysis( df )
colnames( df.marxist ) <- c("Marxist LLM", "occupation")


df <-read.csv(file="capitalist-embedding.csv", header=FALSE, sep=",")

## custom code to modify the CSV for a working format
df <- df[ ! is.na( df$V1 ), ]
df$V1 <- gsub( "▁", "", df$V1 )
df <- df[!duplicated(df[ , c("V1")]),]
row.names( df ) <- df$V1
df$V1 <- NULL

df.capitalist <- make_analysis( df )
colnames( df.capitalist ) <- c("Capitalist LLM", "occupation")

outcome <- merge( df.marxist, df.capitalist, by = "occupation")

outcome <- outcome[ order( outcome[["Marxist LLM"]] ), ]

print( xtable( outcome ), file = "good-evil.tex", include.rownames=FALSE )

print( cor( outcome[["Marxist LLM"]] , outcome[["Capitalist LLM"]], method = 'spearman' ) )
