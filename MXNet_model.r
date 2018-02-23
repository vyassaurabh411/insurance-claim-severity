# reference : https://www.kaggle.com/delemeator/xgboost-mxnet-in-r

#clean memory
rm(list=ls())
gc()

library(devtools)
library(data.table)
library(xgboost)
library(Matrix)
library(MASS)
library(mxnet)


devices <- mx.cpu()
#read file in data frame
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SUBMISSION_FILE = "logtest2.csv"
train=fread(TRAIN_FILE)
test=fread(TEST_FILE)

# save id & loss
final=data.table(id=test$id)
oob_id=train$id
train_Y=train$loss


# define columns for categorical variables
take_columns <- function(file, test_file) {
  f1 <- fread(file)
  f2 <- fread(test_file)
  
  data <- rbind(f1, f2, fill = TRUE)
  data <- data[,grep('(cat|id)', colnames(data)), with = FALSE]
  
  data <- melt(data, id.vars = 'id') # 2D -> 1D
  data <- unique(data[,.(variable, value)]) # take unique columns
  data[,variable_value := paste(variable, value, sep = "_")] #name columns like cat2_A
  setorder(data, variable, value)
  data[,n_var := .N, by = variable] # check number of values for each variable
  data[n_var > 2,column := 1:.N] # those with number of values will be binary coded
  data[,n_var := NULL]
  data[,lin_val := (0:(.N-1))/(.N-1), by = variable] # all variables will be also lex coded (A - 0, B - 0.5, C - 1)
  
  return(data)
}

# read data
load_data <- function(file, columns) {
  data <- fread(file)
  
  # split variables between categorical and numerical
  cn <- colnames(data)
  c_cat <- c("id", cn[grep("cat", cn)]) 
  c_num <- c(cn[-grep("cat", cn)])
  
  cat <- data[,c_cat, with = F]
  num <- data[,c_num, with = F]
  
  cat <- melt(cat, id.vars = "id", measure.vars = c_cat[-1]) # 2D -> 1D
  
  rows <- cat[, .(id = unique(id))] #remember row numers for all id's
  rows[, row := 1:.N]
  
  cat <- columns[cat,,on = c("variable", "value")]
  
  ### assign lex coding values
  lin_cat <- dcast(cat[,.(id, variable, lin_val)], id ~ variable, value.var = 'lin_val', fill = 0)
  lin_cat <- lin_cat[rows[,.(id)],,on = 'id']
  lin_cat <- Matrix(as.matrix(lin_cat[,-'id',with = FALSE]), sparse = TRUE)
  ###
  
  ### assign binary coding
  cat <- cat[!is.na(column), ]
  cat <- rows[cat,,on = "id"]
  
  ### sparse matrix
  cat_mat <- sparseMatrix(i = cat[,row], j = cat[,column], x = 1)
  colnames(cat_mat) <- columns[!is.na(column),variable_value]
  
  num <- Matrix(as.matrix(num[,-'id',with=FALSE]), sparse = TRUE)
  
  ### bind all variables
  data <- cBind(num, cat_mat, lin_cat)
  print("Data loaded")
  return(list(data = data, rows = rows, columns = columns))
}



#load data
columns <- take_columns(TRAIN_FILE, TEST_FILE)
data_list <- load_data(TRAIN_FILE, columns)
test_list <- load_data(TEST_FILE, columns)
data <- data_list$data
test_data <- test_list$data
dim(data)
dim(test_data)


# take index of response
y <- which(colnames(data) == 'loss')
# set cox-box parameter
lambda = 0.3

# take train sample
train_obs <- get_train_sample(frac = 0.8)


# params for nn
params <- list(
  learning.rate = 0.0001,
  momentum = 0.9,
  batch.size = 128,
  wd = 0,
  num.round = 45
)

n = 5
MAE_valid = 0
bagged_pred = 0
set.seed(0)
folds <- cut(seq(1,nrow(data)),breaks=n,labels=FALSE)
oob=data.frame(id=NULL,actual=NULL,pred_mxnet=NULL)


for(j in 1:n){
  #Segement your data by fold using the which() function 
  testIndexes <- which(folds==j,arr.ind=TRUE)
  #train one net
  inp <- mx.symbol.Variable('data')
  l1 <- mx.symbol.FullyConnected(inp, name = "l1", num.hidden = 395)
  a1 <- mx.symbol.Activation(l1, name = "a1", act_type = 'relu')
  d1 <- mx.symbol.Dropout(a1, name = 'd1', p = 0.4)
  l2 <- mx.symbol.FullyConnected(d1, name = "l2", num.hidden = 197)
  a2 <- mx.symbol.Activation(l2, name = "a2", act_type = 'relu')
  d2 <- mx.symbol.Dropout(a2, name = 'd2', p = 0.2)
  l3 <- mx.symbol.FullyConnected(d2, name = "l3", num.hidden = 98)
  a3 <- mx.symbol.Activation(l3, name = "a3", act_type = 'relu')
  d3 <- mx.symbol.Dropout(a3, name = 'd3', p = 0.2)
  l4 <- mx.symbol.FullyConnected(d3, name = "l4", num.hidden = 1)
  outp <- mx.symbol.MAERegressionOutput(l4, name = "outp")
                  
  DeepModel <- mx.model.FeedForward.create(outp, 
                       X = as.array(t(data[-testIndexes, -y])), 
                       y = as.array(data[-testIndexes, y]),
                       eval.data = list(data = as.array(t(data[testIndexes, -y])),
                                        label = as.array(data[testIndexes, y])),
                       array.layout = 'colmajor',ctx=devices,
                       eval.metric=mx.metric.mae,
                       learning.rate = params$learning.rate,
                       momentum = params$momentum,
                       wd = params$wd,
                       array.batch.size = params$batch.size,
                       num.round = params$num.round)

  pred_valid <- predict(DeepModel, as.array(t(data[testIndexes, -y])), array.layout = 'colmajor')
  oob = rbind(oob, cbind(id = oob_id[testIndexes],pred_mxnet = t(pred_valid)))
  pred_test <- predict(DeepModel, as.array(t(test_data)), array.layout = 'colmajor')
  #pred_train <- predict(m, as.array(t(data)), array.layout = 'colmajor')
  MAE_valid <- MAE_valid + mean(abs(pred_valid - data[testIndexes, y]))
  #train_bagged = train_bagged + pred_train
  bagged_pred = bagged_pred + pred_test
}

colnames(oob) = c("id","loss")
MAE_valid = MAE_valid/(n)
bagged_pred = t(bagged_pred/(n))
summary(bagged_pred)
submission = fread(SUBMISSION_FILE, colClasses = c("integer", "numeric"))
submission$loss = bagged_pred
write.csv(submission,'submission_MXNet_cv_1151.csv',row.names = FALSE)
write.csv(oob,"MXnet_oob_prediction.csv",row.names = F)
