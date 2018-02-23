
#Load the libraries
require(caret)
require(Matrix)
require(xgboost)
require(data.table)
require(Metrics)
require(scales)
require(snow)
require(coop)

#read the data set and convert the factors to numbers
train=fread("train.csv")
test=fread("test.csv")
# train_bac=copy(train)


# save id & loss
test_pred = data.table(id=test$id)
oob_id = train$id
train_Y = train$loss


#remove id and loss 
train[,c("id","loss"):=NULL]
test[,c("id"):=NULL]

train_test=rbind(train,test)
remove(train)
remove(test)

# indx=grep("cat",names(train_test),value = T)
# for(i in indx)
#   train_test[[i]]=as.integer(as.factor(train_test[[i]]))


indx=grep("cat",names(train_test),value=T)
for(i in indx)
{
  temp=train_bac[,list(mean=mean(loss)),by=i]
  temp=temp[order(mean)]
  temp$rank=as.integer(factor(temp$mean,levels = unique(temp$mean)))
  train_test[[i]]=temp$rank[match(train_test[[i]],temp[[i]])]
  train_test[[i]][is.na(train_test[[i]])]<--99
}



var=c("cat80","cat87","cat57","cat12","cat79","cat10","cat7","cat89","cat2","cat72",
      "cat81","cat11","cat1","cat13","cat9","cat3","cat16","cat90","cat23","cat36",
      "cat73","cat103","cat40","cat28","cat111","cat6","cat76","cat50","cat5",
      "cat4","cat14","cat38","cat24","cat82","cat25")

for(i in 1:(length(var)-1))
{
  for(j in (i+1):length(var))
  {
    train_test[,eval(as.name(paste(var[i],var[j],sep = "_"))):=eval(as.name(var[i]))*
                 eval(as.name(var[j]))]
    train_test[,eval(as.name(paste(var[i],var[j],sep = "/"))):=eval(as.name(var[i]))-
                 eval(as.name(var[j]))]

    }
  
}  
dim(train_test)

#correlation 
corr_mat=cor(train_test)
corrplot(corr_mat, order = "hclust", tl.cex = .35)
CorVariables <- findCorrelation(corr_mat,.95,names = T)

#removing highly correled variables//this removes a lot of interaction variables
train_test[,c(CorVariables) := NULL]
dim(train_test)

indx=grep("cat",names(train_test))
mat=as.matrix(train_test[,indx,with=F])
clus=makeCluster(6) #100/x
train_test$fecat=parRapply(clus,mat,function(x) sum(table(x)^2))
stopCluster(clus)
train_test$fecat=train_test$fecat/length(indx)^2


#clean-up
remove(mat,clus)
gc(verbose = F)


#simple log
train_X=train_test[1:length(train_Y),]
test_X = train_test[(length(train_Y)+1):nrow(train_test),]
dim(train_X)
dim(test_X)
length(train_Y)
shift=200
train_Y=log(train_Y+shift)



#create xgb model
dtest <- xgb.DMatrix(data = as.matrix(test_X))

logregobj <- function(preds, dtrain){
  labels = getinfo(dtrain, "label")
  con = .7
  x = preds-labels
  grad =con*x / (abs(x)+con)
  hess =con^2 / (abs(x)+con)^2
  return (list(grad = grad, hess = hess))
}


param=list(objective = logregobj,
           eta=0.002, 
           max_depth= 15,
           subsample=.7,
           colsample_bytree=.7,
           min_child_weight=100,
           base_score=7.76
           )


xg_eval_mae <- function (yhat, dtrain) {
  y = as.numeric(getinfo(dtrain, "label"))
  err= as.numeric(mae(expm1(y),expm1(yhat)))
  return (list(metric = "mae", value = round(err,4)))
}

set.seed(0)
#create fold
nfolds=5
folds=createFolds(train_Y,k=nfolds,list = T,returnTrain = T)
prediction=numeric(nrow(test_X))

oob=data.frame(id=NULL,real=NULL,pred=NULL)
for(i in 1:length(folds)) 
{
  cat('starting Fold',i,'\n')
  X_train=train_X[folds[[i]],]
  Y_train=train_Y[folds[[i]]]
  X_val=train_X[-folds[[i]],]
  Y_val=train_Y[-folds[[i]]]
  id_val=oob_id[-folds[[i]]]
  dtrain=xgb.DMatrix(data = as.matrix(X_train),label=Y_train)
  dtrain2=xgb.DMatrix(data = as.matrix(X_val),label=Y_val)
  watchlist=list(train=dtrain,test=dtrain2)
  model=xgb.train(params = param,data = dtrain,watchlist=watchlist, early_stopping_round = 100,   
                  feval=xg_eval_mae,print_every_n = 50,nrounds = 5000,maximize=FALSE)
  
  pred=predict(model,dtest)
  prediction=prediction+exp(pred)-shift
  dval=xgb.DMatrix(as.matrix(X_val))
  pred=exp(predict(model,dval))-shift
  oob=rbind(oob,cbind(id=id_val,real=exp(Y_val)-shift,pred=pred))
  
}

#final mae
print(mae(oob$real,oob$pred))
prediction = prediction/nfolds
test_pred$loss = prediction

#write file to disk
write.csv(final,"XGB_interaction_meanEncoding.csv",row.names = F)
write.csv(oob,"XGB_interaction_meanEncoding.csv",row.names = F)



