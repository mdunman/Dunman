#####----- Problem 2 -----#####
install.packages("glmnet")
library("glmnet")

rm(list=ls())
dev.off()

income = read.csv("income_democracy.csv", header=TRUE)
head(income)
income <- income[ , -which(names(income) %in% c("country","year","code"))]
income <- na.omit(income)
head(income)
y <- income$dem_ind

train_ind <- sample(seq_len(nrow(income)),size = nrow(income)*0.8)
train_x <- scale(as.matrix(income[train_ind,][ , -which(names(income) %in% c("dem_ind"))]))
test_x <- scale(as.matrix(income[-train_ind,][ , -which(names(income) %in% c("dem_ind"))]))
train_y <- y[train_ind]
test_y <- y[-train_ind]
head(train_x)
head(train_y)

typeof(train_x)

p = ncol(train_x)
n = nrow(train_x)

#Ridge Regression
ridge = cv.glmnet(train_x, train_y, family = "gaussian", alpha = 0, intercept = FALSE, standardize=TRUE)
lambda = ridge$lambda.min
lambda
coef.ridge = matrix(coef(ridge, s = lambda))[2:(p+1)]
ridge = glmnet(train_x, train_y, family = "gaussian", alpha = 0, intercept = FALSE, standardize=TRUE)
plot(ridge, xvar = "lambda", label = TRUE)
abline(v = log(lambda))
y_ridge = predict(ridge, test_x, s = lambda)
mse_ridge = sum((test_y-y_ridge)^2)/n
mse_ridge

#LASSO
lasso = cv.glmnet(train_x, train_y, family = "gaussian", alpha = 1, intercept = FALSE, standardize=TRUE)
lambda = lasso$lambda.min
lambda
coef.lasso = matrix(coef(lasso, s = lambda))[2:(p+1)]
lasso = glmnet(train_x, train_y, family = "gaussian", alpha = 1, intercept = FALSE, standardize=TRUE)
plot(lasso, xvar = "lambda", label = TRUE)
abline(v = log(lambda))
y_lasso = predict(lasso, test_x, s = lambda)
mse_lasso = sum((test_y-y_lasso)^2)/n
mse_lasso

#Adaptive LASSO
gamma = 2
b.ols = solve(t(train_x)%*%train_x)%*%t(train_x)%*%train_y
ridge = cv.glmnet(train_x, train_y, family = "gaussian", alpha = 0, intercept = FALSE)
l.ridge = ridge$lambda.min
b.ridge = matrix(coef(ridge, s = l.ridge))[2:(p+1)]
w1 = 1/abs(b.ols)^gamma
w2 = 1/abs(b.ridge)^gamma
alasso1 = cv.glmnet(train_x, train_y, family = "gaussian", alpha = 1, intercept = FALSE, penalty.factor = w1)
alasso2 = cv.glmnet(train_x, train_y, family = "gaussian", alpha = 1, intercept = FALSE, penalty.factor = w2)
lambda1 = alasso1$lambda.min
lambda1
lambda2 = alasso2$lambda.min
lambda2
coef.alasso1 = matrix(coef(alasso1, s = lambda1))[2:(p+1)]
coef.alasso2 = matrix(coef(alasso2, s = lambda2))[2:(p+1)]
alasso1 = glmnet(train_x, train_y, family = "gaussian", alpha = 1, intercept = FALSE, penalty.factor = w1)
alasso2 = glmnet(train_x, train_y, family = "gaussian", alpha = 1, intercept = FALSE, penalty.factor = w2)
par(mfrow=c(1,2))
plot(alasso1, xvar = "lambda", label = TRUE)
abline(v=log(lambda1))
plot(alasso2, xvar = "lambda", label = TRUE)
abline(v=log(lambda2))
y_alasso1 = predict(alasso1, test_x, s = lambda1)
mse_alasso1 = sum((test_y-y_alasso1)^2)/n
mse_alasso2
y_alasso2 = predict(alasso2, test_x, s = lambda2)
mse_alasso2 = sum((test_y-y_alasso2)^2)/n
mse_alasso2

#Elastic Net
a = seq(0.05, 0.95, 0.05)
enlist <- matrix(ncol=3,nrow=length(a))
set.seed(123)
for (i in a){
  en_model <- cv.glmnet(train_x, train_y, family="gaussian", alpha = i, intercept = FALSE, standardize=TRUE)
  enlist[which(a==i),1] = i
  enlist[which(a==i),2] = en_model$lambda.min
  enlist[which(a==i),3] = min(en_model$cvm)
}
al <- a[which(enlist[,3]==min(enlist[,3]))]
al
lambda <- enlist[which(enlist[,3]==min(enlist[,3])),2]
lambda
en = glmnet(train_x, train_y, family = "gaussian", alpha = al, intercept = FALSE, standardize=TRUE)
par(mfrow=c(1,1))
plot(en, xvar = "lambda", label = TRUE)
abline(v = log(lambda))
y_en = predict(en, test_x, s = lambda)
mse_en = sum((test_y-y_en)^2)/n
mse_en


#####----- Problem 3 -----#####
install.packages('R.matlab')
install.packages('pracma')
install.packages('gglasso')
library(fda)
library(pracma)
library(gglasso)
library(R.matlab)

rm(list=ls())
dev.off()

nsc <- readMat('NSC.mat')
xs <- nsc$x
y <- nsc$y
nsc_test <- readMat("NSC.test.mat")
xs_test <- nsc_test$x.test
y_test <- nsc_test$y.test

p <- 10
n <- 203
m <- 150
mt <- 50

x=list()
x_test=list()
for (i in 1:p)
{
  x[[i]] = as.matrix(xs[[i]][[1]])
  x_test[[i]] = as.matrix(xs_test[[i]][[1]])
}

par(mfrow=c(2,5))
for(i in 1:p)
{
  matplot(t(x[[i]]), type = "l", xlab = i,ylab = "")
}

par(mfrow=c(1,1))
matplot(t(y), type = "l", xlab = "",ylab = "")

#B-Spline Dimension Reduction
spb = 10
splinebasis_B=create.bspline.basis(c(0,1),spb)
q = seq(0,1,length=n)
base_B=eval.basis(as.vector(q),splinebasis_B)
X = array(dim=c(m,n,p))
for(i in 1:p)
{
  X[,,i] = x[[i]]
}
Z = array(dim=c(dim(X)[1],spb,p))
for(i in 1:p)
{
  Z[,,i] = X[,,i]%*%base_B/n 
}
Z = matrix(Z,m,spb*p)
Z = eye(p)%x%Z

X_test = array(dim=c(mt,n,p))
for(i in 1:p)
{
  X_test[,,i] = x_test[[i]]
}
Z_test = array(dim=c(dim(X_test)[1],spb,p))
for(i in 1:p)
{
  Z_test[,,i] = X_test[,,i]%*%base_B/n 
}
Z_test = matrix(Z_test,mt,spb*p)
Z_test = eye(p)%x%Z_test

Y = y%*%base_B/n
Y = as.matrix(as.vector(Y))

y_test = y_test%*%base_B/n
y_test = as.matrix(as.vector(y_test))

nrow(Z)
ncol(Z)
nrow(Y)
ncol(Y)

#Group LASSO
group = rep(1:10,each=100)
glasso = cv.gglasso(Z,Y,group,loss = "ls")
lambda = glasso$lambda.min
coef = matrix(coef(glasso,s="lambda.1se")[2:(m+1)],spb,p)
coef = base_B%*%coef
coef[50:55,]
par(mfrow=c(1,1),mar=c(5.1, 4.1, 4.1, 8.1), xpd=TRUE)
matplot(q,coef,col=c(1:10),lty=rep(1,10),type="l",lwd=2)
legend("topright", inset=c(-0.25,0),legend = c(1:10),col=c(1:10),lty = rep(1,10))

#Prediction
pred = predict(glasso, Z_test)
mse = mean(norm(pred-y_test,type='2'))
mse
