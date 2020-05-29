library(stats)
library(kernlab)
library(splines)
library(fda)

##------------ QUESTION 3 -----------##
#PART A
data <- read.csv('load_data.csv', header = FALSE, fileEncoding="UTF-8-BOM")
nrow(data)
ncol(data)

X <- seq(0,1,length.out = 200)
X[1:15]
Y <- c()
for (i in 1:200)
{
  Y[i] <- mean(as.matrix(data[i]))
}
Y[1:20]

MSElist <- matrix(nrow=10,ncol=2)
MSElist[,1] <- c(seq(6,15,by=1))

#CUBIC
h1 = rep(1,length(X))
h2 = X
h3 = X^2
h4 = X^3

# k=6
k <- seq(0,1,length.out = 8)
k <- k[2:7]
h5 = (X-k[1])^3
h5[h5 <= 0] = 0
h6 = (X-k[2])^3
h6[h6 <= 0] = 0
h7 = (X-k[3])^3
h7[h7 <= 0] = 0
h8 = (X-k[4])^3
h8[h8 <= 0] = 0
h9 = (X-k[5])^3
h9[h9 <= 0] = 0
h10 = (X-k[6])^3
h10[h10 <= 0] = 0
H = cbind(h1,h2,h3,h4,h5,h6,h7,h8,h9,h10)
B=solve(t(H)%*%H)%*%t(H)%*%Y
plot(X,Y)
lines(X,H%*%B,col = "red",lwd = 3)

yhat <- H%*%B
MSE <- sum((yhat-Y)^2)/200
MSElist[1,2] <- MSE

# k=7
k <- seq(0,1,length.out = 9)
k <- k[2:8]
h5 = (X-k[1])^3
h5[h5 <= 0] = 0
h6 = (X-k[2])^3
h6[h6 <= 0] = 0
h7 = (X-k[3])^3
h7[h7 <= 0] = 0
h8 = (X-k[4])^3
h8[h8 <= 0] = 0
h9 = (X-k[5])^3
h9[h9 <= 0] = 0
h10 = (X-k[6])^3
h10[h10 <= 0] = 0
h11 = (X-k[7])^3
h11[h11 <= 0] = 0
H = cbind(h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11)
B=solve(t(H)%*%H)%*%t(H)%*%Y
plot(X,Y)
lines(X,H%*%B,col = "red",lwd = 3)

yhat <- H%*%B
MSE <- sum((yhat-Y)^2)/200
MSElist[2,2] <- MSE

# k=8
k <- seq(0,1,length.out = 10)
k <- k[2:9]
h5 = (X-k[1])^3
h5[h5 <= 0] = 0
h6 = (X-k[2])^3
h6[h6 <= 0] = 0
h7 = (X-k[3])^3
h7[h7 <= 0] = 0
h8 = (X-k[4])^3
h8[h8 <= 0] = 0
h9 = (X-k[5])^3
h9[h9 <= 0] = 0
h10 = (X-k[6])^3
h10[h10 <= 0] = 0
h11 = (X-k[7])^3
h11[h11 <= 0] = 0
h12 = (X-k[8])^3
h12[h12 <= 0] = 0
H = cbind(h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12)
B=solve(t(H)%*%H)%*%t(H)%*%Y
plot(X,Y)
lines(X,H%*%B,col = "red",lwd = 3)

yhat <- H%*%B
MSE <- sum((yhat-Y)^2)/200
MSElist[3,2] <- MSE

# k=9
k <- seq(0,1,length.out = 11)
k <- k[2:10]
h5 = (X-k[1])^3
h5[h5 <= 0] = 0
h6 = (X-k[2])^3
h6[h6 <= 0] = 0
h7 = (X-k[3])^3
h7[h7 <= 0] = 0
h8 = (X-k[4])^3
h8[h8 <= 0] = 0
h9 = (X-k[5])^3
h9[h9 <= 0] = 0
h10 = (X-k[6])^3
h10[h10 <= 0] = 0
h11 = (X-k[7])^3
h11[h11 <= 0] = 0
h12 = (X-k[8])^3
h12[h12 <= 0] = 0
h13 = (X-k[9])^3
h13[h13 <= 0] = 0
H = cbind(h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13)
B=solve(t(H)%*%H)%*%t(H)%*%Y
plot(X,Y)
lines(X,H%*%B,col = "red",lwd = 3)

yhat <- H%*%B
MSE <- sum((yhat-Y)^2)/200
MSElist[4,2] <- MSE

# k=10
k <- seq(0,1,length.out = 12)
k <- k[2:11]
h5 = (X-k[1])^3
h5[h5 <= 0] = 0
h6 = (X-k[2])^3
h6[h6 <= 0] = 0
h7 = (X-k[3])^3
h7[h7 <= 0] = 0
h8 = (X-k[4])^3
h8[h8 <= 0] = 0
h9 = (X-k[5])^3
h9[h9 <= 0] = 0
h10 = (X-k[6])^3
h10[h10 <= 0] = 0
h11 = (X-k[7])^3
h11[h11 <= 0] = 0
h12 = (X-k[8])^3
h12[h12 <= 0] = 0
h13 = (X-k[9])^3
h13[h13 <= 0] = 0
h14 = (X-k[10])^3
h14[h14 <= 0] = 0
H = cbind(h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14)
B=solve(t(H)%*%H)%*%t(H)%*%Y
plot(X,Y)
lines(X,H%*%B,col = "red",lwd = 3)

yhat <- H%*%B
MSE <- sum((yhat-Y)^2)/200
MSE
MSElist[5,2] <- MSE

# k=11
k <- seq(0,1,length.out = 13)
k <- k[2:12]
h5 = (X-k[1])^3
h5[h5 <= 0] = 0
h6 = (X-k[2])^3
h6[h6 <= 0] = 0
h7 = (X-k[3])^3
h7[h7 <= 0] = 0
h8 = (X-k[4])^3
h8[h8 <= 0] = 0
h9 = (X-k[5])^3
h9[h9 <= 0] = 0
h10 = (X-k[6])^3
h10[h10 <= 0] = 0
h11 = (X-k[7])^3
h11[h11 <= 0] = 0
h12 = (X-k[8])^3
h12[h12 <= 0] = 0
h13 = (X-k[9])^3
h13[h13 <= 0] = 0
h14 = (X-k[10])^3
h14[h14 <= 0] = 0
h15 = (X-k[11])^3
h15[h15 <= 0] = 0
H = cbind(h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15)
B=solve(t(H)%*%H)%*%t(H)%*%Y
plot(X,Y)
lines(X,H%*%B,col = "red",lwd = 3)

yhat <- H%*%B
MSE <- sum((yhat-Y)^2)/200
MSElist[6,2] <- MSE

# k=12
k <- seq(0,1,length.out = 14)
k <- k[2:13]
h5 = (X-k[1])^3
h5[h5 <= 0] = 0
h6 = (X-k[2])^3
h6[h6 <= 0] = 0
h7 = (X-k[3])^3
h7[h7 <= 0] = 0
h8 = (X-k[4])^3
h8[h8 <= 0] = 0
h9 = (X-k[5])^3
h9[h9 <= 0] = 0
h10 = (X-k[6])^3
h10[h10 <= 0] = 0
h11 = (X-k[7])^3
h11[h11 <= 0] = 0
h12 = (X-k[8])^3
h12[h12 <= 0] = 0
h13 = (X-k[9])^3
h13[h13 <= 0] = 0
h14 = (X-k[10])^3
h14[h14 <= 0] = 0
h15 = (X-k[11])^3
h15[h15 <= 0] = 0
h16 = (X-k[12])^3
h16[h16 <= 0] = 0
H = cbind(h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16)
B=solve(t(H)%*%H)%*%t(H)%*%Y
plot(X,Y)
lines(X,H%*%B,col = "red",lwd = 3)

yhat <- H%*%B
MSE <- sum((yhat-Y)^2)/200
MSElist[7,2] <- MSE

# k=13
k <- seq(0,1,length.out = 15)
k <- k[2:14]
h5 = (X-k[1])^3
h5[h5 <= 0] = 0
h6 = (X-k[2])^3
h6[h6 <= 0] = 0
h7 = (X-k[3])^3
h7[h7 <= 0] = 0
h8 = (X-k[4])^3
h8[h8 <= 0] = 0
h9 = (X-k[5])^3
h9[h9 <= 0] = 0
h10 = (X-k[6])^3
h10[h10 <= 0] = 0
h11 = (X-k[7])^3
h11[h11 <= 0] = 0
h12 = (X-k[8])^3
h12[h12 <= 0] = 0
h13 = (X-k[9])^3
h13[h13 <= 0] = 0
h14 = (X-k[10])^3
h14[h14 <= 0] = 0
h15 = (X-k[11])^3
h15[h15 <= 0] = 0
h16 = (X-k[12])^3
h16[h16 <= 0] = 0
h17 = (X-k[13])^3
h17[h17 <= 0] = 0
H = cbind(h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16,h17)
B=solve(t(H)%*%H)%*%t(H)%*%Y
plot(X,Y)
lines(X,H%*%B,col = "red",lwd = 3)

yhat <- H%*%B
MSE <- sum((yhat-Y)^2)/200
MSElist[8,2] <- MSE

# k=14
k <- seq(0,1,length.out = 16)
k <- k[2:15]
h5 = (X-k[1])^3
h5[h5 <= 0] = 0
h6 = (X-k[2])^3
h6[h6 <= 0] = 0
h7 = (X-k[3])^3
h7[h7 <= 0] = 0
h8 = (X-k[4])^3
h8[h8 <= 0] = 0
h9 = (X-k[5])^3
h9[h9 <= 0] = 0
h10 = (X-k[6])^3
h10[h10 <= 0] = 0
h11 = (X-k[7])^3
h11[h11 <= 0] = 0
h12 = (X-k[8])^3
h12[h12 <= 0] = 0
h13 = (X-k[9])^3
h13[h13 <= 0] = 0
h14 = (X-k[10])^3
h14[h14 <= 0] = 0
h15 = (X-k[11])^3
h15[h15 <= 0] = 0
h16 = (X-k[12])^3
h16[h16 <= 0] = 0
h17 = (X-k[13])^3
h17[h17 <= 0] = 0
h18 = (X-k[14])^3
h18[h18 <= 0] = 0
H = cbind(h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16,h17,h18)
B=solve(t(H)%*%H)%*%t(H)%*%Y
plot(X,Y)
lines(X,H%*%B,col = "red",lwd = 3)

yhat <- H%*%B
MSE <- sum((yhat-Y)^2)/200
MSElist[9,2] <- MSE

# k=15
k <- seq(0,1,length.out = 17)
k <- k[2:16]
h5 = (X-k[1])^3
h5[h5 <= 0] = 0
h6 = (X-k[2])^3
h6[h6 <= 0] = 0
h7 = (X-k[3])^3
h7[h7 <= 0] = 0
h8 = (X-k[4])^3
h8[h8 <= 0] = 0
h9 = (X-k[5])^3
h9[h9 <= 0] = 0
h10 = (X-k[6])^3
h10[h10 <= 0] = 0
h11 = (X-k[7])^3
h11[h11 <= 0] = 0
h12 = (X-k[8])^3
h12[h12 <= 0] = 0
h13 = (X-k[9])^3
h13[h13 <= 0] = 0
h14 = (X-k[10])^3
h14[h14 <= 0] = 0
h15 = (X-k[11])^3
h15[h15 <= 0] = 0
h16 = (X-k[12])^3
h16[h16 <= 0] = 0
h17 = (X-k[13])^3
h17[h17 <= 0] = 0
h18 = (X-k[14])^3
h18[h18 <= 0] = 0
h19 = (X-k[15])^3
h19[h19 <= 0] = 0
H = cbind(h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16,h17,h18,h19)
B=solve(t(H)%*%H)%*%t(H)%*%Y
plot(X,Y)
lines(X,H%*%B,col = "red",lwd = 3)

yhat <- H%*%B
MSE <- sum((yhat-Y)^2)/200
MSElist[10,2] <- MSE

MSElist
plot(MSElist)
lines(MSElist)


#B-Spline
X <- seq(0,1,length.out = 200)
MSElist <- matrix(nrow=10,ncol=2)
MSElist[,1] <- c(seq(6,15,by=1))
MSEs = c()

for (k in 6:15)
{
knots = seq(0,1,length.out = k+2)
B = bs(X, knots = knots, degree = 3)[,1:10]
yhat <- B%*%solve(t(B)%*%B)%*%t(B)%*%Y
MSEs <- c(MSEs, sum((yhat-Y)^2)/200)
}
MSElist[,2] <- MSEs
MSElist
plot(MSElist[1:7,])
lines(MSElist[1:7,])

knots = seq(0,1,length.out = 9)
B = bs(X, knots = knots, degree = 3)[,1:10]
yhat <- B%*%solve(t(B)%*%B)%*%t(B)%*%Y
sigma2 = (1/(n-11))*t(Y-yhat)%*%(Y-yhat)
yn = yhat-3*sqrt(diag(as.numeric(sigma2)*B%*%solve(t(B)%*%B)%*%t(B)))
yp = yhat+3*sqrt(diag(as.numeric(sigma2)*B%*%solve(t(B)%*%B)%*%t(B)))
plot(X,Y,col = "grey")
lines(X,yn,col = "blue")
lines(X,yp,col = "blue")
lines(X,yhat,col = "red")


#Smoothing Spline
X <- seq(0,1,length.out = 200)
RSS = rep(0,200)
df = rep(0,200)
n=200
k=10
for(i in 1:200)
  {
    yhat = smooth.spline(Y, df = k+1, spar = X[i])
    df[i] = yhat$df
    yhat = yhat$y
    RSS[i] = sum((yhat-Y)^2)
  }
# GCV criterion 
GCV = (RSS/n)/((1-df/n)^2)
plot(X,GCV,type = "l", lwd = 3)
spar = X[which.min(GCV)]
points(spar,GCV[which.min(GCV)],col = "red",lwd=5)
spar

yhat = smooth.spline(Y, df = k+1, spar = spar)
yhat = yhat$y
plot(X,Y,col = "black",lwd=3)
lines(X,yhat,col = "red",lwd=3)

MSE = sum((yhat-Y)^2)/200
MSE


#Gaussian Kernel Regression
X <- seq(1,200)
kerf = function(z){exp(-(z^2))/sqrt(2*pi)}
lamb = seq(.001,0.1,0.0001)
er = rep(0, length(Y))
mse = rep(0, length(lamb))
for(j in 1:length(lamb))
{
  h=lamb[j]
  for(i in 1:length(Y))
  {
    X1=X
    Y1=Y
    X1=X[-i]
    Y1=Y[-i]
    z=kerf((X[i]-X1)/h)
    yke=sum(z*Y1)/sum(z)
    er[i]=Y[i]-yke
  }
  mse[j]=sum(er^2)/200
}
plot(lamb,mse,type = "l")
h = lamb[which.min(mse)]
points(h,mse[which.min(mse)],col = "red", lwd=5)
mse
h

f = rep(0,200)
for(i in 1:length(Y))
{
X1=X
Y1=Y
X1=X[-i]
Y1=Y[-i]
z=kerf((X[i]-X1)/h)
f[i]=sum(z*Y1)/sum(z)
}

plot(X,Y)
lines(X,f,col="red")



##------------ QUESTION 4 -----------##

X_train <- seq(0,1,length.out = 200)
data_train = data[1:50,]
data_test = data[51:220,]
Y_train <- c()
for (i in 1:200)
{
  Y_train[i] <- mean(as.matrix(data_train[i]))
}
Y_train[1:20]


#Smoothing Spline
# PART A
RSS = rep(0,200)
df = rep(0,200)
n = 200
for(i in 1:200)
{
  yhat = smooth.spline(Y_train, df = k+1, spar = X_train[i])
  df[i] = yhat$df
  yhat = yhat$y
  RSS[i] = sum((yhat-Y_train)^2)
}
GCV = (RSS/n)/((1-df/n)^2)
spar = X_train[which.min(GCV)]
spar
ss = smooth.spline(Y_train,df=k+1,spar=spar)
ss_coef = ss$fit$coef
ss_coef

plot(X_train,GCV,type = "l", lwd = 3)
points(spar,GCV[which.min(GCV)],col = "red",lwd=5)


#PART B
quantile(ss_coef,c(0.001,0.999))
lb = -19.30175
ub = 180.51385

#PART C
fit = ksvm(as.matrix(data_train),as.factor(ss_coef[2:51]),type="C-svc",C=10,kernel="rbfdot",scaled = TRUE)
pred = predict(fit,data_test)
pred[1:5]
c_test <- rep(0,170)
for (i in 1:170)
{
  if (as.numeric(as.vector(pred[i]))>ub)
  {c_test[i] = 1}
}
for (i in 1:170)
{
  if (as.numeric(as.vector(pred[i]))<lb)
  {c_test[i] = 1}
}
c_test

#FPCA
splinebasis = create.bspline.basis(c(0,1),10)
smooth = smooth.basis(X_train,t(data_train),splinebasis)
Xfun = smooth$fd
pca = pca.fd(Xfun, 10)
var.pca = cumsum(pca$varprop)
nharm = sum(var.pca < 0.95) + 1
nharm
pc = pca.fd(Xfun, nharm)
FPCcoef = pc$scores
FPCcoef[1:5,]

FPCcoef2 <- FPCcoef[,1]+FPCcoef[,2]+FPCcoef[,3]+FPCcoef[,4]+FPCcoef[,5]
FPCcoef2

data_train

#PART D
quantile(FPCcoef2,c(0.001,0.999))
lb2 = -2.975884
ub2 =  1.948723

fit2 = ksvm(as.matrix(data_train),as.factor(FPCcoef2),type="C-svc",C=10,kernel="rbfdot",scaled = TRUE)
pred2 = predict(fit2,data_test)
pred2[1:5]
c_test2 = rep(0,170)
for (i in 1:170)
{
  if (as.numeric(as.vector(pred2[i]))>ub2)
  {c_test2[i] = 1}
}
for (i in 1:170)
{
  if (as.numeric(as.vector(pred2[i]))<lb2)
  {c_test2[i] = 1}
}
c_test2

