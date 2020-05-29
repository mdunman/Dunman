%Gradient Descent with Fixed Step Size
load('emission-1.mat')
mu0 = [500;400;1000];
k = 1; tol = 0.01;
g = grad(mu0,p,y);
f0 = loglik(mu0,p,y);
fun = f0;
alpha = 1;
while g'*g > tol
    mu = mu0 + .01*g;
    mu0 = mu;
    g = grad(mu0,p,y);
    f0 = loglik(mu0,p,y);
    fun = [fun; f0];
    k = k+1;
end
k
g'*g
mu
figure; plot(fun)
xlabel('Number of iterations'); ylabel('Log Likelihood')
true = [749;365;1221];
error = 0;
for i = 1:3
    error = error + (mu(i)- true(i))^2;
end
error

%Accelerated Gradient Descent
mu0 = [500;400;1000];
z = [500;400;1000];
g = grad(mu0,p,y);
f0 = loglik(mu0,p,y);
fun = f0;
k = 1; tol = 0.01;
while g'*g > tol
    mu = z + 0.01*g;
    z = mu + (k-1)/(k+2)*(mu-mu0);
    mu0 = mu;
    g = grad(mu0,p,y);
    f0 = loglik(mu0,p,y);
    fun = [fun; f0];
    k = k+1;
end
k
g'*g
mu
figure; plot(fun)
xlabel('Number of iterations'); ylabel('Log Likelihood')
true = [749;365;1221];
error = 0;
for i = 1:3
    error = error + (mu(i)- true(i))^2;
end
error

%Stochastic Gradient Descent
mu0 = [500;400;1000];
z = [500;400;1000];
g = grad(mu0,p,y);
fun = loglik(mu,p,y);
K = 1000; n = 3;
k = 1; tol = 0.001;
while k <= K
    for i = 1:n
        s = randsample(n,1);
        mu = mu + 0.0001*grad(mu,p,y);
        f = loglik(mu,p,y);
    end
    fun = [f; fun];
    k = k+1;
end
k
g'*g
mu
figure; plot(fun)
xlabel('Number of iterations'); ylabel('Log Likelihood')
true = [749;365;1221];
error = 0;
for i = 1:3
    error = error + (mu(i)- true(i))^2;
end
error

%Newton's Method
mu0 = [500;400;1000];
f0 = loglik(mu0,p,y); fun = f0;
g0 = grad(mu0,p,y);
H = hessian(mu0,p,y);
alpha = 1; lambda = 10^(-10);
O = -(H+lambda*eye(3))\g0;
k = 1;
while k < 10
    mu = mu0 + alpha*O;
    f = loglik(mu,p,y);
    mu0 = mu;
    f0 = f;
    fun = [fun; f0];
    g0 = grad(mu0,p,y);
    O = -(H+lambda*eye(3))\g0;
    k = k+1;
end
k
g'*g
mu
figure; plot(fun)
xlabel('Number of iterations'),ylabel('Function value')
true = [749;365;1221];
error = 0;
for i = 1:3
    error = error + (mu(i)- true(i))^2;
end
error


%%%%% Problem 3 %%%%%
g = Fi - (b0 + b1*T)*(1-exp(-2*b2*t));
J = [ -1 + exp(-2*b2*t), -T + T*(1+exp(-2*b2*t)), -2*t*exp(-2*b2*t)*(b0+b1*T)];
gradg = [ -2*(1-exp(-2*b2*t))*(Fi-(b0+b1*T)*(1-exp(-2*b2*t)));
          -2*t*(1-exp(-2*b2*t))*(Fi-(b0+b1*T)*(1-exp(-2*b2*t)));
          -4*t*exp(-2*b2*t)*(b0 + b1*T)*(Fi-(b0+b1*T)*(1-exp(-2*b2*t)))];
hessg = 2*[(-(1-exp(-2*b2*t))*(Fi-(b0+b1*T)*(1-exp(-2*b2*t))))^2, 
           -(1-exp(-2*b2*t))*(Fi-(b0+b1*T)*(1-exp(-2*b2*t)))*-t*(1-exp(-2*b2*t))*(Fi-(b0+b1*T)*(1-exp(-2*b2*t))),
           -(1-exp(-2*b2*t))*(Fi-(b0+b1*T)*(1-exp(-2*b2*t)))*-2*t*exp(-2*b2*t)*(b0 + b1*T)*(Fi-(b0+b1*T)*(1-exp(-2*b2*t)));
           -t*(1-exp(-2*b2*t))*(Fi-(b0+b1*T)*(1-exp(-2*b2*t)))*-(1-exp(-2*b2*t))*(Fi-(b0+b1*T)*(1-exp(-2*b2*t))),
           -t*(1-exp(-2*b2*t))*(Fi-(b0+b1*T)*(1-exp(-2*b2*t)))^2,
           -t*(1-exp(-2*b2*t))*(Fi-(b0+b1*T)*(1-exp(-2*b2*t)))*-2*t*exp(-2*b2*t)*(b0 + b1*T)*(Fi-(b0+b1*T)*(1-exp(-2*b2*t)));
           -2*t*exp(-2*b2*t)*(b0 + b1*T)*(Fi-(b0+b1*T)*(1-exp(-2*b2*t)))*-(1-exp(-2*b2*t))*(Fi-(b0+b1*T)*(1-exp(-2*b2*t))),
           -2*t*exp(-2*b2*t)*(b0 + b1*T)*(Fi-(b0+b1*T)*(1-exp(-2*b2*t)))*-t*(1-exp(-2*b2*t))*(Fi-(b0+b1*T)*(1-exp(-2*b2*t))),
           (-2*t*exp(-2*b2*t)*(b0 + b1*T)*(Fi-(b0+b1*T)*(1-exp(-2*b2*t))))^2 ];

       


%Gauss-Newton Method
p1 = leak(1:84,:);
p2 = leak(85:168,:);
p3 = leak(169:252,:);
p4 = leak(253:336,:);
p5 = leak(337:420,:);

%Part i = 1
part = p1;
b0 = [15;12;3];
g0 = g(b0,part);
J0 = jacob(b0,part);
f0 = g0'*g0; fun = f0;
alpha = 1; lambda = 10^(-10);
O = -(J0'*J0+lambda*eye(3))\(J0'*g0);
tol = 10^(-10); k = 1;
while max(O) > tol
    b1 = b0 + alpha*O;
    f = g(b1,part)'*g(b1,part);
    while f > f0
        alpha = 0.1*alpha;
        b1 = b0 + alpha*O;
        f = g(b1,part)'*g(b1,part);
    end
    alpha = alpha^0.5;
    b0 = b1;
    g0 = g(b0,part);
    J0 = jacob(b0,part);
    f0 = f;
    fun = [fun; f0];
    O = -(J0'*J0+lambda*eye(3))\(J0'*g0);    
    k = k+1;
end
b0'
b(1,:)
figure; plot(fun)
xlabel('Number of iterations'),ylabel('Function value')

rmse = 0;
for i = 1:3
    rmse = rmse + (b(1,i)-b0(i))^2;
end
rmse

%Part i = 2
part = p2;
b0 = [7;10;15];
g0 = g(b0,part);
J0 = jacob(b0,part);
f0 = g0'*g0; fun = f0;
alpha = 1; lambda = 10^(-10);
O = -(J0'*J0+lambda*eye(3))\(J0'*g0);
tol = 10^(-10); k = 1;
while max(O) > tol
    b1 = b0 + alpha*O;
    f = g(b1,part)'*g(b1,part);
    while f > f0
        alpha = 0.1*alpha;
        b1 = b0 + alpha*O;
        f = g(b1,part)'*g(b1,part);
    end
    alpha = alpha^0.5;
    b0 = b1;
    g0 = g(b0,part);
    J0 = jacob(b0,part);
    f0 = f;
    fun = [fun; f0];
    O = -(J0'*J0+lambda*eye(3))\(J0'*g0);    
    k = k+1;
end
b0'
b(2,:)
figure; plot(fun)
xlabel('Number of iterations'),ylabel('Function value')

rmse = 0;
for i = 1:3
    rmse = rmse + (b(2,i)-b0(i))^2;
end
rmse

%Part i = 3
part = p3;
b0 = [5;12;14];
g0 = g(b0,part);
J0 = jacob(b0,part);
f0 = g0'*g0; fun = f0;
alpha = 1; lambda = 10^(-10);
O = -(J0'*J0+lambda*eye(3))\(J0'*g0);
tol = 10^(-10); k = 1;
while max(O) > tol
    b1 = b0 + alpha*O;
    f = g(b1,part)'*g(b1,part);
    while f > f0
        alpha = 0.1*alpha;
        b1 = b0 + alpha*O;
        f = g(b1,part)'*g(b1,part);
    end
    alpha = alpha^0.5;
    b0 = b1;
    g0 = g(b0,part);
    J0 = jacob(b0,part);
    f0 = f;
    fun = [fun; f0];
    O = -(J0'*J0+lambda*eye(3))\(J0'*g0);    
    k = k+1;
end
b0'
b(3,:)
figure; plot(fun)
xlabel('Number of iterations'),ylabel('Function value')

rmse = 0;
for i = 1:3
    rmse = rmse + (b(3,i)-b0(i))^2;
end
rmse

%Part i = 4
part = p4;
b0 = [3;22;15];
g0 = g(b0,part);
J0 = jacob(b0,part);
f0 = g0'*g0; fun = f0;
alpha = 1; lambda = 10^(-10);
O = -(J0'*J0+lambda*eye(3))\(J0'*g0);
tol = 10^(-10); k = 1;
while max(O) > tol
    b1 = b0 + alpha*O;
    f = g(b1,part)'*g(b1,part);
    while f > f0
        alpha = 0.1*alpha;
        b1 = b0 + alpha*O;
        f = g(b1,part)'*g(b1,part);
    end
    alpha = alpha^0.5;
    b0 = b1;
    g0 = g(b0,part);
    J0 = jacob(b0,part);
    f0 = f;
    fun = [fun; f0];
    O = -(J0'*J0+lambda*eye(3))\(J0'*g0);    
    k = k+1;
end
b0'
b(4,:)
figure; plot(fun)
xlabel('Number of iterations'),ylabel('Function value')

rmse = 0;
for i = 1:3
    rmse = rmse + (b(4,i)-b0(i))^2;
end
rmse

%Part i = 5
part = p5;
b0 = [15;5;10];
g0 = g(b0,part);
J0 = jacob(b0,part);
f0 = g0'*g0; fun = f0;
alpha = 1; lambda = 10^(-10);
O = -(J0'*J0+lambda*eye(3))\(J0'*g0);
tol = 10^(-10); k = 1;
while max(O) > tol
    b1 = b0 + alpha*O;
    f = g(b1,part)'*g(b1,part);
    while f > f0
        alpha = 0.1*alpha;
        b1 = b0 + alpha*O;
        f = g(b1,part)'*g(b1,part);
    end
    alpha = alpha^0.5;
    b0 = b1;
    g0 = g(b0,part);
    J0 = jacob(b0,part);
    f0 = f;
    fun = [fun; f0];
    O = -(J0'*J0+lambda*eye(3))\(J0'*g0);    
    k = k+1;
end
b0'
b(5,:)
figure; plot(fun)
xlabel('Number of iterations'),ylabel('Function value')

rmse = 0;
for i = 1:3
    rmse = rmse + (b(5,i)-b0(i))^2;
end
rmse
