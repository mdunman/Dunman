%%%%%----- Problem 1 -----%%%%%
% a)
A = randn(256,512);
A(1:5,1:5)

%b)
N = 20; s = [26:45];
err = zeros(N,1);
for i = 1:N
    x = zeros(512,1);
    x(randi(512,s(i),1)) = randn(1,s(i),1);
    y = (A*x);
    x0 = A'*y;
    xp = l1eq_pd(x0,A,[],y,10e-4);
    err(i) = sum(abs(xp-x))/sum(abs(x));
end
err
good = find(err<=10e-4)
s(max(good))
figure; plot(s,err); ylim([-1, 5]); yline(10e-4);


%%%%%----- Problem 2 -----%%%%%
%b
load data-1.mat
lambda = 0.5; gamma = 0.5;
cvx_startup
cvx_setup
cvx_begin
variables b(8,1)
variables ba(40,1)
minimize((Y-B*b-Ba*ba)'*(Y-B*b-Ba*ba)+lambda*b'*Omega*b+gamma*norm(ba,1))
cvx_end
b
ba(1:10)
figure;
subplot(2,2,1); plot(Y), title('Y') 
subplot(2,2,2); plot(B*b), title('Smooth')
subplot(2,2,3); plot(Ba*ba), title('Anomaly')
subplot(2,2,4); plot(Y-B*b-Ba*ba), title('Noise')


%%%%%----- Problem 3 -----%%%%%
%a
n = 50; r = 2;
A = randn(n);
[U, S, V] = svd(A);
s = diag(S); 
s(r+1:end) = 0; 
S = [diag(s)];
X = U * S * V';
X0 = X;

A = [rand(n)>=0.5];
X(A) = 0;
m = sum(sum(A==0));
m/2500

%b
Y = zeros(n);
delta = n^2/m;
tau = 250;
vec = zeros(500,1);
for i = 1:500
    [U, S, V] = svd(Y);
    S_t = (S-tau);
    S_t(S_t<0) = 0;
    Z = U*S_t*V';
    P = X-Z;
    P(A) = 0;
    Y0 = Y;
    Y = Y0 + delta*P;
    vec(i) = sum(sum((Y-Y0).^2));
    err(i)=sum(sum((X0-Z).^2))/sum(sum((X0).^2));
end
figure;plot(vec); title('Vec')
Ar = reshape(A, n^2,1);
Xr = reshape(X0, n^2,1);Xr=Xr(Ar);
Xr(1:5,1:5)
Zr = reshape(Z, n^2,1);Zr=Zr(Ar);
Zr
subplot(2,1,1);plot(Xr);hold on; plot(Zr,'r'); title('Z')
subplot(2,1,2);plot(Xr-Zr); title('X-Z')
figure;
subplot(1,2,1); imagesc(X0), title('X0')
subplot(1,2,2); imagesc(Z), title('Z')

%c
err(500)
figure; plot(1:500,err); title('Error')


%%%%%----- Problem 4 -----%%%%%
load peaks.mat

%a
n = size(Y,1);
y = reshape(Y,n^2,1);
lambda = 0.05; c = 0.05;
K = exp(-power(dist(linspace(0,1,n)),2)/(2*c));
K = kron(K,K);
size(K)

%b
y_hat = K *((K + lambda*eye(n^2)) \ y);
y_hat = reshape(y_hat,[100,100]);
y_hat(1:5,1:5)

%c
figure;
subplot(1,2,1); imagesc(Y); title('Original')
subplot(1,2,2); imagesc(y_hat); title('Smooth')

diff = y_hat - Y;
std2(diff)
