%%%%% Problem 1 - e %%%%%
data = readmatrix('RealEstate-1.csv');
data(1:5,2:5)
y = data(:,1); m = size(data,1); n = size(data,2);
Bed0 = zeros(m,1); Bed1 = zeros(m,1); Bed2 = zeros(m,1);
Bed3 = zeros(m,1); Bed4 = zeros(m,1);
for i = 1:m
    if data(i,2) == 0
        Bed0(i) = 1;
    elseif data(i,2) == 1
        Bed1(i) = 1;
    elseif data(i,2) == 2
        Bed2(i) = 1;
    elseif data(i,2) == 3
        Bed3(i) = 1;
    else
        Bed4(i) = 1;
    end
end
Bath1 = zeros(m,1); Bath2 = zeros(m,1);
Bath3 = zeros(m,1); Bath4 = zeros(m,1);
for i = 1:m
    if data(i,3) == 1
        Bath1(i) = 1;
    elseif data(i,3) == 2
        Bath2(i) = 1;
    elseif data(i,3) == 3
        Bath3(i) = 1;
    else
        Bath4(i) = 1;
    end
end
data(:,4) = normalize(data(:,4),'range');
Status1 = zeros(m,1); Status2 = zeros(m,1); Status3 = zeros(m,1);
for i = 1:m
    if data(i,5) == 1
        Status1(i) = 1;
    elseif data(i,5) == 2
        Status2(i) = 1;
    else
        Status3(i) = 1;
    end
end
newdata=[Bed0,Bed1,Bed2,Bed3,Bed4,Bath1,Bath2,Bath3,Bath4,...
    data(:,4),Status1,Status2,Status3];
newdata(1:5,:)

lambda = 0.012; t = 0.001; iterations = 300;
L = zeros(13,iterations); a = zeros(13,iterations);
theta = zeros(13,iterations); theta(:,1) = 10000*ones(13,1);
for k = 2:iterations
    theta_c = theta(:, k-1);
    for j = 1:4
        if j == 1
            Z = newdata(:,1:5);
            d = zeros(m,1);
            for q = 6:13
                d = d + newdata(:,q)*theta_c(q);
            end
            r = y - d;
            a(1:5,k) = theta_c(1:5) - t*(-Z'*(r-Z*theta_c(1:5)));
            if t*lambda < norm(a(1:5,k),2)
                theta(1:5,k) = (1-t*lambda/norm(a(1:5,k),2))*a(1:5,k);
            else
                theta(1:5,k) = 0;
            end
            L(1:5,k) = 1/2*(r-Z*theta(1:5,k))'*(r-Z*theta(1:5,k))+lambda*norm(theta(1:5,k),2);
        elseif j == 2
            Z = newdata(:,6:9);
            d = zeros(m,1);
            for q = 1:5
                d = d + newdata(:,q)*theta_c(q);
            end
            for q = 10:13
                d = d + newdata(:,q)*theta_c(q);
            end
            r = y - d;
            a(6:9,k) = theta_c(6:9) - t*(-Z'*(r-Z*theta_c(6:9)));
            if t*lambda < norm(a(6:9,k),2)
                theta(6:9,k) = (1-t*lambda/norm(a(6:9,k),2))*a(6:9,k);
            else
                theta(6:9,k) = 0;
            end
            L(6:9,k) = 1/2*(r-Z*theta(6:9,k))'*(r-Z*theta(6:9,k))+lambda*norm(theta(6:9,k),2);
        elseif j == 3
            Z = newdata(:,10);
            d = zeros(m,1);
            for q = 1:9
                d = d + newdata(:,q)*theta_c(q);
            end
            for q = 11:13
                d = d + newdata(:,q)*theta_c(q);
            end
            r = y - d;
            a(10,k) = theta_c(10) - t*(-Z'*(r-Z*theta_c(10)));
            if t*lambda < a(10,k)
                theta(10,k) = a(10,k) - t*lambda;
            elseif t*lambda > a(10,k)
                theta(10,k) = a(10,k) + t*lambda;
            else
                theta(10,k) = 0;
            end
            L(10,k) = 1/2*(r-Z*theta(10,k))'*(r-Z*theta(10,k))+lambda*norm(theta(10,k),1);
        else
            Z = newdata(:,11:13);
            d = zeros(m,1);
            for q = 1:10
                d = d + newdata(:,q)*theta_c(q);
            end
            r = y - d;
            a(11:13,k) = theta_c(11:13) - t*(-Z'*(r-Z*theta_c(11:13)));
            if t*lambda < norm(a(11:13,k),2)
                theta(11:13,k) = (1-t*lambda/norm(a(11:13,k),2))*a(11:13,k);
            else
                theta(11:13,k) = 0;
            end
            L(11:13,k) = 1/2*(r-Z*theta(11:13,k))'*(r-Z*theta(11:13,k))+lambda*norm(theta(11:13,k),2);
        end
    end
end
plot(L(:,2:k)'), ylabel('Objective function'); xlabel('Iteration number')
theta(:,iterations)'


%%%%% Problem 2 - d %%%%%
load('img.pb3.mat');
load('MRI.mat');
iterations = 100;
theta = zeros(961,1); ub1 = zeros(961,1);
ub2 = zeros(961,1); ub3 = zeros(961,1);
ub_bar = (ub1+ub2+ub3)/3;
lambda = 1; rho = 0.5;
L = zeros(iterations,1);
for i = 1:iterations
    b1 = (X1'*X1+rho*eye(961))\(X1'*y1+rho*(theta-ub1));
    b2 = (X2'*X2+rho*eye(961))\(X2'*y2+rho*(theta-ub2));
    b3 = (X3'*X3+rho*eye(961))\(X3'*y3+rho*(theta-ub3));
    b_bar = (b1+b2+b3)/3;
    %b_bar = mean(b_bar,'all');
    for j = 1:961
        if b_bar(j) + ub_bar(j) > (lambda/(rho*3))
            theta(j) = b_bar(j) + ub_bar(j) - (lambda/(rho*3));
        elseif b_bar(j) + ub_bar(j) < (lambda/(rho*3))
            theta(j) = b_bar(j) + ub_bar(j) + (lambda/(rho*3));
        else
            theta(j) = 0;
        end
    end
    ub1 = ub1 + (b1 - theta);
    ub2 = ub2 + (b2 - theta);
    ub3 = ub3 + (b3 - theta);
    ub_bar = (ub1+ub2+ub3)/3;
    %ub_bar = mean(ub_bar,'all');
    a1 = 0.5*b1'*X1'*X1*b1 - y1'*X1*b1 + rho/2*(b1-theta+ub1)'*(b1-theta+ub1);
    a2 = 0.5*b2'*X2'*X2*b2 - y2'*X2*b2 + rho/2*(b2-theta+ub2)'*(b2-theta+ub2);
    a3 = 0.5*b3'*X3'*X3*b3 - y3'*X3*b3 + rho/2*(b3-theta+ub3)'*(b3-theta+ub3);
    L(i) = a1 + a2 + a3 + lambda*norm(theta,1);
end
plot(L); ylabel('Objective function'); xlabel('Iteration number')
figure; subplot(2,2,1), imagesc(img), title('Original')
subplot(2,2,2), imagesc(vec2mat(b1,31)), title('b1')
subplot(2,2,3), imagesc(vec2mat(b2,31)), title('b2')
subplot(2,2,4), imagesc(vec2mat(b3,31)), title('b3')
b_final = (b1+b2+b3)/3;
figure; subplot(1,2,1), imagesc(img), title('Original')
subplot(1,2,2), imagesc(vec2mat(b_final,31)), title('ADMM')
