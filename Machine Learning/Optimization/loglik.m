function l = loglik(mu, p, y)
    m = size(p,1);
    n = size(p,2);
    sum = 0;
    for j = 1:m
        a = 0;
        for i = 1:n
            a = a + p(j,i)*mu(i);
        end
        sum = sum - a + y(j)*log(a);
    end
    l = sum;
end