function g = grad(mu, p, y)
    m = size(p,1);
    n = size(p,2);
    g = zeros(n,1);
    for i = 1:n
        sumj = 0;
        for j = 1:m
            sum = 0;
            for l = 1:n
                sum = sum + p(j,l)*mu(l);
            end
            sumj = sumj - p(j,i) + y(j)*p(j,i)/sum;
        end
        g(i) = sumj;
    end
end
