function j = jacob(b0,leak)
    n = size(leak,1);
    sum1 = 0;
    sum2 = 0;
    sum3 = 0;
    for i = 1:n      
        sum1 = sum1 - 1 + exp(-2*b0(3)*leak(i,2));
        sum2 = sum2 - leak(i,3)*(1 - exp(-2*b0(3)*leak(i,2)));
        sum3 = sum3 - 2*leak(i,2)*exp(-2*b0(3)*leak(i,2))*(b0(1)+b0(2)*leak(i,3));
    end
    j = [sum1, sum2, sum3];
end