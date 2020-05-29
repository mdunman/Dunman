load('heatT.mat')
size(T1)

%%% CP DECOMPOSITION
%T1
rng(1234);
TM1 = tenmat(T1,1);
err = zeros(10,1);
for i = 1:10
    P = cp_als(T1,i);
    P1 = tenmat(P,1);
    dif = tensor(TM1-P1);
    err(i) = innerprod(dif,dif);
end
AIC_C1 =  2*err + (2*[1:10])';
[min1,wmin1] = min(AIC_C1);
plot(AIC_C1); hold on;
plot(wmin1,min1,'o','MarkerSize',10); xlabel('R'), ylabel('AIC')

P1 = cp_als(T1,wmin1);
subplot(3,2,1); plot(P1.U{3});
legend('1','2','3','4','5'), xlabel('time')
for i = 1:wmin1
    XY = kron(P1.U{1}(:,i),P1.U{2}(:,i)')*P1.lambda(i);
    subplot(3,2,i+1);
    ScaledXY = XY*(wmin1^2);
    image(ScaledXY);
    xlabel('x'), ylabel('y'), colormap hot
end


%T2
rng(1234);
TM2 = tenmat(T2,1);
err = zeros(10,1);
for i = 1:10
    P = cp_als(T2,i);
    P1 = tenmat(P,1);
    dif = tensor(TM2-P1);
    err(i) = innerprod(dif,dif);
end
AIC_C2 =  2*err + (2*[1:10])';
[min2,wmin2] = min(AIC_C2);
plot(AIC_C2); hold on;
plot(wmin2,min2,'o','MarkerSize',10); xlabel('R'), ylabel('AIC')

P2 = cp_als(T2,wmin2);
subplot(3,2,1); plot(P2.U{3});
legend('1','2','3','4'), xlabel('time')
for i = 1:wmin2
    XY = kron(P2.U{1}(:,i),P2.U{2}(:,i)')*P2.lambda(i);
    subplot(3,2,i+1);
    ScaledXY = XY*(wmin2^2);
    image(ScaledXY);
    xlabel('x'), ylabel('y'), colormap hot
end


%T3
rng(1234);
TM3 = tenmat(T3,1);
err = zeros(10,1);
for i = 1:10
    P = cp_als(T3,i);
    P1 = tenmat(P,1);
    dif = tensor(TM3-P1);
    err(i) = innerprod(dif,dif);
end
AIC_C3 =  2*err + (2*[1:10])';
[min3,wmin3] = min(AIC_C3);
plot(AIC_C3); hold on;
plot(wmin3,min3,'o','MarkerSize',10); xlabel('R'), ylabel('AIC')

P3 = cp_als(T3,wmin3);
subplot(3,2,1); plot(P3.U{3}); 
legend('1','2','3','4','5'), xlabel('time');
for i = 1:wmin3
    XY = kron(P3.U{1}(:,i),P3.U{2}(:,i)')*P3.lambda(i);
    subplot(3,2,i+1);
    ScaledXY = XY*(wmin3^2);
    image(ScaledXY);
    xlabel('x'), ylabel('y'), colormap hot
end



%%% Tucker Decomposition
%T1
TM1 = tenmat(T1,1);
err = zeros(4,4,4);
AIC_T1 = zeros(4,4,4);
for i = 1:4
    for j = 1:4
        for k = 1:4
            rng(1234)
            Tk = tucker_als(T1,[i,j,k]);
            Tk1 = tenmat(Tk,1);
            dif = tensor(TM1-Tk1);
            err(i,j,k) = innerprod(dif,dif);
            AIC_T1(i,j,k) = 2*err(i,j,k) + 2*(i+j+k);
        end
    end    
end
AIC_T1
AIC_T1(3,3,3)
Tk_T1 = tucker_als(T1,[3,3,3]);
XY1_T1 = kron(Tk_T1.U{1}(:,1),Tk_T1.U{2}(:,1)');
XY2_T1 = kron(Tk_T1.U{1}(:,2),Tk_T1.U{2}(:,2)');
XY3_T1 = kron(Tk_T1.U{1}(:,3),Tk_T1.U{2}(:,3)');
figure; suptitle('T1'); 
subplot(2,2,1), image(XY1_T1*512), colormap hot, xlabel('x'), ylabel('y'), title('XY1');
subplot(2,2,2), image(XY2_T1*512), colormap hot, xlabel('x'), ylabel('y'), title('XY2');
subplot(2,2,3), image(XY3_T1*512), colormap hot, xlabel('x'), ylabel('y'), title('XY3');
subplot(2,2,4), plot(Tk_T1.U{3}), legend('1','2','3'), xlabel('time')


%T2
TM2 = tenmat(T2,1);
err = zeros(4,4,4);
AIC_T2 = zeros(4,4,4);
for i = 1:4
    for j = 1:4
        for k = 1:4
            Tk = tucker_als(T2,[i,j,k]);
            Tk1 = tenmat(Tk,1);
            dif = tensor(TM2-Tk1);
            err(i,j,k) = innerprod(dif,dif);
            AIC_T2(i,j,k) = 2*err(i,j,k) + 2*(i+j+k);
        end
    end    
end
AIC_T2
AIC_T2(3,3,3)
Tk_T2 = tucker_als(T2,[3,3,3]);
XY1_T2 = kron(Tk_T2.U{1}(:,1),Tk_T2.U{2}(:,1)');
XY2_T2 = kron(Tk_T2.U{1}(:,2),Tk_T2.U{2}(:,2)');
XY3_T2 = kron(Tk_T2.U{1}(:,3),Tk_T2.U{2}(:,3)');
figure; suptitle('T2');
subplot(2,2,1), image(XY1_T2*512), colormap hot, xlabel('x'), ylabel('y'), title('XY1')
subplot(2,2,2), image(XY2_T2*512), colormap hot, xlabel('x'), ylabel('y'), title('XY2')
subplot(2,2,3), image(XY3_T2*512), colormap hot, xlabel('x'), ylabel('y'), title('XY3')
subplot(2,2,4), plot(Tk_T2.U{3}), legend('1','2','3'), xlabel('time')


%T3
TM3 = tenmat(T3,1);
err = zeros(4,4,4);
AIC_T3 = zeros(4,4,4);
for i = 1:4
    for j = 1:4
        for k = 1:4
            Tk = tucker_als(T3,[i,j,k]);
            Tk1 = tenmat(Tk,1);
            dif = tensor(TM3-Tk1);
            err(i,j,k) = innerprod(dif,dif);
            AIC_T3(i,j,k) = 2*err(i,j,k) + 2*(i+j+k);
        end
    end    
end
AIC_T3
AIC_T3(3,3,3)
Tk_T3 = tucker_als(T3,[3,3,3]);
XY1_T3 = kron(Tk_T3.U{1}(:,1),Tk_T3.U{2}(:,1)');
XY2_T3 = kron(Tk_T3.U{1}(:,2),Tk_T3.U{2}(:,2)');
XY3_T3 = kron(Tk_T3.U{1}(:,3),Tk_T3.U{2}(:,3)');
figure; suptitle('T3');
subplot(2,2,1), image(XY1_T3*512), colormap hot, xlabel('x'), ylabel('y'), title('XY1')
subplot(2,2,2), image(XY2_T3*512), colormap hot, xlabel('x'), ylabel('y'), title('XY2')
subplot(2,2,3), image(XY3_T3*512), colormap hot, xlabel('x'), ylabel('y'), title('XY3')
subplot(2,2,4), plot(Tk_T3.U{3}), legend('1','2','3'), xlabel('time')


subplot(2,2,1), image(tenmat(Tk_T1.core,1).data*512), title('T1');
subplot(2,2,2), image(tenmat(Tk_T2.core,1).data*512), title('T2');
subplot(2,2,3), image(tenmat(Tk_T3.core,1).data*512), title('T3');
