%%%%%%--- Problem 2 ---%%%%%
% part a)
I = imread('horse1.jpg');
imshow(I);

% part b)
size(I)
third = imresize(I,1/3);
size(third)
figure,imshow(third);

% part c)
GRAY = rgb2gray(I);
BW = im2bw(I,0.5);
subplot(1,2,1);imshow(GRAY);subplot(1,2,2);imshow(BW);

% part d) - WHAT OBSERVATIONS?
imhist(GRAY);

% part e) - EXPLAIN PURPOSES
D = double(GRAY);
%linear
L=2^8;
lin = ((L-1) - GRAY);
imshow(lin);

%log; c=20 & c=40
lg20=uint8(20*log(D+1));
lg40=uint8(40*log(D+1));
subplot(1,2,1);imshow(lg20);subplot(1,2,2);imshow(lg40);

%thresholding of 100 & 150
t100 = GRAY;
t150 = GRAY;
size(GRAY)
for i = 1:563
  for j = 1:1000
    if GRAY(i,j) > 100
      t100(i,j) = 255;
    else
      t100(i,j) = 0;
    end
  end
end

for i = 1:563
  for j = 1:1000
    if GRAY(i,j) > 150
      t150(i,j) = 255;
    else
      t150(i,j) = 0;
    end
  end
end
subplot(1,2,1);imshow(im2bw(t100));subplot(1,2,2);imshow(im2bw(t150));

%histogram shift; L=25, U=225, shift=50
L = 25;
U = 225;
s = 50;
sh50 = GRAY;
for i = 1:563
  for j = 1:1000
    if (GRAY(i,j) > (U-s))
      sh50(i,j) = U;
    elseif (GRAY(i,j) <= (L-s))
      sh50(i,j) = L;
    else
      sh50(i,j) = GRAY(i,j) + s;
    end
  end
end
subplot(1,2,1);imshow(GRAY);subplot(1,2,2);imshow(sh50);

%histogram stretch; lambda=200
lamb = 200;
st200 = uint8((D-min(min(D)))/(max(max(D))-min(min(D)))*lamb);
max(GRAY(:)) - min(GRAY(:))
max(st200(:)) - min(st200(:))
figure,imshow(st200);

% part f)
size(GRAY)
Noisy = uint8(double(GRAY)+normrnd(0,100,563,1000));
Kf = ones(3,3)/9;  
Yf = imfilter(Noisy,Kf);
subplot(1,2,1),imshow(Noisy),subplot(1,2,2),imshow(Yf)

% part g)
Kg = [-1 -1 -1; -1 9 -1; -1 -1 -1];
Yg = imfilter(I,Kg);
subplot(1,2,1),imshow(I),subplot(1,2,2),imshow(Yg)

% part h)
for i = 2:5
    Thresh=multithresh(GRAY,i)
    subplot(2,2,i-1);
    imshow(imquantize(GRAY,Thresh),[])
    title(i)
end

% part i)
X=reshape(I,size(I,1)*size(I,2),size(I,3));
K =[2,3,4,5];
for i = 1:4
    [L,Centers] = kmeans(double(X),K(i));
    Y = reshape(L,size(I,1),size(I,2));
    B = labeloverlay(I,Y);
    subplot(2,2,i)
    imshow(B)
    title(K(i))
end

% part j)
b = double(GRAY);
subplot(2,2,1); edge(b,'prewitt',0.5); title('Prewitt - 0.5');
subplot(2,2,2); edge(b,'prewitt',1); title('Prewitt - 1');
subplot(2,2,3); edge(b,'prewitt',5); title('Prewitt - 5');
subplot(2,2,4); edge(b,'prewitt',10); title('Prewitt - 10');

subplot(2,2,1); edge(b,'sobel',0.5); title('Sobel - 0.5');
subplot(2,2,2); edge(b,'sobel',1); title('Sobel - 1');
subplot(2,2,3); edge(b,'sobel',5); title('Sobel - 5');
subplot(2,2,4); edge(b,'sobel',10); title('Sobel - 10');


%%%%%%--- Problem 3 ---%%%%%
% part a)
I = rgb2gray(imread('horse1.jpg'));
gau = [1 4 7 10 7 4 1; 
       4 12 26 33 26 12 4; 
       7 26 55 71 55 26 7;
       10 33 71 91 71 33 10;
       7 26 55 71 55 26 7;
       4 12 26 33 26 12 4;
       1 4 7 10 7 4 1] / 1115;
S = imfilter(I,gau);
figure,imshow(S)

% part b)
S = double(S);
G = zeros(562,999);
theta = zeros(562,999);
for i = 1:562
    for j = 1:999
        dx = (1/2)*( S(i+1,j) - S(i,j) + S(i+1,j+1) - S(i,j+1) );
        dy = (1/2)*( S(i,j+1) - S(i,j) + S(i+1,j+1) - S(i+1,j) );
        G(i,j) = sqrt( dx^2 + dy^2 );
        theta(i,j) = atan( dx / dy );
    end
end

G(1,1)
G(100,100)
G(250,250)
theta(1,1)
theta(100,100)
theta(250,250)

% part c)
phi = zeros(562,999);
for i = 2:561
    for j = 2:998
        if (theta(i,j) > -pi/8) && (theta(i,j) <= pi/8)
            if (G(i,j) > G(i,j-1)) && (G(i,j) > G(i,j+1))
                phi(i,j) = G(i,j);
            end
        elseif (theta(i,j) > pi/8) && (theta(i,j) <= 3*pi/8)
            if (G(i,j) > G(i+1,j-1)) && (G(i,j) > G(i-1,j+1))
                phi(i,j) = G(i,j);
            end
        elseif (theta(i,j) > -3*pi/8) && (theta(i,j) <= -pi/8)
            if (G(i,j) > G(i-1,j-1)) && (G(i,j) > G(i+1,j+1))
                phi(i,j) = G(i,j);
            end
        elseif (theta(i,j) > 3*pi/8) && (theta(i,j) <= pi/2)
            if (G(i,j) > G(i-1,j)) && (G(i,j) > G(i+1,j))
                phi(i,j) = G(i,j);
            end 
        elseif (theta(i,j) >= -pi/2) && (theta(i,j) <= -3*pi/8)
            if (G(i,j) > G(i-1,j)) && (G(i,j) > G(i+1,j))
                phi(i,j) = G(i,j);
            end
        end 
    end
end

phi(50,50)
phi(100,100)
phi(250,250)

figure,imagesc(phi)

% part d)
t1 = 3; t2 = 8;
E = zeros(562,999);
count = 1;
while count ~= 0
    count = 0;
    for i = 2:562
        for j = 2:999
            if (phi(i,j) >= t2) && (E(i,j)==0)
                E(i,j) = 1;
                count = count + 1;
            elseif (phi(i,j) >= t1) && (E(i,j)==0)
                if E(i-1,j-1) == 1
                    E(i,j) = 1;
                    count = count + 1;
                elseif E(i-1,j) == 1
                    E(i,j) = 1;
                    count = count + 1;
                elseif E(i-1,j+1) == 1
                    E(i,j) = 1;
                    count = count + 1;
                elseif E(i,j-1) == 1
                    E(i,j) = 1;
                    count = count + 1;
                elseif E(i,j+1) == 1
                    E(i,j) = 1;
                    count = count + 1;
                elseif E(i+1,j-1) == 1
                    E(i,j) = 1;
                    count = count + 1;
                elseif E(i+1,j) == 1
                    E(i,j) = 1;
                    count = count + 1;
                elseif E(i+1,j+1) == 1
                    E(i,j) = 1;
                    count = count + 1;
                end
            end
        end
    end
end
figure,imagesc(E)
