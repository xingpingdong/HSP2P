function shownReliableRegion(A,B,labelA,labelB,candinate1,candinate2)
% 显示候选的区域
n = length(candinate1);
[h1,w1,d1] = size(A);
[h2,w2,d2] = size(B);
A = reshape(A,h1*w1,d1);
B = reshape(B,h2*w2,d2);
labelA = reshape(labelA,h1*w1,1);
labelB = reshape(labelB,h2*w2,1);
rA = A;
rB = B;
mask1 = ones(h1*w1,1);
mask2 = ones(h2*w2,1);
for i = 1:n
    ind = labelA == candinate1(i);
    mask1(ind) = 0;
    ind = labelB == candinate2(i);
    mask2(ind) = 0;
end
mask1 = logical(mask1);
mask2 = logical(mask2);
rA(mask1,:) = A(mask1,:)*0.5+0.5;
rB(mask2,:) = B(mask2,:)*0.5+0.5;
rA = reshape(rA,[h1,w1,d1]);
rB = reshape(rB,[h2,w2,d2]);
figure,imshow(rA);
figure,imshow(rB);
imwrite(rA,'maskA.png');
imwrite(rB,'maskB.png');
end