function  displaySuperpixelMatching(I_rgb_A,I_rgb_B,spA,spB,sp_match,spA_center,spB_center)

labelA = getLabel(I_rgb_A,spA);
labelB = getLabel(I_rgb_B,spB);
[~,~,imgMarkupA]=segoutput(im2double(I_rgb_A),double(labelA));
[~,~,imgMarkupB]=segoutput(im2double(I_rgb_B),double(labelB));
[hA,wA,d] = size(imgMarkupA);
[hB,wB,d] = size(imgMarkupB);
imwrite(imgMarkupA,'spA.png');
imwrite(imgMarkupB,'spB.png');
figure,imshow(imgMarkupA);
figure,imshow(imgMarkupB);

% img = imgMarkupA;
% img(1:hB,end+1:end+wB,:) = imgMarkupB;
img = I_rgb_A;
img(1:hB,end+1:end+wB,:) = I_rgb_B;
spA_center = max(floor(spA_center),1);
spB_center = max(floor(spB_center),1);
spB_center(:,2) = spB_center(:,2)+wA;
figure;imshow(img);hold on
% draw center points of superpixels
% scatter(spA_center(:,2),spA_center(:,1),'o');
% scatter(spB_center(:,2),spB_center(:,1),'o');
nr = 100;
r= rand(nr,1); r = unique(ceil(r*size(sp_match,1)));r=r';
% count = 1;
% lr = length(r);
% color_table = lines(lr);
for i = r%1:size(sp_match,1)%100:120%
    for j = 1:size(sp_match,2)
        plot([ spA_center(i,2),spB_center(sp_match(i,j),2) ]...
    ,[spA_center(i,1),spB_center(sp_match(i,j),1) ],...
            '-','LineWidth',2,'MarkerSize',10,...
            'color', rand(1,3));
%         count = mod(count,lr)+1;
    end
end
end

