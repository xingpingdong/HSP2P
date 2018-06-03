function [matchB2A_img,A_orig,B_orig]  = do_lab(fileNameA,fileNameB,showFlag)

if nargin < 3
    showFlag = 0;
end
%% input images
%  fileNameA = 'imgs/Dataset_CSH/Never back down 0727.jpg';%'6_24_s.bmp';%'Saba1.bmp';%
% fileNameB = 'imgs/Dataset_CSH/Never back down 0743.jpg';%'6_20_s.bmp';%'Saba2.bmp';%
% fileNameA = 'imgs/ArchSequence/A_3_02.jpg';%'01a.png'; % 'imgs/co_seg/14_20_s.bmp';%'imgs/dataset/cliffs1_-2.jpg';%'imgs/image_0001.jpg';%'imgs/frame0.jpg';%
% fileNameB = 'imgs/ArchSequence/A_5_01.jpg';%'01b.png'; %'imgs/co_seg/14_14_s.bmp';%'imgs/dataset/cliffs1_0.jpg';'imgs/image_0082.jpg';%'imgs/frame1.jpg';%
% fileNameA = 'imgs/Scenes/FeedingTime/Img1.tif';
% fileNameB = 'imgs/Scenes/FeedingTime/Img2.tif';%HDR_tonemapped.jpg
imgNames = {fileNameA,fileNameB};
nameA = fileNameA(1:end-4); nameB = fileNameB(1:end-4);
%% parameter setup
run('vlfeat-0.9.16/toolbox/vl_setup')
addpath('features');
addpath('dicts');
addpath('others');

alpha0 = 1;
alpha = 5;
is_pixelmatch = 0;
is_pixelmatch_n = 1;
t00=clock;

I_rgb_A = imread(fileNameA);
if strcmp(fileNameA(end-2:end),'tif')
    I_rgb_A=uint8(I_rgb_A/255);
end
I_rgb_B = imread(fileNameB);
if strcmp(fileNameB(end-2:end),'tif')
	I_rgb_B=uint8(I_rgb_B/255);
end
rf = max(max(size(I_rgb_A)))  / 640;
if (rf > 1)
    scale = 1.0/rf;
    disp('source image is too big. resize automatically.')
%     Src = imresize(Src, 1.0/rf);
%     Ref = imresize(Ref, 1.0/rf);
    I_rgb_A = imresize(I_rgb_A,scale);
    I_rgb_B = imresize(I_rgb_B,scale);
end

%% extract SIFT
t0 = clock;
pca_basis = [];
sift_size = 4;

[sift1, bbox1] = ExtractSIFT_WithPadding(I_rgb_A, pca_basis, sift_size);
[sift2, bbox2] = ExtractSIFT_WithPadding(I_rgb_B, pca_basis, sift_size);
% [sift1, bbox1] = ExtractSIFT(I_rgb_A, pca_basis, sift_size);
% [sift2, bbox2] = ExtractSIFT(I_rgb_B, pca_basis, sift_size);
% I_rgb_A = I_rgb_A(bbox1(3):bbox1(4), bbox1(1):bbox1(2), :);
% I_rgb_B = I_rgb_B(bbox2(3):bbox2(4), bbox2(1):bbox2(2), :);
A_orig = I_rgb_A;
B_orig = I_rgb_B;

fprintf('Extract SIFT: %f\n',etime(clock,t0));
%% Calculate the superpixels and features
[hA,wA,dA] = size(I_rgb_A);
[hB,wB,dB] = size(I_rgb_B);
set_options;
opts.seg_method = 'slic';% 'felz';%
opts.slic_region_size =20; % default 20

% I_rgb_A = imresize(I_rgb_A,scale);
lab_A  = colorspace('Lab<-', I_rgb_A);
[ spA,K_A,feaA,feaS_A,labelA ] = cal_sp_feature_sift( lab_A ,fileNameA,sift1,opts );% cal

% I_rgb_B = imresize(I_rgb_B,[hA,wA]);
% I_rgb_B = imresize(I_rgb_B,scale);
lab_B  = colorspace('Lab<-', I_rgb_B);
[ spB,K_B,feaB,feaS_B,labelB] = cal_sp_feature_sift( lab_B ,fileNameB,sift2,opts );

max_iter = 1;
iters = 1;
origin_feaA = feaA;
while iters<=max_iter
    %% Calculate the distances of all superpixels pairs between two images

   
%         t0 = clock;    
        Distance_c = sqdist(feaA',feaB');
        Distance_sift = sqdist(feaS_A',feaS_B');
        Distance = alpha0*Distance_c+alpha*Distance_sift;
%         fprintf('Calculate similarity scores of \n all superpixels pairs: %f\n',etime(clock,t0));
  
    %% Calculate the candidate superpixel set for image A from image B
   
    [~,ind]=min(Distance,[],2);

    nA = length(spA);  nB = length(spB);   
        % center cordinate of each superpixel
    spA_center = zeros(nA,2); spB_center = zeros(nB,2);
    for i = 1:nA
        spA_center(i,:)= mean(spA{i}.pixels,1);
    end
    for i = 1:nB
        spB_center(i,:)= mean(spB{i}.pixels,1);
    end

    final_sp_match = ind(:,1);
        % display the result of superpixel matching
        if showFlag
            displaySuperpixelMatching(I_rgb_A,I_rgb_B,spA,spB,final_sp_match,spA_center,spB_center);
        end
    %% global color mapping
    scores = cal_consistence_sp(feaA,spA_center,spB_center,final_sp_match,spA);
    [~,ind_s] = sort(scores);
    n_cand = ceil(nA*0.5);
    candinate1 = ind_s(1:n_cand);
    candinate2 = final_sp_match(candinate1);
    % shownReliableRegion
%     AA = im2double(I_rgb_A); BB = im2double(I_rgb_B);
%     shownReliableRegion(AA,BB,labelA,labelB,candinate1,candinate2);

    % [ paraMat ] = InterSpline_lab( feaA, feaB, candinate1,candinate2, 1 );
    % feaA = colorspace('RGB<-Lab', feaA);
    % feaB = colorspace('RGB<-Lab', feaB);
    % Cubic spline interpolation
    for channel = 1 : 3 % R, G, B
        eval(['paraMat_' num2str(channel) '  = InterSpline_lab( origin_feaA, feaB, candinate1,candinate2, channel );']);
    end
    newfeaA(:, 1) = ppval(paraMat_1, origin_feaA(:, 1)/100.0)*100;
    newfeaA(:, 2) = ppval(paraMat_2, (origin_feaA(:, 2)+128)/255)*255-128;
    newfeaA(:, 3) = ppval(paraMat_3, (origin_feaA(:, 3)+128)/255)*255-128;
    
    % A = I_rgb_A; B = I_rgb_B;
% A = lab_A; B = lab_B;
% % Test display
% % if 1
% % t_xx = 0 : 0.05 : 1;
% % t_yyr = ppval(paraMat_1, t_xx);
% % t_yyg = ppval(paraMat_2, t_xx);
% % t_yyb = ppval(paraMat_3, t_xx);
% % figure,plot(t_xx, t_yyr, 'r', t_xx, t_yyg, 'g', t_xx, t_yyb, 'b');
% % end
% 
% % Color transfer 
% % A = double(A) / 255;
% newA = A;
% newA(:, :, 1) = ppval(paraMat_1, A(:, :, 1)/100.0)*100;
% newA(:, :, 2) = ppval(paraMat_2, (A(:, :, 2)+128)/255)*255-128;
% newA(:, :, 3) = ppval(paraMat_3, (A(:, :, 3)+128)/255)*255-128;
% 
% % B = double(B) / 255;
% % fprintf('all time: %f\n',etime(clock,t00));
% % figure,imshow([A, newA, B]);
% % imwrite(newA, [fileNameA(1:end-4) '_gm_result.png']);
% % A = A*255; newA = newA*255;
% A  = colorspace('RGB<-Lab', A);
% newA  = colorspace('RGB<-Lab', newA);
% B = colorspace('RGB<-Lab', B);
% figure,imshow([A,newA,B]);
    error = feaB(final_sp_match,:)-newfeaA;
    error = mean(sqrt(sum(error.^2,2)));
    if  error<7
        fprintf('error is: %f\n',error);
        break
    else
        fprintf('error is: %f\n',error);
        feaA = newfeaA;
        alpha = alpha*0.5;
        iters = iters+1;
    end
end

%% 使用映射后的颜色特征 重新计算超像素匹配
        Distance_c = sqdist(feaA',feaB');
%         Distance_sift = sqdist(feaS_A',feaS_B');
        Distance = alpha0*Distance_c+1*alpha*Distance_sift;
        [~,ind]=min(Distance,[],2);
        final_sp_match = ind(:,1);
%% 计算映射后的源图
% figure,imshow([A,newA,B]);
% A = I_rgb_A; B = I_rgb_B;
A = lab_A; B = lab_B;
% Test display
if showFlag
    t_xx = 0 : 0.05 : 1;
    t_yyr = ppval(paraMat_1, t_xx);
    t_yyg = ppval(paraMat_2, t_xx);
    t_yyb = ppval(paraMat_3, t_xx);


    figure;
    hold on
    plot(t_xx, t_yyr, 'r','LineWidth',3);%, t_xx, t_yyg, 'g','LineWidth',4,  t_xx, t_yyb, 'b','LineWidth',4 );
    plot(t_xx, t_yyg, 'g','LineWidth',3);
    plot(t_xx, t_yyb, 'b','LineWidth',3);
    hold off

end

% Color transfer 
% A = double(A) / 255;
newA = A;
newA(:, :, 1) = ppval(paraMat_1, A(:, :, 1)/100.0)*100;
newA(:, :, 2) = ppval(paraMat_2, (A(:, :, 2)+128)/255)*255-128;
newA(:, :, 3) = ppval(paraMat_3, (A(:, :, 3)+128)/255)*255-128;

% B = double(B) / 255;
fprintf('all time: %f\n',etime(clock,t00));
% figure,imshow([A, newA, B]);
% % imwrite(newA, [fileNameA(1:end-4) '_gm_result.png']);
% % A = A*255; newA = newA*255;
A  = colorspace('RGB<-Lab', A);
newA  = colorspace('RGB<-Lab', newA);
B = colorspace('RGB<-Lab', B);
if showFlag
    figure,imshow([newA]);
    % figure,imshow([A, newA, B]);
    imwrite(newA, [fileNameA(1:end-4) '_map.png']);
end
% %% Pixel-wise match (minization of distance)
matchB2A = zeros(hA,wA,2);
matchB2A_img = zeros(hA,wA,dA);

if is_pixelmatch
tic
I_rgb_A = newA*255;
%只使用图A超像素的最近邻 作为候选超像素

for i = 1:nA
    pixels1 = spA{i}.pixels;
    pixels2 = spB{final_sp_match(i)}.pixels;
    n1 = size(pixels1,1); n2 = size(pixels2,1);
    vector1 = zeros(n1,dA); vector2 = zeros(n2,dB);
    t0 = clock;
    for j = 1:n1
        vector1(j,:) = I_rgb_A(pixels1(j,1),pixels1(j,2),:);
    end
    for j = 1:n2
        vector2(j,:) = I_rgb_B(pixels2(j,1),pixels2(j,2),:);
    end
    
    if i == 1
    fprintf('Calculate the distance matrix \nof pixels in each superpixel: %f\n',etime(clock,t0));
    end
    dist_v1v2 = sqdist(double(vector1)',double(vector2)');
    
    [~,ind_min] = min(dist_v1v2,[],2);
    
%     ind_min=vl_kdtreequery(vl_kdtreebuild(double(vector2)'),double(vector2)',double(vector1)','NUMNEIGHBORS',1,'MAXNUMCOMPARISONS',2);
%     ind_min = ind_min';
    for j = 1:size(pixels1)
        sub1 = pixels1(j,:);
        sub2 = pixels2(ind_min(j),:);
        matchB2A(sub1(1),sub1(2),:) = sub2;
        matchB2A_img(sub1(1),sub1(2),:) = I_rgb_B(sub2(1),sub2(2),:);
    end
end
toc
if showFlag
     figure,imshow(uint8(matchB2A_img));

     resultpath = [fileNameA(1:end-4),'_pm1.png'];
    imwrite(uint8(matchB2A_img),resultpath);
end

end

% 使用图A的超像素及其相邻的超像素 它们所对应的图B中的最近邻 作为候选的像素集
if is_pixelmatch_n
tic
I_rgb_A = newA*255;
for i = 1:nA
    
    if i==1 
%         t1 = clock;   
        tic
    end 
    sp1 = spA{i};
    pixels1 = sp1.pixels;  
    n1 = size(pixels1,1);% vector1 = zeros(n1,dA);vector1s = zeros(n1,128);
    for j = 1:n1
        vector1(j,:) = I_rgb_A(pixels1(j,1),pixels1(j,2),:);
%         vector1s(j,:) = sift1(pixels1(j,1),pixels1(j,2),:);
    end
    
    neighbors = sp1.neighbors;
    neighbors = [i,neighbors];
    candidate = final_sp_match(neighbors');
    
    count = 1; vector2=[]; pixels2 = [];vector2s=[];
    for k = 1:length(candidate)
        pixels_t = spB{candidate(k)}.pixels;
        pixels2 = [pixels2;pixels_t];
        n2 = size(pixels_t,1);
        for j = 1:n2
            vector2(count,:) = I_rgb_B(pixels_t(j,1),pixels_t(j,2),:);
%             vector2s(count,:) = sift2(pixels_t(j,1),pixels_t(j,2),:);
            count = count+1;
        end
    end   
 
    if i == 1
        toc
%         fprintf('Calculate the distance matrix \nof pixels in each superpixel: %f\n',etime(clock,t1));
    end
    if i==1
%         t2=clock;
        tic
    end
%     dist_v1v2c = sqdist(double(vector1)',double(vector2)');
%     dist_v1v2s = sqdist(double(vector1s)',double(vector2s)');
%     dist_v1v2 = alpha0*dist_v1v2c+alpha*dist_v1v2s;
    dist_v1v2 = sqdist(double(vector1)',double(vector2)');
    [~,ind_min] = min(dist_v1v2,[],2);
%     ind_min=vl_kdtreequery(vl_kdtreebuild(double(vector2)'),double(vector2)',double(vector1)','NUMNEIGHBORS',1,'MAXNUMCOMPARISONS',2);
%     ind_min = ind_min';
    if i == 1
        toc
%         fprintf('Calculate the distance matrix \nof pixels in each superpixel: %f\n',etime(clock,t2));
    end
    if i ==1
%         t3=clock; 
        tic
    end
    for j = 1:size(pixels1)
        sub1 = pixels1(j,:);
        sub2 = pixels2(ind_min(j),:);
        matchB2A(sub1(1),sub1(2),:) = sub2;
        matchB2A_img(sub1(1),sub1(2),:) = I_rgb_B(sub2(1),sub2(2),:);
    end
    if i == 1
        toc
%         fprintf('Calculate the distance matrix \nof pixels in each superpixel: %f\n',etime(clock,t3));
    end
end

toc
if showFlag
     figure,imshow(uint8(matchB2A_img));
     resultpath = [fileNameA(1:end-4),'_pm.png'];
    imwrite(uint8(matchB2A_img),resultpath);
end

end

A = A;
im_r = matchB2A_img/255.0;
    error = cal_error(A,im_r);
        fprintf('The mean error is: %f\n',error);
        matchB2A_img = uint8(matchB2A_img);
end
% 
%  error = double(I_rgb_A)-matchB2A_img;
%  error = reshape(error,hA*wA,dA);
%   error = sqrt(sum((sum(error.^2,2))))/(hA*wA);
%  fprintf('The mean error is: %f\n',error);
%  
%  fprintf('all time: %f\n',etime(clock,t00));
%  
% %  addpath('CSH_code_v2');
% %   addpath('CSH_code_v2/matlab');
% %   width = 8; % Default patch width value
% % PlotExampleResults(I_rgb_A,I_rgb_B,matchB2A ,width,1,[],'default CSH');
% % XX  = reshape(X,nA,k); %XX = XX';
% 
% %% color tansfer
% addpath('colorTansfer');
% [yA,xA] = meshgrid(1:wA,1:hA);
% coorA(:,:,1)=xA; coorA(:,:,2)=yA;
% coorA = reshape(coorA,hA*wA,2); coorA =coorA';
% coorB = reshape(matchB2A,hA*wA,2); coorB = coorB';
% coorMatches(1,:) = coorA(2,:);
% coorMatches(2,:) = coorA(1,:);
% coorMatches(3,:) = coorB(2,:);
% coorMatches(4,:) = coorB(1,:);
% A = I_rgb_A; B = I_rgb_B;
% paraMat_1  = InterSpline( A, B, coorMatches, 1 );
% 
% %% Cubic spline interpolation
% for channel = 1 : 3 % R, G, B
%     eval(['paraMat_' num2str(channel) '  = InterSpline( A, B, coorMatches, channel );']);
% end
% 
% % Test display
% if 1
% t_xx = 0 : 0.05 : 1;
% t_yyr = ppval(paraMat_1, t_xx);
% t_yyg = ppval(paraMat_2, t_xx);
% t_yyb = ppval(paraMat_3, t_xx);
% figure,plot(t_xx, t_yyr, 'r', t_xx, t_yyg, 'g', t_xx, t_yyb, 'b');
% end
% 
% %% Color transfer 
% A = double(A) / 255;
% newA = A;
% newA(:, :, 1) = ppval(paraMat_1, A(:, :, 1));
% newA(:, :, 2) = ppval(paraMat_2, A(:, :, 2));
% newA(:, :, 3) = ppval(paraMat_3, A(:, :, 3));
% 
% B = double(B) / 255;
% figure,imshow([A, newA, B]);
% 
% imwrite(newA, [fileNameA(1:end-4) '.png']);
%% pixel-wise match between superpixels

