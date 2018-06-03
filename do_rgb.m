function [matchB2A_img,A_orig,B_orig]  = do_rgb(fileNameA,fileNameB,showFlag)

if nargin < 3
    showFlag = 0;
end
%% input images
%  fileNameA = 'imgs/Dataset_CSH/Avatar FULL 1080p 00907.jpg';%'6_24_s.bmp';%'Saba1.bmp';%
% fileNameB = 'imgs/Dataset_CSH/Avatar FULL 1080p 00942.jpg';%'6_20_s.bmp';%'Saba2.bmp';%
% fileNameA =   'imgs/ArchSequence/A_3_02.jpg';%'imgs/co_seg/10_15_s.bmp';%'imgs/frame0.jpg';% 'imgs/src.png';%'01a.png'; %
% fileNameB =  'imgs/ArchSequence/A_5_01.jpg';%'imgs/co_seg/10_14_s.bmp';%'imgs/frame1.jpg';%'imgs/ref.png';%'01b.png'; %
imgNames = {fileNameA,fileNameB};
nameA = fileNameA(1:end-4); nameB = fileNameB(1:end-4);
%% parameter setup
run('vlfeat-0.9.16/toolbox/vl_setup')
addpath('features');
addpath('dicts');
addpath('others');

alpha0 = 1;
alpha = 5;
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



%% test kd tree
% tic
% [hA,wA,dA] = size(I_rgb_A);
% [hB,wB,dB] = size(I_rgb_B);
% fA = reshape(I_rgb_A,hA*wA,dA);
% fB = reshape(I_rgb_B,hB*wB,dB);
% % fA = fA';fB=fB';
% ind_min=vl_kdtreequery(vl_kdtreebuild(double(fB)'),double(fB'),double(fA'),'NUMNEIGHBORS',1,'MAXNUMCOMPARISONS',2);
% final_img = fB(ind_min',:);
% final_img = reshape(final_img,[hA,wA,dA]);
% figure,imshow(uint8(final_img));
% toc
%% extract SIFT
t0 = clock;
pca_basis = [];
sift_size = 4;

[sift1, bbox1] = ExtractSIFT(I_rgb_A, pca_basis, sift_size);
[sift2, bbox2] = ExtractSIFT(I_rgb_B, pca_basis, sift_size);
I_rgb_A = I_rgb_A(bbox1(3):bbox1(4), bbox1(1):bbox1(2), :);
I_rgb_B = I_rgb_B(bbox2(3):bbox2(4), bbox2(1):bbox2(2), :);
A_orig = I_rgb_A;
B_orig = I_rgb_B;
fprintf('Extract SIFT: %f\n',etime(clock,t0));
%% Calculate the superpixels and features
[hA,wA,dA] = size(I_rgb_A);
[hB,wB,dB] = size(I_rgb_B);
set_options;
opts.seg_method = 'slic';% 'felz';%
opts.slic_region_size = 20; % default 20

% I_rgb_A = imresize(I_rgb_A,scale);
% lab_A  = colorspace('Lab<-', I_rgb_A);
[ spA,K_A,feaA,feaS_A ] = cal_sp_feature_sift( I_rgb_A,fileNameA,sift1,opts );% cal

% I_rgb_B = imresize(I_rgb_B,[hA,wA]);
% I_rgb_B = imresize(I_rgb_B,scale);
% lab_B  = colorspace('Lab<-', I_rgb_B);
[ spB,K_B,feaB,feaS_B] = cal_sp_feature_sift( I_rgb_B,fileNameB,sift2,opts );

%% Calculate similarity scores of all superpixels pairs between two images
if ~exist('temp/')
    mkdir('temp/')
end
% scores_mat = ['temp/scores_' nameA '_' nameB '.mat'];
% if ~exist(scores_mat)
    t0 = clock;
%     scores_AB = img_sp_similarity3(spA,spB,opts);
    Distance_c = sqdist(feaA',feaB');
    Distance_sift = sqdist(feaS_A',feaS_B');
    Distance = alpha0*Distance_c+alpha*Distance_sift;
    fprintf('Calculate similarity scores of \n all superpixels pairs: %f\n',etime(clock,t0));
%     save(scores_mat,'scores_AB');
% else
%     load(scores_mat);
% end
%% Calculate the candidate superpixel set for image A from image B

% [~,ind]=sort(scores_AB,2,'descend');
% [~,ind]=sort(Distance,2);
[~,ind]=min(Distance,[],2);


nA = length(spA);  nB = length(spB);
% k = max(ceil(0.03*nB),10);
% 
% candidate_match = ind(:,1:k);
% 
% temp = ind(:,1:k); matchB_ind = unique(temp);
% inv_matchB_ind = zeros(nB,1); inv_matchB_ind(matchB_ind) = [1:length(matchB_ind)];
% group1 = zeros(nA*k,nA); group2 = zeros(nA*k,length(matchB_ind));
    % center cordinate of each superpixel
spA_center = zeros(nA,2); spB_center = zeros(nB,2);
for i = 1:nA
    spA_center(i,:)= mean(spA{i}.pixels,1);
end
for i = 1:nB
    spB_center(i,:)= mean(spB{i}.pixels,1);
end
% displaySuperpixelMatching(I_rgb_A,I_rgb_B,spA,spB,ind(:,1),spA_center,spB_center);
    % translation vector
% Trans = zeros(nA*k,2); pairs = zeros(nA*k,4);
% d_app1 = zeros(nA*k,1);
% count = 0;
% for j = 1:k
%     for i = 1:nA
%         sub = ind(i,j);  count = count+1;
%         Trans(count,:) = spB_center(sub,:)-spA_center(i,:); 
%         d_app1(count) = 1-scores_AB(i,sub);
%         group1(count,i)=1;group2(count,inv_matchB_ind(sub)) = 1;
% %         pairs(count,1:2) = spA_center(i,:);
% %         pairs(count,3:4) = spB_center(j,:);
%     end
% end
% 
% group1 = sparse(group1); group2 = sparse(group2);
% % invTrans = -Trans;
% %% Calculate the Pairs-wise similarity matrix 
%     % Geometric dissimilarity between pairs
% n_pairs = nA*k;
% temp1 = repmat(Trans(:,1),1,n_pairs );
% temp2 = repmat(Trans(:,2),1,n_pairs );
% temp3 = zeros(n_pairs,n_pairs,2); temp4 = temp3;
% temp3(:,:,1) = temp1; temp3(:,:,2) = temp2; 
% temp4(:,:,1) = temp1'; temp4(:,:,2) = temp2'; 
% d_geo = mean(abs(temp3-temp4),3);
%     % Appearance dissimilarity between pairs
% temp1 = repmat(d_app1,1,n_pairs );
% temp2 = temp1'; 
% d_app=max(temp1,temp2);
%     % Pairs-wise similarity matrix 
% alpha = 500;
% sigma2 = 1000;
% d = d_geo+alpha*d_app;
% M = exp(-(d)/sigma2);
% % affinity_max = 200;               % maximum value of affinity
% % M = max(affinity_max-d,0);
% %% Calculate the steady state distribution of RRWM
% addpath('RRWM');
% [ Xraw ] = RRWM( M, group1, group2);
% X = greedyMapping(Xraw, group1, group2);

% final_sp_match = candidate_match(logical(X));
final_sp_match = ind(:,1);
if showFlag
    % display the result of superpixel matching
    displaySuperpixelMatching(I_rgb_A,I_rgb_B,spA,spB,final_sp_match,spA_center,spB_center);
end
%% global color mapping
scores = cal_consistence_sp(feaA,spA_center,spB_center,final_sp_match,spA);
[~,ind_s] = sort(scores);
n_cand = ceil(nA*0.5);
candinate1 = ind_s(1:n_cand);
candinate2 = final_sp_match(candinate1);
% [ paraMat ] = InterSpline2( feaA, feaB, candinate1,candinate2, 1 );
% feaA = colorspace('RGB<-Lab', feaA);
% feaB = colorspace('RGB<-Lab', feaB);
% Cubic spline interpolation
for channel = 1 : 3 % R, G, B
    eval(['paraMat_' num2str(channel) '  = InterSpline2( feaA, feaB, candinate1,candinate2, channel );']);
end
A = I_rgb_A; B = I_rgb_B;
% A = lab_A; B = lab_B;
% Test display
if showFlag
t_xx = 0.0 : 0.05 : 1;
t_yyr = ppval(paraMat_1, t_xx);
t_yyg = ppval(paraMat_2, t_xx);
t_yyb = ppval(paraMat_3, t_xx);
figure,plot(t_xx, t_yyr, 'r', t_xx, t_yyg, 'g', t_xx, t_yyb, 'b');
end

% Color transfer 
A = double(A) / 255;
newA = A;
newA(:, :, 1) = ppval(paraMat_1, A(:, :, 1));
newA(:, :, 2) = ppval(paraMat_2, A(:, :, 2));
newA(:, :, 3) = ppval(paraMat_3, A(:, :, 3));

B = double(B) / 255;
fprintf('all time: %f\n',etime(clock,t00));
if showFlag
    figure,imshow([A, newA, B]);
    imwrite(newA, [fileNameA(1:end-4) '_map_result.png']);
end
% A = A*255; newA = newA*255;
% A  = colorspace('RGB<-Lab', A);
% newA  = colorspace('RGB<-Lab', newA);
% B = colorspace('RGB<-Lab', B);
% figure,imshow([A, newA, B]);
% imwrite(newA, [fileNameA(1:end-4) '_gm_lab_result.png']);

% %% Pixel-wise match (minization of distance)
% matchB2A = zeros(hA,wA,2);
% matchB2A_img = zeros(hA,wA,dA);
% 
% 
% %tic
% %只使用图A超像素的最近邻 作为候选超像素
% % for i = 1:nA
% %     pixels1 = spA{i}.pixels;
% %     pixels2 = spB{final_sp_match(i)}.pixels;
% %     n1 = size(pixels1,1); n2 = size(pixels2,1);
% %     vector1 = zeros(n1,dA); vector2 = zeros(n2,dB);
% %     t0 = clock;
% %     for j = 1:n1
% %         vector1(j,:) = I_rgb_A(pixels1(j,1),pixels1(j,2),:);
% %     end
% %     for j = 1:n2
% %         vector2(j,:) = I_rgb_B(pixels2(j,1),pixels2(j,2),:);
% %     end
% %     
% %     if i == 1
% %     fprintf('Calculate the distance matrix \nof pixels in each superpixel: %f\n',etime(clock,t0));
% %     end
% %     dist_v1v2 = sqdist(double(vector1)',double(vector2)');
% %     
% %     [~,ind_min] = min(dist_v1v2,[],2);
% %     
% % %     ind_min=vl_kdtreequery(vl_kdtreebuild(double(vector2)'),double(vector2)',double(vector1)','NUMNEIGHBORS',1,'MAXNUMCOMPARISONS',2);
% % %     ind_min = ind_min';
% %     for j = 1:size(pixels1)
% %         sub1 = pixels1(j,:);
% %         sub2 = pixels2(ind_min(j),:);
% %         matchB2A(sub1(1),sub1(2),:) = sub2;
% %         matchB2A_img(sub1(1),sub1(2),:) = I_rgb_B(sub2(1),sub2(2),:);
% %     end
% % end
% 
% 使用图A的超像素及其相邻的超像素 它们所对应的图B中的最近邻 作为候选的像素集
if is_pixelmatch_n 
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


% toc
if showFlag
 figure,imshow(uint8(matchB2A_img));
 resultpath = [fileNameA(1:end-4),'_result.png'];
imwrite(uint8(matchB2A_img),resultpath);
end
matchB2A_img = uint8(matchB2A_img);
end

end
% 
%  error = double(I_rgb_A)-matchB2A_img;
%  error = reshape(error,hA*wA,dA);
%  error = mean(sqrt(sum(error.^2,2)));
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

