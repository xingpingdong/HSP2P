close all;  clear ;

%% input images

fileNameA =   'imgs/src.png'; 
fileNameB =  'imgs/ref.png';
imgNames = {fileNameA,fileNameB};
nameA = fileNameA(1:end-4); nameB = fileNameB(1:end-4);
%% parameter setup
run('vlfeat-0.9.16/toolbox/vl_setup')
addpath('features');
addpath('dicts');
addpath('others');

alpha0 = 1;
alpha = 0;
is_pixelmatch_n = 1;
t00=clock;

I_rgb_A = imread(fileNameA);
I_rgb_B = imread(fileNameB);
rf = max(max(size(I_rgb_A)))  / 640;
if (rf > 1)
    scale = 1.0/rf;
    disp('source image is too big. resize automatically.')
%     Src = imresize(Src, 1.0/rf);
%     Ref = imresize(Ref, 1.0/rf);
    I_rgb_A = imresize(I_rgb_A,scale);
    I_rgb_B = imresize(I_rgb_B,scale);
end

%% Calculate the superpixels and features
[hA,wA,dA] = size(I_rgb_A);
[hB,wB,dB] = size(I_rgb_B);
set_options;
opts.seg_method = 'slic';% 'felz';%
opts.slic_region_size = 20; % default 20

[ spA,K_A,feaA ] = cal_sp_feature3( I_rgb_A,fileNameA,opts );% cal

[ spB,K_B,feaB] = cal_sp_feature3( I_rgb_B,fileNameB,opts );

%% Calculate similarity scores of all superpixels pairs between two images
if ~exist('temp/')
    mkdir('temp/')
end

    t0 = clock;

    Distance = sqdist(feaA',feaB');

    fprintf('Calculate similarity scores of \n all superpixels pairs: %f\n',etime(clock,t0));

%% Calculate the candidate superpixel set for image A from image B

[~,ind]=min(Distance,[],2);


nA = length(spA);  nB = length(spB);

spA_center = zeros(nA,2); spB_center = zeros(nB,2);
for i = 1:nA
    spA_center(i,:)= mean(spA{i}.pixels,1);
end
for i = 1:nB
    spB_center(i,:)= mean(spB{i}.pixels,1);
end

final_sp_match = ind(:,1);
    % display the result of superpixel matching
displaySuperpixelMatching(I_rgb_A,I_rgb_B,spA,spB,final_sp_match,spA_center,spB_center);


% 使用图A的超像素及其相邻的超像素 它们所对应的图B中的最近邻 作为候选的像素集
if is_pixelmatch_n 
%     I_rgb_A = newA*255;
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

    dist_v1v2 = sqdist(double(vector1)',double(vector2)');
    [~,ind_min] = min(dist_v1v2,[],2);

    for j = 1:size(pixels1)
        sub1 = pixels1(j,:);
        sub2 = pixels2(ind_min(j),:);
        matchB2A(sub1(1),sub1(2),:) = sub2;
        matchB2A_img(sub1(1),sub1(2),:) = I_rgb_B(sub2(1),sub2(2),:);
    end
end


% toc
 figure,imshow(uint8(matchB2A_img));
 resultpath = [fileNameA(1:end-4),'_result.png'];
imwrite(uint8(matchB2A_img),resultpath);
end
% 
%  error = double(I_rgb_A)-matchB2A_img;
%  error = reshape(error,hA*wA,dA);
%  error = mean(sqrt(sum(error.^2,2)));
%  fprintf('The mean error is: %f\n',error);
%  
%  fprintf('all time: %f\n',etime(clock,t00));
%  