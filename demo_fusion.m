close all 
clear

dataPath = 'imgs/fusion/';
dataName = '1';%'HappyHour';%'Dome';
fileNameA = [dataPath,dataName,'/3.jpg'];%'imgs/ArchSequence/A_3_02.jpg';%'01a.png'; % 'imgs/co_seg/14_20_s.bmp';%'imgs/dataset/cliffs1_-2.jpg';%'imgs/image_0001.jpg';%'imgs/frame0.jpg';%
fileNameB = [dataPath,dataName,'/2.jpg'];%'01b.png'; %'imgs/co_seg/14_14_s.bmp';%'imgs/dataset/cliffs1_0.jpg';'imgs/image_0082.jpg';%'imgs/frame1.jpg';%
fileNameC = [dataPath,dataName,'/4.jpg'];

[matchB2A_img imgA] = do_lab(fileNameA,fileNameB);
matchC2A_img = do_lab(fileNameA,fileNameC);
% imgA = imread(fileNameA);
% rf = max(max(size(imgA)))  / 640;
% if (rf > 1)
%     scale = 1.0/rf;
%     disp('source image is too big. resize automatically.')
%     imgA = imresize(imgA,scale);    
% end

resultPath = ['./fusionRes/'  dataName ,'/'];
if ~exist(resultPath,'dir')
    mkdir(resultPath);
end
imwrite(matchB2A_img,[resultPath,'1.jpg']);
imwrite(imgA,[resultPath,'2.jpg']);
imwrite(matchC2A_img,[resultPath,'3.jpg']);

fusionRes = ifuse(resultPath);
figure,imshow(fusionRes);
imwrite(fusionRes,['./fusionRes/',dataName '_fusion.jpg']);