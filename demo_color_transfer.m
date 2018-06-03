close all;  clear ;

%% input images
%  fileNameA = 'imgs/Dataset_CSH/Avatar FULL 1080p 00907.jpg';%'6_24_s.bmp';%'Saba1.bmp';%
% fileNameB = 'imgs/Dataset_CSH/Avatar FULL 1080p 00942.jpg';%'6_20_s.bmp';%'Saba2.bmp';%
fileNameA =  'imgs/color_transfer/14_14_s.bmp';%'imgs/color_transfer/russ3/1.jpg';% '01a.png'; %'imgs/ArchSequence/A_3_02.jpg';%'imgs/co_seg/10_15_s.bmp';%'imgs/frame0.jpg';% 
fileNameB =  'imgs/color_transfer/14_20_s.bmp';%'imgs/color_transfer/russ3/2.jpg';%'01b.png'; %'imgs/ArchSequence/A_5_01.jpg';%'imgs/co_seg/10_14_s.bmp';%'imgs/frame1.jpg';%

[matchB2A_img imgA] = do_rgb(fileNameA,fileNameB);
 figure,imshow(uint8(matchB2A_img));
 resultpath = [fileNameA(1:end-4),'_result.png'];
imwrite(uint8(matchB2A_img),resultpath);



