%Program for Data augmentation (resizing,flipping,rotating etc.)
%Author: @leopauly

clc
clear 
close all

re_size=[100 100]

file_list = dir(['./*.bmp']);  % Read all images in current folder 
L=length(file_list);
j=0; 

for i=1:L
    i
    for angle=1:180
 frames = imread(['./',file_list(i).name]); %#ok<TNMLP>
 frames=imresize(frames,re_size);  % Resizing
 %frames=flipdim(frames,1); % Flipping
 frames=imrotate(frames,angle,'bilinear','crop'); % Rotating
 frames=imresize(frames,re_size);  % Resizing
 imwrite(frames,['../0_rotate/angle_',int2str(angle),'_',file_list(i).name]); % Destination folder
    end
end
