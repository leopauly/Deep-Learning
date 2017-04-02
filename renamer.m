% Matlab script to read all images in a folder and rename them in order

clc
clear all
close all
imagefiles=dir('*.png')
num=124
for i=1:130
    imagename=imagefiles(i).name
    image=imread(imagename);
    image=imresize(image,[99,99]);
    figure,imshow(image);
    c=int2str(num);
    newname=strcat('./',c,'.jpg');
    num=num+1;
    imwrite(image,newname);
end
