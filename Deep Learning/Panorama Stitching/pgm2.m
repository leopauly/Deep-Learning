clc 
clear 

obj=VideoReader('vid2.mp4');
for n=100:600
   a=0;
   a=(read(obj,n));
   c=int2str(n);
   imwrite(a,strcat('a',c,'.jpg'));
end


