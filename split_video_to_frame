%Program for splitting a video sequences of actions into frames 
%Author: @leopauly

clc
clear 
close all

total_actions=5;
action_list=containers.Map({0,1,2,3,4,5}, {'boxing','clapping','jogging','running','walking','waving'});

for action_id=0:total_actions
file_list = dir(['./',action_list(action_id),'/*.avi']);
L=length(file_list);
j=0; 

for seq_num=1:L
vid = VideoReader(['./',action_list(action_id),'/',file_list(seq_num).name]); %#ok<TNMLP>
num_frames = vid.NumberOfFrames; %#ok<VIDREAD>
step_size=30;
re_size=[150,300];
frame_num=0;
 
for i = 1:step_size:num_frames
 frames = read(vid,i); %#ok<VIDREAD>
 frames=imresize(frames,re_size);
 %imwrite(frames,['./LAS/',int2str(action_id),int2str(seq_num),int2str(frame_num),'.png']);
 imwrite(frames,['./LAS/',int2str(action_id),'/',int2str(j),'.png']);
 j=j+1;
 frame_num=frame_num+1;
 im(i)=image(frames);
end

end

end
