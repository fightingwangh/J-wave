function [jwave,width] = j_wave(x,start)

width=normrnd(0.02,0.0001);%0.025   0.045  0.001 正常0.02 

height=normrnd(0.15,0.0001);%正常0.2  0.01 当去掉s波ST正常应加大0.26，ST抬高0.3

% while width<=0.02
%     width=normrnd(0.02,0.01);
% end
% while height<=0.2
%     height=normrnd(0.2,0.01);
% end

len_x = length(x);
interval = (x(end)-x(1))/(len_x-1); % 采样间隔
sin_wave = zeros(1,5*len_x);% 初始化
temp_x=0:interval:width;
% temp_x=interval*25:interval:width;
sin_wave(ceil(start/interval):ceil(start/interval)+length(temp_x)-1) =height*sin(pi*temp_x/width);

jwave=sin_wave(1:len_x);
%x1=x(start+width:start+width+0.1)+0.1;

end

