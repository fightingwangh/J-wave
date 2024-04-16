clear; clc;

% 加载单个心电信号数据
data = load('e0107.mat');
%data=data.val;%访问了结构体中名为 "data" 的字段，以获取其中的数据
data = data.val(:, 2); % 选择第二列数据

% 加载标签信息
label = load('e0107_.mat');
% 提取第一列心拍索引值val
num=label.val(:, 1);
num=num.';

% 设置采样率
f = 250;

% % 初始化存放只有 J 波的信号
zero_data = zeros(length(data),1);

for i = 2:length(num) % 遍历所有心拍
    
    index = num(i);   % 该心拍的中心区域
    %disp(index);
    temp_data = data(index-120:index+120); % 获取心拍信号

    temp_data = temp_data.';  % 转置为1×n

    point = find_inflection_point(temp_data);   % 找到拐点
    point = (point-1) / f;  % 量化拐点位置

    x = 0:1/f:(length(temp_data)-1)/f;
    jwave = j_wave(x, point); % 随机 J 波
 
    % disp(size(jwave))
    % disp(size(temp_data))
    data(index-120:index+120) = temp_data + jwave;  % 将 J 波添加到心拍信号中
    

    % 保存 J 波信息
    for ti = 1:length(jwave)
        if jwave(ti) > 0
            jwave(ti) = 1;
        end
    end
    zero_data(index-120:index+120) = jwave;

    disp(['第 ', num2str(i), ' 拍已完成']);
end
