function max_differ_index = find_max_differ(x)

len = length(x);
% 从起点到终点的斜线
line_k = (x(end)-x(1))/(len-1); %斜率
line_x = 0:len-1;
line_y = x(1) + line_k*line_x;

% 实际信号与线段的差值
difference = x - line_y;
[~,max_differ_index] = max(difference); % 差值的最大值

end
