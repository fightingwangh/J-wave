function max_differ_index = find_max_differ(x)

len = length(x);
% ����㵽�յ��б��
line_k = (x(end)-x(1))/(len-1); %б��
line_x = 0:len-1;
line_y = x(1) + line_k*line_x;

% ʵ���ź����߶εĲ�ֵ
difference = x - line_y;
[~,max_differ_index] = max(difference); % ��ֵ�����ֵ

end
