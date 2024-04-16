%% 该函数用于找到信号的s波与st段交点，即拐点
function index = find_inflection_point(x)

% 预处理，确保信号为1×n
s = size(x); %1*X维度
if s(2) == 1 %第二维是1 即只有一个数据
    x = x.';
end

% [~,max_index]=max(x);   % 在全局范围内找极大值点（r点），容易被相邻心拍干扰
% min_index=max_index;
[~,max_index]=max(x(ceil(length(x)/2-50):ceil(length(x)/2+50)));   % 在中心附近找极大值点（r点）的索引 共101拍 R点在所截取片段的索引为50
%disp(max_index);
%[~,max_index]=max(x(ceil(length(x)/2-20):ceil(length(x)/2+20)));
% [C,index] = max(A)：返回返回行向量C和index，C向量记录矩阵A每列的最大值，index记录A每列最大值的行号（即每列最大值的索引）
min_index=max_index+ceil(length(x)/2-50);  %此时最小值点在R峰 50
value_temp=sum(x(min_index:min_index+2));
value_next=sum(x(min_index+1:min_index+3));
while(value_temp>=value_next && min_index<300)
    min_index = min_index+1;
    value_temp=sum(x(min_index:min_index+2));
    value_next=sum(x(min_index+1:min_index+3));
end
min_index=min_index+1;  % 找到极小值点（s点）,即线段起点
end_index=min_index+20; % 线段终点 25
max_differ_index=find_max_differ(x(min_index:end_index));
max_differ_index=max_differ_index+min_index-1;  % 差值最大点
index=max_differ_index; %返回拐点



%index=min_index; %去掉s波时，返回S点

% if x(max_differ_index) < x(max_differ_index+1)
%     % 若信号仍缓慢上升时，采用二次差值提高精度
%     max_differ_index_2=find_max_differ(x(max_differ_index:end_index));
%     max_differ_index_2=max_differ_index_2+max_differ_index-1;   % 二次差值最大点
%     index=max_differ_index_2;
% end

end

