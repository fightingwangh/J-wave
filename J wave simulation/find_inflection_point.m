%% �ú��������ҵ��źŵ�s����st�ν��㣬���յ�
function index = find_inflection_point(x)

% Ԥ����ȷ���ź�Ϊ1��n
s = size(x); %1*Xά��
if s(2) == 1 %�ڶ�ά��1 ��ֻ��һ������
    x = x.';
end

% [~,max_index]=max(x);   % ��ȫ�ַ�Χ���Ҽ���ֵ�㣨r�㣩�����ױ��������ĸ���
% min_index=max_index;
[~,max_index]=max(x(ceil(length(x)/2-50):ceil(length(x)/2+50)));   % �����ĸ����Ҽ���ֵ�㣨r�㣩������ ��101�� R��������ȡƬ�ε�����Ϊ50
%disp(max_index);
%[~,max_index]=max(x(ceil(length(x)/2-20):ceil(length(x)/2+20)));
% [C,index] = max(A)�����ط���������C��index��C������¼����Aÿ�е����ֵ��index��¼Aÿ�����ֵ���кţ���ÿ�����ֵ��������
min_index=max_index+ceil(length(x)/2-50);  %��ʱ��Сֵ����R�� 50
value_temp=sum(x(min_index:min_index+2));
value_next=sum(x(min_index+1:min_index+3));
while(value_temp>=value_next && min_index<300)
    min_index = min_index+1;
    value_temp=sum(x(min_index:min_index+2));
    value_next=sum(x(min_index+1:min_index+3));
end
min_index=min_index+1;  % �ҵ���Сֵ�㣨s�㣩,���߶����
end_index=min_index+20; % �߶��յ� 25
max_differ_index=find_max_differ(x(min_index:end_index));
max_differ_index=max_differ_index+min_index-1;  % ��ֵ����
index=max_differ_index; %���عյ�



%index=min_index; %ȥ��s��ʱ������S��

% if x(max_differ_index) < x(max_differ_index+1)
%     % ���ź��Ի�������ʱ�����ö��β�ֵ��߾���
%     max_differ_index_2=find_max_differ(x(max_differ_index:end_index));
%     max_differ_index_2=max_differ_index_2+max_differ_index-1;   % ���β�ֵ����
%     index=max_differ_index_2;
% end

end

