P=number;
T=targets;
[R,Q]=size(P); 
S1=10;  % S1����һ�����Ԫ����Ϊ10  
[S2,Q]=size(T); % S2���ڶ������Ԫ����ΪS2  

% ����BP����  %minmax()�����õ���ÿһ�е���Сֵ�����ֵ

net=newff(minmax(P), [S1,S2],{'logsig','logsig'},'traingdx');   
net.LW{2,1}=net.LW{2,1}*0.01; % �����ڶ����Ȩֵ  
net.b{2}=net.b{2}*0.01;       % �����ڶ������ֵ  

%  ��ǰ�����Ȩֵ����ֵ   
layerWeights=net.LW{2,1};   
layerbias=net.b{2};  
% ������ѵ��  
net.performFcn='sse';       % ���ܺ��������ƽ����  
net.trainParam.epochs=800; % ѵ������ 
net.trainParam.lr = 0.01;%ѧϰ����
net.trainParam.lr_inc = 1.05;%������ѧϰ����
net.trainParam.lr_dec = 0.7;

net.trainParam.goal=0.1;  
net.trainParam.mc=0.9;     % ���Ӷ��� 
net.trainParam.min_grad=1e-10;
net.trainParam.show=50;  
P=number;
T=targets;
 
[net,tr]=train(net,P,T); 
