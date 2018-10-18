P=number;
T=targets;
[R,Q]=size(P); 
S1=10;  % S1，第一层的神经元个数为10  
[S2,Q]=size(T); % S2，第二层的神经元个数为S2  

% 构建BP网络  %minmax()函数得到了每一行的最小值和最大值

net=newff(minmax(P), [S1,S2],{'logsig','logsig'},'traingdx');   
net.LW{2,1}=net.LW{2,1}*0.01; % 调整第二层的权值  
net.b{2}=net.b{2}*0.01;       % 调整第二层的阈值  

%  当前网络层权值和阈值   
layerWeights=net.LW{2,1};   
layerbias=net.b{2};  
% 无噪声训练  
net.performFcn='sse';       % 性能函数，误差平方和  
net.trainParam.epochs=800; % 训练次数 
net.trainParam.lr = 0.01;%学习速率
net.trainParam.lr_inc = 1.05;%增长的学习速率
net.trainParam.lr_dec = 0.7;

net.trainParam.goal=0.1;  
net.trainParam.mc=0.9;     % 附加动量 
net.trainParam.min_grad=1e-10;
net.trainParam.show=50;  
P=number;
T=targets;
 
[net,tr]=train(net,P,T); 
