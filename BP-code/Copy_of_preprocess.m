clear;clc;

trainset = 'H:\2018下\SIAT\AI\AI培训\培训资料\CNN\BP神经网络识别手写阿拉伯数字-wqt201511\trainSet';
subdir = dir(trainset);

Features = zeros(8,12,10);

for i = 1:length(subdir)
    if(isequal(subdir(i).name,'.')||...
       isequal(subdir(i).name,'..')||...   //去掉文件夹  
       subdir(i).isdir)
        continue;
    end
    
    image = fullfile (trainset,subdir(i).name);
    I = imread(image);
    grayImage = rgb2gray(I);
    subplot(2,1,1),imshow(grayImage),title('Gray Image');
    [y,x]=find(grayImage==0);
    croppedImage=imcrop(grayImage,[min(x),min(y),max(x)-min(x),max(y)-min(y)]);
    subplot(2,1,2),imshow(croppedImage),title('Cropped Image');
    
    %将每幅图片进行8*12分割
    [y,x] = find(croppedImage == 0);
    lengthx=max(x)/8;
    lengthy=max(y)/12;
    x02=0:lengthx:max(x);
    y02=0:lengthy:max(y);
    [X,Y]=meshgrid(x02,y02);
     
    %求特征向量
    lengx=floor(lengthx);
    lengy=floor(lengthy);
    for n=1:8
        for m=1:12
             for p=lengx*(n-1)+1:lengx*n
                 for j=lengy*(m-1)+1:lengy*m
                       if croppedImage(j,p)==0
                         Features(n,m,i-2)=1;
                         break;
                       else
                         Features(n,m,i-2)=0;
                       end

                 end
                 if Features(n,m,i-2)==1
                     break;
                 end
            end
        end
    end

end

number = reshape(Features,96,10);  %讲特征进行拉伸
targets=eye(10);
