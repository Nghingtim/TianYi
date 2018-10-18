I=imread('.\testSet\T3.jpg');%读取图像
grayImage = rgb2gray(I);
subplot(2,1,1),imshow(T),title('Gray Image');
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
         for i=lengx*(n-1)+1:lengx*n
             for j=lengy*(m-1)+1:lengy*m
                   if croppedImage(j,i)==0
                     feature(n,m)=1;
                     break;
                   else
                     feature(n,m)=0;
                   end
                   
             end
             if feature(n,m)==1
                 break;
             end
        end
    end
end

testpoint=feature(1:96)';
a0=sim(net,testpoint);
[a0x, a0y]=max(a0);
disp(['Recognized number is:',num2str(a0y-1)])
disp(['Its possibility is:',num2str(a0x)])
