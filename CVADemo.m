function  testImage = CVADemo( )

%% 变化向量分析法 CVA 获取差异影像


%load two images envi data
% fisrt_image=freadenvi('2002roi',1);  %280*358 4band     100240*4
% second_image=freadenvi('2005roi',1);  %280*358 4band

[X,p]=freadenvi('data\\Taizhou\\2000TM',1);
[Y,p2]=freadenvi('data\\Taizhou\\2003TM',1);

rows=p(1);
cols=p(2);
bands=p(3);
disp(size(X));


X=double(X);
Y=double(Y);

%滤波
% for i=1:1
% h2 = fspecial('gaussian',5,1); %高斯低通滤波 hsize表示模板尺寸，默认值为【3 3】，sigma为滤波器的标准值，单位为像素，默认值为0.5.这里是1
% 
% X = imfilter(X, h2, 'symmetric'); %滤波器
% Y = imfilter(Y, h2, 'symmetric');
% 
% end;
%计算CVA强度 欧几里得距离
Diff_V=Y-X;
A=sqrt(sum(Diff_V.*Diff_V,2));
% A=sqrt(Diff_V.*Diff_V);
%PCA降维 
% [A,T,meanValue,test] = PCA(A,0.75);

% disp(size(A));
% 
% %像素四舍五入取整
% A=round(A);

% 对CVA强度进行归一化
A=mapminmax(A,0,255);
amax = max(max(A));  
amin = min(min(A)); 
A=255*(A-amin)/(amax-amin);

% A(A<50)=255;

%显示输入和结果图像
%imshow(A')
%保存图像
CM=KmeansMap(A,rows,cols);

testImage=CM;
changeImage=CM;
changeImage(find(changeImage==1))=255;
% imwrite(A,'CVAImage_Taizhou.tif');
enviwrite2(A,'CVAImageTaizhou_IntensityImage');
enviwrite2(changeImage,'CVAImageTaizhou_BinaryValue');
 figure, imshow(changeImage)
end