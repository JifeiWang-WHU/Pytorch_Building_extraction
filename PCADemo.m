function  testImage =PCADemo()


%load two images envi data

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
% A=sqrt(sum(Diff_V.*Diff_V,2));
A=sqrt(Diff_V.*Diff_V);
%PCA降维 
 [A,T,meanValue,test] = PCA(A,0.75);


%显示输入和结果图像
%imshow(A')
%保存图像
CM=KmeansMap(A,rows,cols);

testImage=CM;
changeImage=CM;
changeImage(find(changeImage==1))=255;
enviwrite2(A,'PCAImageTaizhou_IntensityImage');
enviwrite2(changeImage,'PCAImageTaizhou_BinaryValue');
 figure, imshow(changeImage)
end