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

%�˲�
% for i=1:1
% h2 = fspecial('gaussian',5,1); %��˹��ͨ�˲� hsize��ʾģ��ߴ磬Ĭ��ֵΪ��3 3����sigmaΪ�˲����ı�׼ֵ����λΪ���أ�Ĭ��ֵΪ0.5.������1
% 
% X = imfilter(X, h2, 'symmetric'); %�˲���
% Y = imfilter(Y, h2, 'symmetric');
% 
% end;
%����CVAǿ�� ŷ����þ���
Diff_V=Y-X;
% A=sqrt(sum(Diff_V.*Diff_V,2));
A=sqrt(Diff_V.*Diff_V);
%PCA��ά 
 [A,T,meanValue,test] = PCA(A,0.75);


%��ʾ����ͽ��ͼ��
%imshow(A')
%����ͼ��
CM=KmeansMap(A,rows,cols);

testImage=CM;
changeImage=CM;
changeImage(find(changeImage==1))=255;
enviwrite2(A,'PCAImageTaizhou_IntensityImage');
enviwrite2(changeImage,'PCAImageTaizhou_BinaryValue');
 figure, imshow(changeImage)
end