function  testImage = CVADemo( )

%% �仯���������� CVA ��ȡ����Ӱ��


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
A=sqrt(sum(Diff_V.*Diff_V,2));
% A=sqrt(Diff_V.*Diff_V);
%PCA��ά 
% [A,T,meanValue,test] = PCA(A,0.75);

% disp(size(A));
% 
% %������������ȡ��
% A=round(A);

% ��CVAǿ�Ƚ��й�һ��
A=mapminmax(A,0,255);
amax = max(max(A));  
amin = min(min(A)); 
A=255*(A-amin)/(amax-amin);

% A(A<50)=255;

%��ʾ����ͽ��ͼ��
%imshow(A')
%����ͼ��
CM=KmeansMap(A,rows,cols);

testImage=CM;
changeImage=CM;
changeImage(find(changeImage==1))=255;
% imwrite(A,'CVAImage_Taizhou.tif');
enviwrite2(A,'CVAImageTaizhou_IntensityImage');
enviwrite2(changeImage,'CVAImageTaizhou_BinaryValue');
 figure, imshow(changeImage)
end