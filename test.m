I = imread('Train\captcha_0002.png');
%imshow(I);

BW = rgb2gray(I);
K1 = imgaussfilt(BW,2);
%imshow(K1);

counts = imhist(K1);
T = otsuthresh(counts);
K2 = ~imbinarize(K1,T); 

%imshow(K2);

K2 = imerode(K2, strel('disk',4));
    K2 = bwareaopen(K2, 400); % Remove specs(noise) less that are < 400px
    K3 = imdilate(K2, strel('disk',3));
    
%imshow(K3);
% figure
% subplot(1,2,1),imshow(I),title('Original Image');
% subplot(1,2,2),imshow(BW),title('Grayscale image');
% subplot(1,2,1),imshow(K1),title('Image after applying Gaussian filter');
% subplot(1,2,2),imshow(K2),title('Binarized Image');
% subplot(1,2,1),imshow(K3),title('Image after Erosion and Dialation');

% Segmentation of the digits
    CC = bwconncomp(K3,4); %4-connected - If their edges are connected
    RP = regionprops(CC, 'Image'); %Returns measurements for CC properties
    %If all are connected
    if CC.NumObjects == 1
        allDigits = RP(1).Image;
        [h,w] = size(allDigits);
        div = round(w/3);
        d1 = imcrop(allDigits,[0 0 div h]); %(xmin ymin width height)
        d2 = imcrop(allDigits,[div 0 div h]);
        d3 = imcrop(allDigits,[div*2 0 div h]);
        F(1,:,:) = ShapeFeats(d1);F(2,:,:) = ShapeFeats(d2);F(3,:,:) = ShapeFeats(d3);
    %If 2 are connected and one is not  
    elseif CC.NumObjects == 2
        digLeft = RP(1).Image;
        digRight = RP(2).Image;
        [h1,w1] = size(digLeft);
        [h2,w2] = size(digRight);
        divLeft = round(w1/2); %If first two digits are connected
        divRight = round(w2/2); %If last two digits are connected
        if w1 > w2 % Divide first two digits
            d1 = imcrop(digLeft,[0 0 divLeft h1]);
            d2 = imcrop(digLeft,[divLeft 0 divLeft h1]);
            d3 = digRight;
        else % Divide last two digits
            d1 = digLeft;
            d2 = imcrop(digRight,[0 0 divRight h2]);
            d3 = imcrop(digRight,[divRight 0 divRight h2]);
        end
        F(1,:,:) = ShapeFeats(d1);F(2,:,:) = ShapeFeats(d2);F(3,:,:) = ShapeFeats(d3);
    %If all 3 digits are seperated  
    elseif CC.NumObjects == 3
        digOne = RP(1).Image;
        digTwo = RP(2).Image;
        digThree = RP(3).Image;
        F(1,:,:) = ShapeFeats(digOne);
        F(2,:,:) = ShapeFeats(digTwo);
        F(3,:,:) = ShapeFeats(digThree);
    end

figure;
subplot(1,3,1),imshow(I),title('Original Image');
subplot(1,3,2),imshow(BW),title('Grayscale image');
subplot(1,3,3),imshow(K1),title('Image after applying Gaussian filter');
figure;
subplot(1,3,1),imshow(K2),title('Binarized Image');
subplot(1,3,2),imshow(K3),title('Image after Erosion and Dialation');
subplot(1,3,3),imshow(d1),title('Segmented digit');

    
function F=ShapeFeats(S)
	fts={'Circularity','Area','ConvexArea'}; 
	Ft=regionprops('Table',S,fts{:});
	[~,idx]=max(Ft.Area);
	F=[Ft(idx,:).Variables];
end