clc;
clear all;
close all;

%% INITIALIZATION
%=================
% Read image sequence
imPath = 'car'; imExt = 'jpg';

% check if directory and files exist
if exist(imPath, 'dir') ~= 7
error('ERROR USER: The image directory does not exist');
end

filearray = dir([imPath filesep '*.' imExt]); % get all files in the directory
NumImages = size(filearray,1); % get the number of images
if NumImages < 0
error('No image in the directory');
end

disp('Loading image files from the video sequence, please be patient...');
imgname = [imPath filesep filearray(1).name];
I = imread(imgname);
VIDEO_WIDTH = size(I,2);
VIDEO_HEIGHT = size(I,1);
ImSeq = zeros(VIDEO_HEIGHT, VIDEO_WIDTH,NumImages);

for i=1:NumImages
imgname = [imPath filesep filearray(i).name];
image = imread(imgname);
image = im2double(image);
b(i,:) = image(:);
ImSeq(:,:,i) = image;
end
disp(' ... OK!');
c = b';
d = c';
m = sum(b)/NumImages; % Calculating mean
meanim = reshape(m,VIDEO_HEIGHT,VIDEO_WIDTH); % Mean image

for loop = 1:NumImages
    
    X(:,loop) = c(:,loop) - m';
end

[U S V] = svds(X); % Doing SVD
Ur = U; % Taking diagonalised matrix with singular values
sizeofimg = ImSeq(:,:,1);
[rows,cols] = size(sizeofimg);
T = graythresh(m)*0.5; % Thresholding using Otsu method
dummy = 0;

for i = 1:NumImages
input = ImSeq(:,:,i);
input = input(:);
p = Ur'*(input - m');
y_bar = Ur * p + m'; % Projecting image
im = reshape(y_bar,rows,cols);

% figure(1);
% pause(0.01);
% imshow(im,[]);
% title('Projected Image');

diff = abs(input-y_bar);
diffre = reshape(diff,rows,cols);

% figure(2);
% pause(0.01);
% imshow(diffre,[]);
% title('Difference of input and projected image');

diff(diff > T) = 0;
Tdiff = reshape(diff,rows,cols);

% figure(3);
% pause(0.01);
% imshow(Tdiff,[]);
% title('Difference of i/p and proj image with threshold');
% end

% kalman filter initialization
xc = zeros(NumImages,1);
yc = zeros(NumImages,1);
prediction = zeros(NumImages,4);
origi = zeros(NumImages,4);

% Initializing kalman filter parameters
R = [[0.2845,0.0045]',[0.0045,0.0455]']; % measurement noise
H = [[1,0]',[0,1]',[0,0]',[0,0]']; % Transform fron measure to state
Q = 0.01 * eye(4); % system noise
P = 100 * eye(4); % Covariance matrix
A = [[1,0,0,0]',[0,1,0,0]',[1,0,1,0]',[0,1,0,1]']; % state transform matrix

thresh = T;
    figure(4);
    imshow(ImSeq(:,:,i),[]);
    title('Tracking');
    hold on
    currentimg = double(ImSeq(:,:,i));
    diffimg = zeros(VIDEO_WIDTH,VIDEO_HEIGHT);
    hh = ((abs(currentimg-Tdiff))>thresh)*0.5;
    hh = hh/255;
    labelObj = bwlabel(hh,8);

    diffimg = (abs(currentimg-meanim)>thresh); % Difference between current image and eigenbackground image
    labelimg = bwlabel(diffimg,4); % getting label image
    prop = regionprops(labelimg,'basic'); % Getting properties of image
    [pr pc] = size(prop);
   
    % Getting lowest area around object
    for pc = 1:pr
        if prop(pc).Area > prop(1).Area
            tempimg = prop(1);
            prop(1) = prop(pc);
            prop(pc) = tempimg;
        end 
    end
    
    % Getting bounding box
    bb = prop(1).BoundingBox;
    xcorner = bb(1);
    ycorner = bb(2);
    x_width = bb(3);
    y_width = bb(4);
    center = prop(1).Centroid;
    xc(i) = center(1);
    yc(i) = center(2);
    
    hold on
    rectangle('Position',bb,'EdgeColor','g','LineWidth',3);
    hold on
    plot(xc(i),yc(i),'bx','LineWidth',1);
    
    % Kalman Window calculation
    kalmanx = xc(i) - xcorner;
    kalmany = yc(i) - ycorner;
    if dummy == 0
        prediction = [xc(i),yc(i),0,0]';
    else
        prediction = A*origi(i-1,:)';
    end
    dummy = 1;
    Ppre = A * P * A' + Q;
    K = (Ppre * H')/(H * Ppre * H' +R);
    origi(i,:) = (prediction + K * ([xc(i),yc(i)]' - H*prediction))';
    P = (eye(4)-K * H) * Ppre;
    xc_update = (origi(i,1)-kalmanx);
    yc_update = (origi(i,2)-kalmany);
    
    hold on
    rectangle('Position',[xc_update yc_update x_width y_width],'EdgeColor','r','LineWidth',1);
    hold on;
    plot(origi(i,1),origi(i,2),'rx','LineWidth',1);
    drawnow;

end
     
