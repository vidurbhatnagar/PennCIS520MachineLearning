close all;

images = images_test;
numImages = size(images,1);

% labels = zeros(numImages,1);
for iter = 4186:numImages
    iter
    imshow(uint8(reshape(images(iter,:),[100 100 3])));
    [~,~,button] = ginput(1);
    
    if button == 1
        labels(iter) = 0;
    else
        labels(iter) = 1;
    end
end
