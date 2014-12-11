faceDetector = vision.CascadeObjectDetector();

images = dir([ 'Images/*.' 'jpg']);

counter = 0;
corrupted = 0;

for im=1:size(images(:))
    % Read a video frame and run the detector.
    image      = imread(strcat('Images/',images(im).name));
    bbox       = step(faceDetector, image);

    for i=1:size(bbox(:,1))
        try
        counter = counter + 1;
        string = strcat('Images/faces/cara',int2str(i+counter),'.jpg');
        imwrite(image(bbox(i,2):bbox(i,2)+bbox(i,4),bbox(i,1):bbox(i,1)+bbox(i,3),:),string);
        counter = counter + 1;
        image = imread(string);
        string = strcat('Images/faces/cara',int2str(i+im),'.jpg');
        imwrite(rgb2gray(image),string);
        catch
            corrupted = corrupted + 1;
        end
    end
end