for i = 0:9

imname = [num2str(i) '.bmp'];
im = imread(imname);

im = [im(:)]';

%im = ones(size(im)) - im;

%load('ex3data1.mat');
%load('ex3weights.mat');

pred = predict(Theta1, Theta2, im);

fprintf('\n %s == digit %d\n', imname, mod(pred, 10));


end