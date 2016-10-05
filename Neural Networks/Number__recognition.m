for i = 0:9

imname = [num2str(i) '.bmp'];
im = imread(imname);

im = [im(1,:) im(2,:) im(3,:) im(4,:) im(5,:) ...
    im(6,:) im(7,:) im(8,:) im(9,:) im(10,:)...
    im(11,:) im(12,:) im(13,:) im(14,:) im(15,:)...
    im(16,:) im(17,:) im(18,:) im(19,:) im(20,:)];

im = ones(size(im)) - im;

load('ex3data1.mat');
load('ex3weights.mat');

pred = predict(Theta1, Theta2, im);

fprintf('\n %s == digit %d\n', imname, mod(pred, 10));


end