% Script to save png indexed images with the colormap defined by VOCdevkit

clear all; close all; clc;

dirAddress = '/mnt/a/pathak/fcn_mil_cache/visualized_output/seg12test_size/';
inputImages = dir(fullfile(dirAddress,'results/VOC2012/Segmentation/comp6_test_cls/*.png'));

cmap = VOClabelcolormap(256);
for i=1:length(inputImages)
    if mod(i,100)==0
        fprintf('Image # %d\n',i);
    end
    im = imread(fullfile(dirAddress,'results/VOC2012/Segmentation/comp6_test_cls/',inputImages(i).name));
    imwrite(im,cmap,fullfile(dirAddress,'results/VOC2012/Segmentation/comp6_test_cls/',inputImages(i).name));
end
