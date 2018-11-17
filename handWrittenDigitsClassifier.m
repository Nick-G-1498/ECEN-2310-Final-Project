% Nick Goralka
% ECEN 2310 -> Final Project 
% Last Update: 11.15.18

    
% The purpose of this code is to train and test a handwritten digit image
% classifier

[imgDataTrain, labelsTrain, imgDataTest, labelsTest] = prepData;
    % load the data from the prepData script
    % MNIST testing images in a 28x28x10000 double array (28x28 is height
    % & width of an image, and there are 10,000 images)
    
% display some of the images -> mainly for debuggin purposes    
%   perm = randperm(numel(labelsTrain), 25); % get 25 unique values
%   subset = imgDataTrain(:,:,1,perm); % get a subset of 25 unique images
%   montage(subset) % display

[imgDataTrain,labelsTrain] = digitTrain4DArrayData;
    % digitTrain4DArrayData loads the digit training set as 4-D array data
    % imgDataTrain => 28 pixcels x 28 pixcels x 4000 images
    % lablesTrain  => 4000 labels

% format data for the learning algorithum
idx = randperm(size(imgDataTrain,4),1000); % leave some data for validation
XValidation = imgDataTrain(:,:,:,idx); 
imgDataTrain(:,:,:,idx) = [];
YValidation = labelsTrain(idx);
labelsTrain(idx) = [];


% CNN I found online
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same') 
    batchNormalizationLayer
    reluLayer   
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];


% traing options ripped directly from doccumentation
options = trainingOptions('sgdm', ...
    'MaxEpochs',8, ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

% train the network
net = trainNetwork(imgDataTrain,labelsTrain,layers,options); 

% test the data 
[imgDataTest,labelsTest]= digitTest4DArrayData;
predLabelsTest = net.classify(imgDataTest);
accuracy = sum(predLabelsTest == labelsTest)/numel(labelsTest);

% print the accuracy of the network
fprintf('Test Accuracy => %.1f%% \n \n',100*accuracy);


