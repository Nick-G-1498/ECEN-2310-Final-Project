
function [imgDataTrain, labelsTrain, imgDataTest, labelsTest] = prepData
% The purpose of this code is to prepare the data files so the hand 
% written digit classier code can work with the IDX Files that the data is 
% encoded in
%   Nick Goralka 
%   ECEN 2310 -> Final Project 
%   Last Update: 11.15.18

% Unzip Files from .gx files
cd MNISTData
gunzip *.gz

% Return to the directory
cd .. 

filePrefix = 'MNISTData';
files = {   "train-images-idx3-ubyte",
            "train-labels-idx1-ubyte",
            "t10k-images-idx3-ubyte",
            "t10k-labels-idx1-ubyte"  };     

% Open the "train-images-idx3-ubyte" file for binary read access
fileID = fopen(fullfile(filePrefix, char(files{1})),'r', 'b'); 
    
% Get all of the header data for the image training set
magicNum = fread(fileID, 1, 'uint32');
    if (magicNum ~= 2051)
       error('Known Magic Number of TRAINING SET IMAGE FILE didnt match');
    end
numImgs  = fread(fileID, 1, 'uint32');
numRows  = fread(fileID, 1, 'uint32');
numCols  = fread(fileID, 1, 'uint32');

% get/put all data into uint8 class vector
rawImgDataTrain = uint8(fread(fileID, numImgs * numRows * numCols, 'uint8'));

fclose(fileID); % close the file

% take all the raw data and put into a 4D array
rawImgDataTrain = reshape(rawImgDataTrain, [numRows, numCols, numImgs]);
rawImgDataTrain = permute(rawImgDataTrain, [2,1,3]);
imgDataTrain(:,:,1,:) = uint8(rawImgDataTrain(:,:,:));

% get the header data for the  labels
fileID = fopen(fullfile(filePrefix, char(files{2})), 'r', 'b');
magicNum  = fread(fileID, 1, 'uint32');
    if (magicNum ~= 2049)
        error('Known Magic Number of TRAINING SET LABELS FILE didnt match');
    end
    
% read the data    
numLabels = fread(fileID, 1, 'uint32');
        
% Read the data for the labels
labelsTrain = fread(fileID, numLabels, 'uint8');
fclose(fileID);


% Open the image test file for binary read access
fileID = fopen(fullfile(filePrefix, char(files{3})),'r', 'b'); 
    
% Get all of the header data for the image test set
magicNum = fread(fileID, 1, 'uint32');
    if (magicNum ~= 2051)
       error('Known Magic Number of TEST SET IMAGE FILE didnt match');
    end
numImgs  = fread(fileID, 1, 'uint32');
numRows  = fread(fileID, 1, 'uint32');
numCols  = fread(fileID, 1, 'uint32');

% get/put all data into uint8 class vector
rawImgDataTrain = uint8(fread(fileID, numImgs * numRows * numCols, 'uint8'));

fclose(fileID); % close the file

% take all the raw data and put into a 4D array
rawImgDataTrain = reshape(rawImgDataTrain, [numRows, numCols, numImgs]);
rawImgDataTrain = permute(rawImgDataTrain, [2,1,3]);
imgDataTest(:,:,1,:) = uint8(rawImgDataTrain(:,:,:));

% get the header data for the test labels
fileID = fopen(fullfile(filePrefix, char(files{4})), 'r', 'b');
magicNum  = fread(fileID, 1, 'uint32');
    if (magicNum ~= 2049)
        error('Known Magic Number of TEST SET LABELS FILE didnt match');
    end
    
% read the data    
numLabels = fread(fileID, 1, 'uint32');
        
% Read the data for the labels
labelsTest = fread(fileID, numLabels, 'uint8');
fclose(fileID);

end
