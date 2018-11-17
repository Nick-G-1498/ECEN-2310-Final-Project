% Hello World -> Deep Learning Library

cam = webcam(1); %Connect to the Camera
net = alexnet; % Load the Neural Network

while true
    % have to hit CTR-C to Exit the Loop and have to clear the workspace
    % every time
    
    im = snapshot(cam);       % Take a picture
    image(im);                   % Show the picture
    im = imresize(im,[227 227]); % Resize the picture for alexnet
    label = classify(net,im);    % Classify the picture
    object = char(label);        % Save what alexnet classifies it as
    switch lower(object)         % Decide whether it is a hot dog 
                                 % or not
        case 'hot dog'
            title(char(label)); 
        otherwise
            title('Not hot dog');
    end
    drawnow                     % display the webcam w/ the class 
end
