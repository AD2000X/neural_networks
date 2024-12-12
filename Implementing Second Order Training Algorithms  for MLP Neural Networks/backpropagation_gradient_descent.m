clear all;
clc;

% Backpropagation (Gradient Descent)

% load and preprocess(normalize the data) the sunspot dataset
load sunspot.dat
% the first column is the year, stored in the variable year
% the second column is the associated sunspot number, stored in the variable relNums.
year=sunspot(:,1); relNums=sunspot(:,2); 
% calculate the mean (ynrmv) and standard deviation (sigy) of relNums. 
% the mean is for subsequent centering (demeaning), and the standard deviation is for subsequent normalization.
ynrmv=mean(relNums(:)); sigy=std(relNums(:)); 
% using calculated the mean and standard deviationin in the previous step  
% to normalize the raw sunspot data by subtracting the mean and 
% dividing by the standard deviation from each data point. 
nrmY=(relNums(:)-ynrmv)./sigy; 
% calculate the minimum (ymin) and maximum (ymax) values ​​of the normalized data nrmY, 
% then applied to linearly transform the data into the range [-1, 1].
ymin=min(nrmY(:)); ymax=max(nrmY(:)); 
% the normalized data nrmY is further transformed into the range [-1, 1]. 
% first normalize each value of nrmY to [0, 1] by subtracting ymin and 
% dividing by the range (ymax - ymin), 
% then adjust the range to [-1, 1] by multiplying by 2.0 and subtracting 0.5.
relNums=2.0*((nrmY-ymin)/(ymax-ymin)-0.5);

% create the data matrix with lagged values
odim = length(relNums) - idim;  % output dimension (number of predicted values)
x = zeros(odim, idim);  % input data matrix
y = zeros(odim, 1); % target output vector

% create inputs and desired outputs
for i = 1:odim
    y(i) = relNums(i + idim);   % target output is the next value after the input sequence
    x(i, :) = relNums(i:i+idim-1)'; % input sequence is a column vector of lagged values
end

% neural network parameters
idim = 10;  % number of input neurons
NHID = 5;   % number of hidden neurons
NOUT = 1;   % number of output neurons
lr = 0.01;  % learning rate

% define activation functions
leaky_relu = @(x) (x > 0) .* x + (x <= 0) .* 0.01 .* x; leaky ReLU activation function
leaky_relu_derivative = @(x) (x > 0) + (x <= 0) * 0.01; % derivative of leaky ReLU
linear_activation = @(x) x; % linear activation function for output layer

% initialize weights randomly
weights_IH = randn(idim, NHID); % weights from input to hidden layer
weights_HO = randn(NHID, NOUT); % weights from hidden to output layer

% neural network forward and backward propagation
network_output = zeros(odim, 1);    % predicted output values
num_epochs = 100;   % number of training epochs

% main training loop
for epoch = 1:num_epochs
    for i = 1:odim

        % feedforward
        hidden_input = x(i, :) * weights_IH;    % input to hidden layer
        hidden_output = leaky_relu(hidden_input);   % output of hidden layer
        output_input = hidden_output * weights_HO;  % input to output layer
        network_output(i) = linear_activation(output_input);    % output of the network

        % backpropagation
        delta_output = -(y(i) - network_output(i)); % output layer error
        delta_hidden = (delta_output * weights_HO') .* leaky_relu_derivative(hidden_input); % hidden layer error
        grad_weights_IH = x(i, :)' * delta_hidden;  % gradient of weights from input to hidden layer
        grad_weights_HO = hidden_output' * delta_output;    % gradient of weights from hidden to output layer

        % gradient clipping
        grad_clip_threshold = 1;    % threshold for gradient clipping
        grad_weights_IH = max(min(grad_weights_IH, grad_clip_threshold), -grad_clip_threshold); % clip gradients for weights_IH
        grad_weights_HO = max(min(grad_weights_HO, grad_clip_threshold), -grad_clip_threshold); % clip gradients for weights_HO

        % update weights (gradient descent)
        weights_IH = weights_IH - lr * grad_weights_IH; % update weights_IO
        weights_HO = weights_HO - lr * grad_weights_HO; % update weights_HO
    end
end

% calculate and display NMSE loss
mean_y = mean(y);
nmse_loss = sum((y - network_output).^2) / sum((y - mean_y).^2);    % normalized mean squared error
disp(['NMSE Loss: ', num2str(nmse_loss)]);

% plot the actual vs predicted data
figure;
plot(1:length(y), y, 'b', 'LineWidth', 2);  % plot actual values in blue
hold on;
plot(1:length(network_output), network_output, 'r--', 'LineWidth', 2);  % plot predicted values in red dashed line
hold off;
title('Sunspot Series: Actual vs. Predicted');
xlabel('Time Steps');
ylabel('Normalized Sunspot Number');
legend('Actual', 'Predicted');
