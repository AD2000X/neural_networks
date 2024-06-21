clear all;
clc;

% Exact Hessian in Skip Connection

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
% calculate the minimum (ymin) and maximum (ymax) values of the normalized data nrmY, 
% then applied to linearly transform the data into the range [-1, 1].
ymin=min(nrmY(:)); ymax=max(nrmY(:)); 
% the normalized data nrmY is further transformed into the range [-1, 1]. 
% first normalize each value of nrmY to [0, 1] by subtracting ymin and 
% dividing by the range (ymax - ymin), 
% then adjust the range to [-1, 1] by multiplying by 2.0 and subtracting 0.5.
relNums=2.0*((nrmY-ymin)/(ymax-ymin)-0.5);

% parameters for the neural network
idim = 2;   % input dimension (number of lagged values used as input)
odim = length(relNums) - idim;  % output dimension (number of predicted values)
x = zeros(odim, idim);  % input data matrix
y = zeros(odim, 1); % target output vector

% create inputs and desired outputs
for i = 1:odim
y(i) = relNums(i + idim);    % target output is the next value after the input sequence
x(i, :) = relNums(i:i+idim-1)';  % input sequence is a column vector of lagged values
end

% network parameters
NIPT = idim;    % number of input neurons
NHID = 3;   % number of hidden neurons
NOUT = 1;   % number of output neurons
lr = 0.001; % learning rate

% activation functions and derivatives
leaky_relu = @(x) (x > 0) .* x + (x <= 0) .* 0.01 .* x; % leaky ReLU activation function
leaky_relu_derivative = @(x) (x > 0) + (x <= 0) * 0.01; % derivative of leaky ReLU
linear_activation = @(x) x; % linear activation function for output layer

% initialize Weights
w = [-0.25, 0.33, 0.14, -0.17, 0.16, 0.43, 0.21, 0.25];

% weights from Input layer to Hidden layer
weights_IH = [w(1), 0, w(3);    % connection from x1 to y1, y2 (none), and y3
0, w(4), w(5)];   % connection from x2 to y1 (none), y2, and y3

% weights from Hidden layer to Output layer
weights_HO = [w(6); w(7); w(8)];    % connections from y1, y2, y3 to output y

% weights from Input layer to Output layer (Skip connection)
weights_IO = w(2); % Direct connection from x1 to output y

% initialize network output
network_output = zeros(odim, 1);    % predicted output values

% train the network
num_epochs = 100;   % number of training epochs
for epoch = 1:num_epochs
for i = 1:odim
% feedforward
hidden_input = x(i, :) * weights_IH;    % input to hidden layer
hidden_output = leaky_relu(hidden_input);   % output of hidden layer
output_input = hidden_output * weights_HO + weights_IO * x(i, 1);   % input to output layer
network_output(i) = linear_activation(output_input);    % output of the network

% backpropagation
delta_output = -(y(i) - network_output(i)); % output layer error
delta_hidden = (delta_output * weights_HO') .* leaky_relu_derivative(hidden_input); % hidden layer error
grad_weights_IH = x(i, :)' * delta_hidden;  % gradient of weights from input to hidden layer
grad_weights_HO = hidden_output' * delta_output;    % gradient of weights from hidden to output layer
grad_weights_IO = x(i, 1) * delta_output;   % gradient of weights for skip connection

% gradient clipping
grad_clip_threshold = 1;     % threshold for gradient clipping
grad_weights_IH = max(min(grad_weights_IH, grad_clip_threshold), -grad_clip_threshold); % clip gradients for weights_IH
grad_weights_HO = max(min(grad_weights_HO, grad_clip_threshold), -grad_clip_threshold); % clip gradients for weights_HO
grad_weights_IO = max(min(grad_weights_IO, grad_clip_threshold), -grad_clip_threshold); % clip gradients for weights_IO

% calculate Hessian with regularization
weights = [weights_IH(:); weights_HO; weights_IO];  % combie all weights into a single vector
% compute Hessian matrix with regularization
hessian = compute_hessian(weights, x(i,:), y(i), leaky_relu, leaky_relu_derivative, linear_activation) + 1e-3 * eye(numel(weights));
hessian_IH = hessian(1:6, 1:6); % extract Hessian for weights_IH
hessian_HO = hessian(7:9, 7:9); % extract Hessian for weights_HO
hessian_IO = hessian(10, 10);   % extract Hessian for weights_IO

% update weights
lambda = 1e-1;  % regularization parameter
update_IH = pinv(hessian_IH + lambda * eye(size(hessian_IH))) * grad_weights_IH(:); % update for weights_IH
update_HO = pinv(hessian_HO + lambda * eye(size(hessian_HO))) * grad_weights_HO;    % update for weights_HO
update_IO = (hessian_IO + lambda) \ grad_weights_IO;    % update for weights_IO

% update clipping
update_clip_threshold = 1;  % threshold for update clipping
update_IH = max(min(update_IH, update_clip_threshold), -update_clip_threshold); % clip updates for weights_IH
update_HO = max(min(update_HO, update_clip_threshold), -update_clip_threshold); % clip updates for weights_HO
update_IO = max(min(update_IO, update_clip_threshold), -update_clip_threshold); % clip updates for weights_IO

% upply updates
weights_IH = weights_IH - lr * reshape(update_IH, size(weights_IH));    % update weights_IH
weights_HO = weights_HO - lr * update_HO;   % update weights_HO
weights_IO = weights_IO - lr * update_IO;   % update weights_IO

% calculate loss with regularization
[loss_value, ~] = compute_loss(weights_IH, weights_HO, weights_IO, x(i, :), y(i), lambda, leaky_relu, leaky_relu_derivative, linear_activation);

% check for NaN or Inf loss
if isnan(loss_value) || isinf(loss_value)
disp('Loss value becomes NaN or Inf. Training aborted.');
break;
end
end
end

% calculate NMSE
mean_y = mean(y);
nmse_loss = sum((y - network_output).^2) / sum((y - mean_y).^2);    % normalized mean squared error
disp(['NMSE Loss: ', num2str(nmse_loss)]);

% plot results
figure;
plot(1:length(y), y, 'b', 'LineWidth', 2);  % plot actual values in blue
hold on;
plot(1:length(network_output), network_output, 'r--', 'LineWidth', 2);  % plot predicted values in red dashed line
hold off;
title('Sunspot Series: Actual vs. Predicted');
xlabel('Time Steps');
ylabel('Normalized Sunspot Number');
legend('Actual', 'Predicted');

% Hessian calculation with regularization
function H = compute_hessian(weights, x, y, leaky_relu, leaky_relu_derivative, linear_activation)
eps = 1e-6; % perturbation size for numerical approximation to slightly perturb the weights
num_weights = numel(weights);   % total number of weights
H = zeros(num_weights, num_weights);     % initialize Hessian matrix

% computes the gradient of the loss function without regularization
% use to calculate the Hessian matrix elements
[~, grad] = compute_loss(reshape(weights(1:6), 2, 3), weights(7:9), weights(10), x, y, 0, leaky_relu, leaky_relu_derivative, linear_activation);

% approximate second-order derivatives based on the neural network Hessian matrix
for i = 1:num_weights
for j = 1:num_weights
% perturb the i-th weight
weights_perturbed_i = weights;  
weights_perturbed_i(i) = weights_perturbed_i(i) + eps;
% calculate the gradient of the loss function after the i-th perturbation weight
[~, grad_perturbed_i] = compute_loss(reshape(weights_perturbed_i(1:6), 2, 3), weights_perturbed_i(7:9), weights_perturbed_i(10), x, y, 0, leaky_relu, leaky_relu_derivative, linear_activation);

% perturb the j-th weight
weights_perturbed_j = weights;
weights_perturbed_j(j) = weights_perturbed_j(j) + eps;
% calculate the gradient of the loss function after the j-th perturbation weight
[~, grad_perturbed_j] = compute_loss(reshape(weights_perturbed_j(1:6), 2, 3), weights_perturbed_j(7:9), weights_perturbed_j(10), x, y, 0, leaky_relu, leaky_relu_derivative, linear_activation);

% perturb i and j weights at the same time
weights_perturbed_ij = weights;
weights_perturbed_ij(i) = weights_perturbed_ij(i) + eps;
weights_perturbed_ij(j) = weights_perturbed_ij(j) + eps;
% calculate the gradient after simultaneous perturbation
[~, grad_perturbed_ij] = compute_loss(reshape(weights_perturbed_ij(1:6), 2, 3), weights_perturbed_ij(7:9), weights_perturbed_ij(10), x, y, 0, leaky_relu, leaky_relu_derivative, linear_activation);

% approximate second-order derivatives for weights(i) and (j)
H(i, j) = (grad_perturbed_ij(i) - grad_perturbed_i(i) - grad_perturbed_j(i) + grad(i)) / (eps^2);
end
end
end

% Loss calculation with regularization
% calcualte 3 layers and 1 skip connection weights, input X and y, lambda,
% and all acticvation function
function [loss_value, grad] = compute_loss(weights_IH, weights_HO, weight_IO, x, y, lambda, leaky_relu, leaky_relu_derivative, linear_activation)

% feedforward
hidden_input = x * weights_IH;  % input to hidden layer
hidden_output = leaky_relu(hidden_input);   % output of hidden layer
output_input = hidden_output * weights_HO + weight_IO * x(1);   % input to output layer
network_output = linear_activation(output_input);   % output of the network

% loss calculation with regularization
% mean squared error with L2 regularization
loss_value = mean((y - network_output).^2) + lambda * (sum(weights_IH(:).^2) + sum(weights_HO(:).^2) + weight_IO^2);

% Backpropagation
% calculate the error of the output layer
% weight decay
delta_output = -(y - network_output);   % output layer error
delta_hidden = (delta_output * weights_HO') .* leaky_relu_derivative(hidden_input); % hidden layer error
grad_weights_IH = x' * delta_hidden + 2 * lambda * weights_IH;  % gradient of weights from input to hidden layer with weight decay
grad_weights_HO = hidden_output' * delta_output + 2 * lambda * weights_HO;  % gradient of weights from hidden to output layer with weight decay
grad_weights_IO = x(1) * delta_output + 2 * lambda * weight_IO; % gradient of weights for skip connection with weight decay

% get accurate first-order gradient information for the loss function with respect to each weight in the network
grad = [grad_weights_IH(:); grad_weights_HO; grad_weights_IO];
end
