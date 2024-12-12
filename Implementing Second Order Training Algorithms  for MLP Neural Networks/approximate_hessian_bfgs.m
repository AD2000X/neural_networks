clear all;
clc;

% Approximate Hessian (BFGS)

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

% parameters for the neural network
idim = 10;  % input dimension (number of lagged values used as input)
odim = length(relNums) - idim;  % output dimension (number of predicted values)
x = zeros(odim, idim);  % input data matrix
y = zeros(odim, 1); % target output vector

% create inputs and desired outputs
for i = 1:odim
    y(i) = relNums(i + idim);   % target output is the next value after the input sequence
    x(i, :) = relNums(i:i+idim-1)'; % input sequence is a column vector of lagged values
end

% network parameters
NIPT = idim;    % number of input neurons
NHID = 5;   % number of hidden neurons
NOUT = 1;   % number of output neurons
lr = 0.01;  % learning rate

% activation functions and derivatives
leaky_relu = @(x) (x > 0) .* x + (x <= 0) .* 0.01 .* x; % leaky ReLU activation function
leaky_relu_derivative = @(x) (x > 0) + (x <= 0) * 0.01; % derivative of leaky ReLU
linear_activation = @(x) x; % linear activation function for output layer

% initialize weights randomly
weights_IH = randn(NIPT, NHID); % weights from input to hidden layer
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
        
        % combine weight gradients between different layers to a vector
        [loss_value, grad] = compute_loss(weights_IH, weights_HO, x(i, :), y(i), NIPT, NHID, NOUT);

        % gradient clipping
        grad_clip_threshold = 1;     % threshold for gradient clipping
        % extracts the gradient components from the total gradient vector that correspond to the weights between:
        % the input layer and the hidden layer / the hidden layer and the output layer
        % which are used to update these weights
        grad_weights_IH = grad(1:NIPT*NHID);
        grad_weights_HO = grad(NIPT*NHID+1:end);
        grad_weights_IH = max(min(grad_weights_IH, grad_clip_threshold), -grad_clip_threshold); % clip gradients for weights_IH
        grad_weights_HO = max(min(grad_weights_HO, grad_clip_threshold), -grad_clip_threshold); % clip gradients for weights_HO

        % combine weight gradients between different layers to a vector
        grad = [grad_weights_IH; grad_weights_HO];

        % update weights by using BFGS
        weights = [weights_IH(:); weights_HO(:)];   % combine weights into a single vector
        % objective function for BFGS
        obj_fun = @(w) compute_loss(reshape(w(1:NIPT*NHID), NIPT, NHID), reshape(w(NIPT*NHID+1:end), NHID, NOUT), x(i, :), y(i), NIPT, NHID, NOUT);
        weights = bfgs(obj_fun, weights, grad, 10); % update weights using BFGS
                
        % rebuild the weight matrix for each layer from the updated weight vector
        weights_IH = reshape(weights(1:NIPT*NHID), NIPT, NHID); % extract weights for input-hidden layer
        weights_HO = reshape(weights(NIPT*NHID+1:end), NHID, NOUT); % extract weights for hidden-output layer
    end
end

% calculate NMSE
mean_y = mean(y);
nmse_loss = sum((y - network_output).^2) / sum((y - mean_y).^2);    % normalized mean squared error
disp(['NMSE Loss: ', num2str(nmse_loss)]);

% plot results
figure;
plot(1:odim, y, 'b', 'LineWidth', 2);   % plot actual values in blue
hold on;
plot(1:odim, network_output, 'r--', 'LineWidth', 2);    % plot predicted values in red dashed line
hold off;
xlabel('Time Step');
ylabel('Normalized Sunspot Number');
legend('Actual Data', 'Predicted Data');
title('Sunspot Series: Actual vs. Predicted');
grid on;

% loss calculation function
function [loss_value, grad] = compute_loss(weights_IH, weights_HO, x, y, NIPT, NHID, NOUT)
    leaky_relu = @(x) (x > 0) .* x + (x <= 0) .* 0.01 .* x; % leaky ReLU activation function
    leaky_relu_derivative = @(x) (x > 0) + (x <= 0) * 0.01; % derivative of leaky ReLU
    linear_activation = @(z) z; % linear activation function for output layer
    
    % feedforward
    hidden_input = x * weights_IH;  % input to hidden layer
    hidden_output = leaky_relu(hidden_input);   % output of hidden layer
    output_input = hidden_output * weights_HO;  % input to output layer
    network_output = linear_activation(output_input);   % output of the network
    
    % calculate MSE
    loss_value = mean((y - network_output).^2);
    
    % Backpropagation
    % calculate the error of the output layer
    delta_output = -(y - network_output);
    % calculate the error of the hidden layer by the error of backpropagation of the output layer
    delta_hidden = (delta_output * weights_HO') .* leaky_relu_derivative(hidden_input);
    % calculate the gradient from the input to the hidden layer weight by the hidden layer error
    grad_weights_IH = x' * delta_hidden;
    % calculate the gradient from the hidden to the output layer weight by the error of the output layer
    grad_weights_HO = hidden_output' * delta_output;
    
    % combine weight gradients between different layers to a vector
    grad = [grad_weights_IH(:); grad_weights_HO(:)];
end

% BFGS (Broyden-Fletcher-Goldfarb-Shanno) function
function weights = bfgs(f, weights, grad, max_iter)
    n = numel(weights); % get the number of elements of the weight vector
    H = eye(n); % initialize the approximate Hessian matrix to an identity matrix

    for iter = 1:max_iter   % maximum iterations
        d = -H * grad;  % calculate search direction
        step_size = line_search(f, weights, d, grad); % use linear search to determine the appropriate step size
        weights_new = weights + step_size * d;  % update weight

        [loss_new, grad_new] = f(weights_new);  % calculate the loss and gradient of the new weights
        s = weights_new - weights;  % weight change amount
        y = grad_new - grad;    % gradient change amount

        if all(abs(y) < 1e-8) % convergence check for whether the gradient change is too small toend the iteration early
            break;  % exit the loop if convergence is reached
        end

        rho = 1 / (y' * s); % calculate the scaling factor
        % update the approximate Hessian matrix
        H = (eye(n) - rho * (s * y')) * H * (eye(n) - rho * (y * s')) + rho * (s * s');

        weights = weights_new;  % save the new weights
        grad = grad_new;    % update gradient
    end
end

% Linear search function
function step_size = line_search(f, x, d, grad)
    alpha = 0.1;    % initial step size
    beta = 0.5; % step size reduction factor
    t = 1;  % set initial trial step size as 1
    
    % determine the step size is small enough to ensure "f" decreases 
    % engough along the direction of the gradient descent "d"
    while f(x + t*d) > f(x) + alpha*t*grad'*d
        t = beta * t;   % reduce step size
    end
    
    step_size = t;  % determine the step size
end
