clear all;
clc;

% Parameters
NIPT = 2;   % Number of neurons in the input layer
NHID = 3;   % Number of neurons in the hidden layer
NOUT = 1;   % Number of neurons in the output layer
lr = 0.2;   % Learning rate

% Training Vectors for both sets of inputs and outputs
inputs = [0 1; 1 0];
desired_outputs = [1; 1];

% Define Sigmoid function
sigmoid = @(x) 1.0 ./ (1.0 + exp(-x));

% Initialize Weights
w = [0.1, 0.2, 0.2, -0.1, 0.2, 0.1, -0.3, 0.2, -0.1];

% Weights from Input layer to Hidden layer
weights_IH = zeros(NIPT, NHID);
weights_IH(:, 1) = [w(1); 0];  
weights_IH(:, 2) = [w(3); w(4)];
weights_IH(:, 3) = [0; w(6)];

% Weights from Hidden layer to Output layer
weights_HO = [w(9); w(8); w(7)];

% Weights from Input layer to Output layer
weights_IO = [w(2); w(5)];

% Store original weights for comparison
original_weights_IH = weights_IH;
original_weights_IO = weights_IO;
original_weights_HO = weights_HO;

% Online Learning Training process for both sets of inputs
for phase = 1:2
    current_input = inputs(phase, :);
    current_output = desired_outputs(phase);

    % Initialize and calculate hidden layer outputs
    hidden_outputs = zeros(NHID, 1);
    for j = 1:NHID
        activation_sum = 0;
        for i = 1:NIPT
            activation_sum = activation_sum + weights_IH(i, j) * current_input(i);
        end
        hidden_outputs(j) = sigmoid(activation_sum);
    end

    % Weighted sum from hidden layer to output layer
    weighted_sum_output = 0;
    for j = 1:NHID
        weighted_sum_output = weighted_sum_output + hidden_outputs(j) * weights_HO(j);
    end

    % Weighted sum from input layer directly to output layer
    direct_sum_output = 0;
    for i = 1:NIPT
        direct_sum_output = direct_sum_output + weights_IO(i) * current_input(i);
    end

    % Total output calculation
    total_sum_output = weighted_sum_output + direct_sum_output;

    % Calculate output error
    output_error = current_output - total_sum_output;

    % Calculate errors for hidden layer
    hidden_errors = zeros(NHID, 1);
    for j = 1:NHID
        hidden_errors(j) = output_error * weights_HO(j) * hidden_outputs(j) * (1 - hidden_outputs(j));
    end

    % Update weights
    for i = 1:NIPT
        for j = 1:NHID
            weights_IH(i, j) = weights_IH(i, j) + lr * hidden_errors(j) * current_input(i);
        end
    end
    for i = 1:NIPT
        weights_IO(i) = weights_IO(i) + lr * output_error * current_input(i);
    end
    for j = 1:NHID
        weights_HO(j) = weights_HO(j) + lr * output_error * hidden_outputs(j);
    end
    
    weights_IH(1,3) = 0;
    weights_IH(2,1) = 0;
    
    % Display weight updates
    if phase == 2
        disp(['Phase ' num2str(phase) ' Weight Updates:']);
        disp(['w1 : Original = ' num2str(round(original_weights_IH(1, 1), 4), '%.4f') ', Updated = ' num2str(round(weights_IH(1, 1), 4), '%.4f')]);
        disp(['w2 : Original = ' num2str(round(original_weights_IO(1), 4), '%.4f') ', Updated = ' num2str(round(weights_IO(1), 4), '%.4f')]);
        disp(['w3 : Original = ' num2str(round(original_weights_IH(1, 2), 4), '%.4f') ', Updated = ' num2str(round(weights_IH(1, 2), 4), '%.4f')]);
        disp(['w4 : Original = ' num2str(round(original_weights_IH(2, 2), 4), '%.4f') ', Updated = ' num2str(round(weights_IH(2, 2), 4), '%.4f')]);
        disp(['w5 : Original = ' num2str(round(original_weights_IO(2), 4), '%.4f') ', Updated = ' num2str(round(weights_IO(2), 4), '%.4f')]);
        disp(['w6 : Original = ' num2str(round(original_weights_IH(2, 3), 4), '%.4f') ', Updated = ' num2str(round(weights_IH(2, 3), 4), '%.4f')]);
        disp(['w7 : Original = ' num2str(round(original_weights_HO(3), 4), '%.4f') ', Updated = ' num2str(round(weights_HO(3), 4), '%.4f')]);
        disp(['w8 : Original = ' num2str(round(original_weights_HO(2), 4), '%.4f') ', Updated = ' num2str(round(weights_HO(2), 4), '%.4f')]);
        disp(['w9 : Original = ' num2str(round(original_weights_HO(1), 4), '%.4f') ', Updated = ' num2str(round(weights_HO(1), 4), '%.4f')]);
    end
end
