# Multilayer Perceptron (MLP) Overview

## Features
- **Suitable Problems:** Solves simple classification or regression problems, such as the XOR problem or other binary classification tasks.
- **Three-Layer Architecture:**
  - **Input Layer (1 layer):** Contains 2 neurons.
  - **Hidden Layer (1 layer):** Contains 3 neurons.
  - **Output Layer (1 layer):** Contains 1 neuron.

## Training Methodology
- **Weight Initialization:** 
  - Initial weights are set for `weights_IH` (input to hidden layer), `weights_HO` (hidden layer to output layer), and `weights_IO` (skip connection).
- **Stepwise Training:** 
  - Trains one sample at a time (`current_input`), computes the corresponding output (`current_output`), and updates weights.
- **Feedforward:**
  - Uses the **Sigmoid activation function** for non-linear transformation in the hidden layer.
- **Backpropagation:**
  - Computes output and hidden layer errors.
  - Updates weights using gradients.
- **Skip Connection:**
  - Directly connects the input layer to the output layer, potentially speeding up training.
- **Incremental Updates:**
  - Processes and updates weights after each input-output pair.

## Training Explanation

### **Feedforward**
- **Data Flow:** Input data flows from the input layer through the hidden layer, transformed by the activation function, and is passed to the output layer.
- **Hidden Layer Input:** Calculated as the weighted sum of inputs and `weights_IH`, then transformed by the **Sigmoid activation function**.
- **Output Layer Input:** Combines:
  - Weighted output of the hidden layer and `weights_HO`.
  - Weighted input from the skip connection (`weights_IO`).

### **Activation Function**
- Uses the **Sigmoid activation function** for non-linear transformation in the hidden layer.

### **Backpropagation**
1. **Error Calculation:** 
   - Computes the error at the output layer.
   - Propagates the error backward to the hidden layer.
2. **Weight Update:** 
   - Adjusts weights (`weights_IH`, `weights_HO`, and `weights_IO`) based on gradients.
   - Uses a learning rate (`lr = 0.2`) to control the update step size.

### **Skip Connection**
- Direct connection between the input layer and the output layer via `weights_IO`, enabling the output layer to rely on inputs and hidden layer outputs simultaneously.

### **Weight Constraints**
- Certain weights (e.g., `weights_IH(1,3)` and `weights_IH(2,1)`) are set to zero, indicating that specific inputs do not influence certain hidden neurons.

## Key Concepts Summary
- **Three-layer structure:** Input layer → Hidden layer → Output layer.
- **Feedforward and backpropagation:** Core methodologies used for training.
- **Skip connection:** Enhances gradient flow and potentially accelerates training.
- **Weight constraints:** Introduces sparsity in connections for focused learning.
