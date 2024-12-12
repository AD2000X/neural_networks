# Implementation of Newton's Algorithm for Training MLP Neural Networks

This project focuses on designing and implementing Newton's algorithm to train a Multi-Layer Perceptron (MLP) neural network. The algorithm utilizes an approximate Hessian matrix and second-order derivatives with respect to the weights. The implementation includes testing with a simple neural network and evaluating performance on a real-world dataset.

---

## Algorithm Structure and Weight Update Formula

### Newton's Algorithm Overview
1. Compute the gradient of the loss function with respect to the network weights.
2. Approximate the Hessian matrix using second-order derivatives.
3. Update the weights using the formula:
   \[
   w_{new} = w_{old} - H^{-1} \nabla L(w)
   \]
   where:
   - \( w_{new} \): Updated weights.
   - \( w_{old} \): Current weights.
   - \( H \): Approximate Hessian matrix.
   - \( \nabla L(w) \): Gradient of the loss function with respect to weights.

4. Repeat the process until convergence or a predefined number of iterations is reached.

### MATLAB Implementation
The algorithm is implemented in MATLAB, and the steps include:
- Defining the neural network structure.
- Calculating gradients and approximating the Hessian.
- Updating weights using Newton's method.

---

## Experimental Setup

### Simple Neural Network Test
A simple neural network is created to verify the correctness of the Newton's algorithm implementation.

### Real-World Neural Network
A fully connected neural network is designed with:
- **10 inputs**
- **5 hidden nodes**

The network is trained on the real-world **sunspot series dataset** using the Newton's algorithm with an approximate Hessian matrix.

---

## Comparison with Other Algorithms

### Algorithms Used for Comparison
1. **Classic Backpropagation Algorithm**
2. **Newton's Algorithm with Exact Hessian Matrix**
3. **Newton's Algorithm with Approximate Hessian Matrix** (our implementation)

### Performance Metric
The performance is evaluated using the **normalized mean squared error (NMSE)**.

### Visualization
A common plot is created to illustrate the approximation of the sunspot series using the three algorithms to highlight their differences.

---

## Results and Observations
- Performance of the Newton's algorithm (approximate Hessian) is compared with the classic backpropagation algorithm and the exact Hessian version.
- Observations on convergence speed, computational cost, and accuracy are recorded.

---

## Usage Instructions

### MATLAB Implementation
1. Clone the repository.
2. Run the provided MATLAB scripts to train the neural network using different algorithms.
3. Visualize the results using the included plotting functions.

---

## Conclusion
Newton's algorithm, with its utilization of second-order derivatives, offers faster convergence in training MLP neural networks. However, its performance varies depending on the accuracy of the Hessian approximation and the dataset used.

---

## References
1. Sunspot series dataset: [Source Link]
2. Neural network theory and Newton's method: [Relevant Literature]
