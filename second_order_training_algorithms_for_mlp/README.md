# Implementation of Newton's Algorithm for Training MLP Neural Networks

This project implements Newton's algorithm to train Multi-Layer Perceptron (MLP) neural networks, leveraging approximate Hessian matrices and second-order derivatives with respect to weights. The algorithm's performance is evaluated on both a simple neural network and a real-world sunspot series dataset.

---

## Algorithm Overview

Newton's algorithm is a second-order optimization method that updates the weights as follows:

\[
w_{new} = w_{old} - H^{-1} \nabla L(w)
\]

Where:
- \( w_{new} \): Updated weights.
- \( w_{old} \): Current weights.
- \( H \): Approximate Hessian matrix.
- \( \nabla L(w) \): Gradient of the loss function with respect to weights.

Key steps in the implementation:
1. Compute the gradient of the loss function.
2. Approximate the Hessian matrix.
3. Update weights using the formula above.

---

## Experiments

### Simple Neural Network
Newton's Algorithm with Exact Hessian Matrix in Skip Connection

### Fully Connected Neural Network
A network with the following architecture is trained on the sunspot series dataset:
- **10 input nodes**
- **5 hidden nodes**

---

## Comparisons with Other Algorithms

### Methods Compared
1. **Classic Backpropagation Algorithm with Gradient Descent**
2. **Newton's Algorithm with Approximate Hessian Matrix**

### Evaluation Metric
Performance is measured using **Normalized Mean Squared Error (NMSE)**.

### Results Visualization
All three methods' approximations of the sunspot series dataset are plotted together to illustrate their differences.

---

## References

1. **Sunspot Series Dataset**:  
   [https://www.cs.cmu.edu/afs/cs/academic/class/15782-f06/matlab/] by Dave Touretzky (modified by Nikolay Nikolaev)
