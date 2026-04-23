# FNN: A Feedforward Neural Network in C++

## Overview

A lightweight, header-only feedforward neural network implementation in C++ using Eigen for matrix operations. Supports both regression and classification tasks with mini-batch SGD training.

## Algorithm

### Forward Propagation

For a network with $L$ layers, the forward pass computes:

$$\mathbf{a}^{(l)} = f^{(l)}(\mathbf{z}^{(l)})$$

where $\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$

- **Linear layer**: $\mathbf{z} = \mathbf{X} \mathbf{W} + \mathbf{b}$ (row-wise broadcasting)
- **ReLU activation**: $f(x) = \max(0, x)$
- **Softmax**: $p_i = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_j e^{z_j - \max(\mathbf{z})}}$

### Backward Propagation

Gradients are computed via the chain rule, propagating error signals from output to input:

**Linear Layer:**
$$\frac{\partial L}{\partial \mathbf{W}} = (\mathbf{A}^{(l-1)})^T \frac{\partial L}{\partial \mathbf{Z}^{(l)}}$$
$$\frac{\partial L}{\partial \mathbf{b}} = \sum_{batch} \frac{\partial L}{\partial \mathbf{Z}^{(l)}}$$
$$\frac{\partial L}{\partial \mathbf{A}^{(l-1)}} = \frac{\partial L}{\partial \mathbf{Z}^{(l)}} (\mathbf{W}^{(l)})^T$$

**ReLU Layer:**
$$\frac{\partial L}{\partial \mathbf{Z}^{(l-1)}} = \frac{\partial L}{\partial \mathbf{A}^{(l-1)}} \odot \mathbb{1}(\mathbf{Z}^{(l-1)} > 0)$$

**Loss Gradients:**
- **MSE (Regression)**: $\frac{\partial L}{\partial \hat{\mathbf{y}}} = \hat{\mathbf{y}} - \mathbf{y}$
- **Cross-Entropy + Softmax (Classification)**: $\frac{\partial L}{\partial \mathbf{z}} = \text{softmax}(\mathbf{z}) - \mathbf{y}_{onehot}$

### Parameter Update

Mini-batch SGD with fixed learning rate $\eta$:

$$\mathbf{W} \leftarrow \mathbf{W} - \eta \cdot \frac{1}{m} \sum \frac{\partial L}{\partial \mathbf{W}}$$
$$\mathbf{b} \leftarrow \mathbf{b} - \eta \cdot \frac{1}{m} \sum \frac{\partial L}{\partial \mathbf{b}}$$

## Experiments

### Regression: California Housing

| Configuration | Value |
|:---|:---|
| Architecture | 8 → 64 → 32 → 1 |
| Epochs | 300 |
| Learning Rate | 0.001 |
| Batch Size | 64 |

**Results:**
| Metric | Value |
|:---|:---|
| Train MSE | 0.2498 |
| Test MSE | 0.2815 |

### Classification: MNIST

| Configuration | Value |
|:---|:---|
| Architecture | 784 → 256 → 128 → 10 |
| Epochs | 50 |
| Learning Rate | 0.01 |
| Batch Size | 64 |

**Results:**
| Metric | Value |
|:---|:---|
| Train Accuracy | 99.95% |
| Test Accuracy | 97.65% |
| Training Time | 617s |

## Usage

```cpp
#include "FNN.hpp"

// Regression
FNN net;
net.add(new Linear(8, 64));
net.add(new ReLU());
net.add(new Linear(64, 1));

net.fit_regression(X_train, y_train, 100, 0.001, 64);
vector<double> pred = net.predict_regression(X_test);

// Classification
FNN cls;
cls.add(new Linear(784, 256));
cls.add(new ReLU());
cls.add(new Linear(256, 10));
cls.add(new softmax());

cls.fit_classify(X_train, y_train, 10, 50, 0.01, 64);
vector<int> pred = cls.predict_classify(X_test);
vector<Point> proba = cls.predict_proba(X_test);
```

## Limitations

- No regularization (L1/L2, Dropout)
- No batch normalization
- Fixed learning rate (no decay or adaptive methods)
- Single-threaded CPU execution
- Row-vector convention assumed throughout; broadcasting behavior relies on Eigen's type deduction

**Note:** This implementation uses `using namespace std;` and `using namespace Eigen;` globally in the header. Users should be aware of potential namespace collisions when integrating with other libraries.

## References

- Li, Hang. *Machine Learning Methods*. Deep Learning Chapter.
- Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks.
- He, K., et al. (2015). Delving deep into rectifiers.
