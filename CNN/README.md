# Simple_CNN: A Lightweight Handwritten CNN

A minimal yet complete implementation of a Convolutional Neural Network (CNN) from scratch using C++ and Eigen. Designed for educational purposes, following the principles of **Chapter 26 of Li Hang's "Machine Learning Methods"**.

---

## Algorithm Overview

This implementation follows the standard CNN pipeline:

1. **Convolution** – Extracts spatial features using learnable kernels.
2. **Activation (ReLU)** – Introduces non‑linearity.
3. **Pooling (MaxPooling)** – Reduces spatial dimensions and provides translation invariance.
4. **Flatten** – Converts 3D feature maps into 1D vectors for the fully connected layers.
5. **Fully Connected Layers** – Perform high‑level reasoning and classification.
6. **Softmax** – Outputs class probabilities.

**Backpropagation** computes gradients of the cross‑entropy loss with respect to all parameters (convolutional kernels and linear layer weights), using the chain rule. **Mini‑batch stochastic gradient descent (SGD)** updates the parameters.

The loss is calculated implicitly within the forward pass, and the gradient for the final Softmax layer is fused with the loss computation to avoid a separate one‑hot encoding step, improving memory and speed.

---

## Class Design

### Layer Hierarchy

- `Layer` – Abstract base class with `forward()`, `backward()`, and `update_params()`.
- Concrete layers: `Conv`, `ReLu`, `MaxPooling`, `MeanPooling`, `Flatten`, `Linear`, `Softmax`, `ResidualUnit`.
- Each layer stores its own input cache (for backpropagation) and learnable parameters (where applicable).

### `Simple_CNN` Encapsulation

- Constructs a fixed CNN architecture suitable for small square images (e.g., 28×28 MNIST).
- Handles training (`fit`) and inference (`predict`).
- Manages memory: destructor deallocates all layer objects.

### Key Optimizations

- **No separate one‑hot encoding** – The gradient for Softmax + Cross‑Entropy is computed in‑place during loss calculation.
- **Efficient memory usage** – Uses `std::vector::reserve` for batch construction.
- **Minimal temporary objects** – Reuses tensors where possible.

---

## Usage

### Training

```cpp
Simple_CNN cnn(28, 1);          // 28x28 single‑channel images
cnn.fit(X_train, y_train,       // dataset
        10,                     // number of classes
        5,                      // epochs
        0.001,                  // learning rate
        32,                     // batch size
        true);                  // verbose output
```

### Prediction

```cpp
vector<int> predictions = cnn.predict(X_test);
```

---

## Dataset and Experimental Results

**Dataset**: MNIST handwritten digits (28×28 grayscale, 10 classes).  
**Training samples**: 60,000  
**Test samples**: 10,000  
**Data normalization**: Standardized to zero mean and unit variance (done in Python before loading).

### MNIST Performance (5 epochs, batch size 32, lr=0.001)

| Epoch | Loss   | Test Accuracy |
|-------|--------|---------------|
| 1     | 0.4907 | 92.4%         |
| 2     | 0.2018 | 94.7%         |
| 3     | 0.1524 | 96.0%         |
| 4     | 0.1253 | 96.8%         |
| 5     | 0.1076 | **97.0%**     |

**Final test accuracy**: 96.98%  
**Training time**: ~15 minutes per epoch (MSVC `/Ox` optimization, single CPU core).

The accuracy is well within the expected range for a hand‑implemented CNN without data augmentation, batch normalization, or advanced optimizers (Adam, etc.).

---

## Limitations

1. **Fixed architecture** – Designed only for small square images like MNIST (28×28). For larger inputs (e.g., 224×224), the dimension calculations fail.
2. **No advanced regularization** – Dropout, L2 weight decay, or batch normalization are not included.
3. **Performance** – Slower than optimized libraries (e.g., TensorFlow, PyTorch) due to naive convolution implementation (sliding window) and single‑threaded execution.
4. **No GPU support** – Uses only CPU (Eigen on CPU). No CUDA or OpenCL.
5. **Gradient checkpointing** – Stores all intermediate activations, increasing memory usage for deeper networks.
6. **Assumes square inputs** – Non‑square images are not supported by the constructor.

Despite these limitations, `Simple_CNN` serves as an excellent learning tool to understand the internal mechanics of CNNs without relying on heavyweight frameworks.

---

## References

- Li Hang. "Statistical Learning Methods" (2nd ed.), Chapter 26. Tsinghua University Press.
- Eigen library: https://eigen.tuxfamily.org
```
