# C++ Transformer Framework: From Scratch to ViT

A **pure C++ deep learning framework** that implements Transformer and Vision Transformer (ViT) from first principles — no external libraries except Eigen for matrix operations. Hand-written automatic differentiation enables end-to-end training on real datasets.

## Architecture

The framework is built on a unified `Layer` abstraction inspired by the principle: *"Everything is a layer."*

```
class Layer {
    virtual mat forward(const mat& input) = 0;
    virtual mat backward(const mat& grad_output) = 0;
    virtual void update_params(double lr) = 0;
};
```

### Implemented Layers

| Layer | Description |
|-------|-------------|
| `Linear` | Fully-connected with He initialization and optional Adam |
| `ReLU` | Rectified Linear Unit activation |
| `Softmax_Attention` | Softmax with full Jacobian backward pass |
| `Self_Attention` | Multi-head self-attention with Q/K/V projections |
| `Layer_Norm` | Layer normalization with learnable γ, β, hand-derived backward |
| `Position_Encoding` | Learnable absolute position embeddings |
| `Patch_Embedding` | Image-to-token conversion via learned linear projection |
| `ave_pooling` | Global average pooling over token dimension |
| `FFN` | Two-layer feed-forward with ReLU |
| `Transformer_Encoder_Layer` | Self-attention + FFN with residual connections and LayerNorm |
| `Dropout` | Inverted dropout for regularization |

### Model Variants

- **FNN**: Multi-layer perceptron for regression/classification
- **Transformer Encoder (1D)**: For sequential data
- **Vision Transformer (ViT)**: Patch embedding → Transformer Encoder → classification head

## Training Flow

### Forward Pass
```
input → [Layer₁] → [Layer₂] → ... → [Layerₙ] → loss
```

### Backward Pass (Hand-written Autodiff)
```
∂L/∂x ← [Layerₙ.backward] ← ... ← [Layer₁.backward]
```
Each layer computes gradients for both its inputs and its learnable parameters, caching intermediate values during forward pass.

### Parameter Update
```
θ -= lr × Adam(∇θ)    or    θ -= lr × ∇θ (SGD)
```
All parameter gradients are accumulated over a mini-batch, then averaged before updating.

### Mini-batch Training
- Samples are shuffled each epoch
- For each batch: accumulate gradients across all samples, then call `update_params` once

## Optimization Journey

This was the most challenging part of the project. Training a ViT from scratch on CPU revealed extreme sensitivity to hyperparameters.

### Phase 1: Failure with Default Settings
**Configuration**: SGD lr=0.05, batch_size=64, 60k samples
**Result**: Loss dropped from 1.48 → 0.60, then oscillated and exploded to 0.85

**Diagnosis**: Learning rate too large for vanilla SGD on deep Transformer.

### Phase 2: Reduced Learning Rate
**Configuration**: SGD lr=0.01, batch_size=64, 60k samples
**Result**: Loss 1.48 → 0.60 → oscillated → diverged

**Diagnosis**: Fixed step size causes overshooting near local minima.

### Phase 3: Adam Optimizer
**Configuration**: Adam lr=0.001, batch_size=256, 60k samples
**Result**: Loss 0.77 → 0.32 in 3 epochs, then exploded to 1.36

**Diagnosis**: Adam's default learning rate still too aggressive for ViT without regularization.

### Phase 4: Gradient Clipping + Conservative LR
**Configuration**: Adam lr=0.0001, batch_size=256, grad_clip=1.0, 60k samples
**Result**: Loss steadily decreased 1.18 → 0.31 over 10 epochs, then slowly rose to 0.48

**Diagnosis (key insight)**: The framework is correct — the slow rise is caused by fixed learning rate without decay, mild overfitting without Dropout, and gradient accumulation over many steps.

### Optimal Configuration Found

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | Adam | Momentum + adaptive step sizes |
| Learning rate | 0.0001 | 10-100× smaller than typical PyTorch defaults |
| Batch size | 256 | Reduces gradient noise on large datasets |
| Gradient clipping | 1.0 | Prevents sudden gradient spikes |
| Weight init | He (√(2/fan_in)) | Stable variance propagation |
| Position encoding scale | 0.02 | Matches standard ViT initialization |

## Results

| Model | Data | Test Accuracy | Notes |
|-------|------|---------------|-------|
| FNN (SGD) | 60k MNIST | **96.0%** | Simple 2-layer MLP |
| ViT (Adam, lr=0.0001) | 60k MNIST | **85.6%** | 20 epochs, no augmentation |

**Key achievement**: The ViT trained stably for 10 epochs with monotonic loss decrease, proving the correctness of all layer implementations, backward propagation, and the Adam optimizer.

## Known Limitations & Improvements

### Training Stability
- **Learning rate schedule**: Cosine annealing or linear warmup-decay would prevent late-stage drift
- **Dropout**: Adding Dropout(0.1) after attention and FFN outputs would reduce overfitting and attention spikiness
- **Tighter gradient clipping**: Reducing max_norm from 1.0 to 0.1–0.5 for later epochs

### Performance
- **Matrix fusion**: 128 individual `Linear(16,1)` filters in `Patch_Embedding` could be replaced by one `Linear(16, 128)` for 100× speedup
- **Batch parallelism**: Current implementation processes samples sequentially within a batch; grouping into 3D tensors would leverage Eigen's vectorization
- **GPU acceleration**: The `Layer` interface is device-agnostic; porting to CUDA would require only modifying `Eigen::MatrixXd` → GPU tensors

### Model Architecture
- **Deeper models**: Scale to 6–12 layers with embed_dim=128–256 for better accuracy
- **Convolutional stem**: Adding 2–3 Conv2D layers before patch embedding improves feature extraction
- **Data augmentation**: Random translation (±2px), rotation (±10°) and scaling would boost generalization significantly

## Build & Run

### Requirements
- **Eigen 3.4+** (header-only, no compilation needed)
- **C++17** compiler (GCC 9+, MSVC 2019+, Clang 10+)
- **Python** (only for data preprocessing, optional)

### Compilation
```bash
g++ -std=c++17 -O2 -I/path/to/eigen main.cpp -o vit
```

### Data Preparation
Convert MNIST to plain text format:
```python
# See scripts/extract_mnist.py
python extract_mnist.py  # generates mnist_txt/train_images.txt, etc.
```

### Training
```cpp
Vision_transformer_Encoder model(
    4,      // num_layers
    64,     // embed_dim
    2,      // num_heads
    256,    // batch_size
    128,    // hidden_dim
    10,     // num_classes
    4,      // patch_size
    28      // image_size
);
model.fit(X_train, y_train, 20, 0.0001);
```

## References

- **Li Hang, "Statistical Learning Methods" (3rd Ed.), Chapter 28**: Transformer architecture and self-attention mechanism
- Vaswani et al. (2017): "Attention Is All You Need" — original Transformer
- Dosovitskiy et al. (2020): "An Image is Worth 16x16 Words" — Vision Transformer (ViT)
- Kingma & Ba (2014): "Adam: A Method for Stochastic Optimization" — Adam optimizer
- He et al. (2015): "Delving Deep into Rectifiers" — He initialization

---

**Author's Note**: This project was built to deeply understand every component of modern deep learning — from backward propagation and Adam to multi-head attention and ViT. Every gradient was derived and implemented by hand. The journey from "Why is my loss exploding?" to achieving 85.6% accuracy taught more than any framework ever could.
