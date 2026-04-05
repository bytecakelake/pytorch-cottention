# Cottention: Cosine Similarity Attention (CSA)

**Efficient Attention in PyTorch: Linear Space and Time Complexity**

This library provides a high-performance implementation of **Cosine Similarity Attention (Cottention)**. Unlike standard Softmax Attention which suffers from quadratic ( $O(n^2)$ ) complexity, this implementation achieves **Linear complexity ( $O(n)$ )** in both space and time, enabling the processing of significantly longer sequences with a constant memory footprint during inference.

## Key Features

- **Linear Complexity**: Scalable to extremely long sequences where traditional Transformers fail.
- **Constant Memory Inference**: Can be reformulated as an RNN for efficient auto-regressive generation.
- **PyTorch Native**: Built with standard PyTorch operations for maximum compatibility.

## Installation

Install directly from the source:

```bash
pip install git+https://github.com/bytecakelake/pytorch-cottention
```

## Usage

### Simple Example

```python
import torch as pt
from pt_cotten import cottention

# Dynamic input: (Batch, Heads, Sequence Length, Head Dimension)
Q = pt.randn(1, 8, 1024, 64)
K = pt.randn(1, 8, 1024, 64)
V = pt.randn(1, 8, 1024, 64)

# Output has the same shape as V
output = cottention(Q, K, V)
print(output.shape)  # pt.Size([1, 8, 1024, 64])
```

## How it Works

Standard attention computes $Softmax(QK^T)V$. Cottention replaces this with:
$$\text{Attn}(Q, K, V) = \text{norm}(Q) \left( \text{norm}(K)^T V \right)$$
By changing the order of operations (associativity), we avoid computing the large $N \times N$ attention matrix, reducing memory usage from $O(N^2)$ to $O(N \cdot d)$.

## Acknowledgements

This implementation is based on the following paper:

- **Title**: Cottention: Linear Transformers With Cosine Attention
- **Authors**: Gabriel Mongaras, Trevor Dohm, Eric C. Larson
- **Journal**: arXiv:2409.18747 [cs.LG] (2024)
- **Link**: [https://arxiv.org/abs/2409.18747](https://arxiv.org/abs/2409.18747)
