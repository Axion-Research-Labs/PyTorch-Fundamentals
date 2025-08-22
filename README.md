# **PyTorch Fundamentals**

This repository is part of Axion Research Lab’s public learning track, beginning with PyTorch fundamentals. The aim is to establish rigorous foundations and reproducible tooling to support future research. Each notebook here is a polished version of internal study notes.

---

## **01. PyTorch Fundamentals**

**Notebook:** `01.ipynb`  
**Goal:** Build a precise mental model for tensors, shapes, dtypes, device placement, and basic linear algebra.

### **What this notebook covers**

- Creating tensors: random, zeros, ones, ranges, and tensor-like
- Tensor dtypes and device management (CPU and GPU)
- Inspecting tensor metadata: `shape`, `ndim`, `dtype`, `device`
- Elementwise operations vs matrix multiplication
- Aggregations, argmin/argmax, indexing, reshaping, stacking
- Interop with NumPy: `torch.from_numpy`, `tensor.numpy()`
- Reproducibility: random seeds and deterministic behavior
- Moving data between CPU and GPU

### **Learning outcomes**
- Know when you are doing elementwise vs linear-algebra operations
- Handle shape mismatches without guesswork
- Move tensors across devices safely and predictably
- Set seeds and explain what is or is not reproducible

---

## **Acknowledgements**

The learning track builds on excellent open-source materials. In particular:

- [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html) — the primary reference for API usage, tensor operations, and framework best practices.  
- [mrdbourke/pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning) — for the foundational tutorials and exercises that inspired and guided this notebook series.
