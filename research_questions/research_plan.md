# Research Plan: DenseNet vs ResNet Comparison
**Date Created:** 2025-10-21
**Last Updated:** 2025-10-22


## 1. Core Research Questions

### Primary Questions
1. How do residual connections (ResNet) vs Dense Connections (DenseNet) affect:
    - Parameter efficiency on CIFAR-10/100?
    - Training convergence speed?
    - Feature reuse patterns?

### Secondary Questions
1. What is the optimal depth for each architecture on small-scale datasets?
2. How sensitive are theses architecture to regularization techniques?

## 2. Success Metrics
- **Primary:** >90% accuracy on CIFAR-10, >70% on CIFAR-100
- **Secondary:** Parameter count  <5M, training time < 4 hours per model

## 3. Experimental Scope
**In Scope:**
- ResNet-20, 32, 56 vs DenseNet-BC-100-12, 250-24
- Standard data augmentation
- Hyperparameter sensitivity analysis

**Out of Scope:**
- Very large models (>100 layers)
- Advance regularization techniques (initially)
- Other dataset domains


<!-- ### 1.1 Architectural Efficiency
- How do ResNet and DenseNet compare in terms of parameter?
- what is the computational cost (FLOPs) difference betweeen 

### 1.2 Performance Analysis
- Accuracy comparison on CIFAR-10 and CIFAR-100
- Training convergence speed and Stability
- Sensitivity to hyperparameters and regularization

### 1.3 Ablation  Studies
- Impact of Network depth on both architecture
- Effect of different connection patterns
- Role of bottleneck layers and growth rates -->

## 2. Experimental Setup

### 2.1 Models to Compare
**ResNet Variants**
- ResNet-20, ResNet-32, ResNet-56

**DenseNet Varaint**
- DenseNet-BC (L=100, k=12)
- DenseNet-BC (L=250, k=24)



### 2.2 Training Protocol
- SGD with momentum (0.9)
- Cosine Annealing learning rate schedule
- Standard data augmentation
- 200 epochs training


## 3. Evaluation Metrics
- Top-1 accuracy
- Training/validation loss curves
- Parameter count and Flops
- Memory usage during training
