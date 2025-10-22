# Research Plan: DenseNet vs ResNet Comparison

## 1. Core Research Questions

### 1.1 Architectural Efficiency
- How do ResNet and DenseNet compare in terms of parameter?
- what is the computational cost (FLOPs) difference betweeen 

### 1.2 Performance Analysis
- Accuracy comparison on CIFAR-10 and CIFAR-100
- Training convergence speed and Stability
- Sensitivity to hyperparameters and regularization

### 1.3 Ablation  Studies
- Impact of Network depth on both architecture
- Effect of different connection patterns
- Role of bottleneck layers and growth rates

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
