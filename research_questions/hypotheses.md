<!-- Hypotheses template -->
# Research Hypothes- Phase 1

## H1: Parameter Efficiency
**Hypothesis:** DenseNet will achieve comparable accuracy to ResNet with 30% fewer parameters due to better feature reuse.

**Test:** Compare parameter counts vs accuracy for similar performance levels.

## H2: Training Dynamics
**Hypothesis:** ResNet will converge faster in early epochs, but EdnseNet will achieve better final accuracy.

**Test:** Analyze loss/accuracy curves across training epochs.


## H3: Depth Sensitivity
**Hypothesis:** ResNet performance will plateau faster with increasing depth compared to DenseNet on small datasets.

**Test:** Train multiple depth variants and measure accuracy gains per additional layer

## H4: Regularization Response
**Hypothesis:** DenseNet will be more sensitive to dropout due to its dense connectivity pattern.

**Test:** Ablation study with varying dropout rates.



