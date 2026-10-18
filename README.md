# ResNet-50 Cecum Detection in Endoscopy Images

Fine-tunes an ImageNet-pretrained ResNet-50 to perform **binary classification** on lower-GI-tract endoscopy images: *cecum* vs. *not cecum*. Training is done in [training.py](training.py).

## Data

- Dataset: the lower-GI-tract labeled images from [HyperKvasir](https://datasets.simula.no/hyper-kvasir/), loaded with `torchvision.datasets.ImageFolder` (one sub-folder per class).
- The path is hard-coded in `training.py` (`root=...`); edit it to point to your local copy.
- Preprocessing: resize to 224×224, convert to tensor, normalize with ImageNet mean/std.
- Split: random 80% train / 20% validation.

## Handling class imbalance

The multi-class folder labels are collapsed into a binary target (`1` = cecum, `0` = everything else). Since cecum images are a minority, imbalance is addressed in two ways:

1. **Weighted sampling** – the training loader uses a `WeightedRandomSampler` (with replacement) so each batch is drawn roughly evenly from both classes.
2. **Weighted loss** – `BCEWithLogitsLoss` is given `pos_weight = n_non_cecum / n_cecum`.

## Model

- `torchvision` ResNet-50 with `IMAGENET1K_V2` weights.
- All parameters are frozen, then the **last 20 parameter tensors** are unfrozen for fine-tuning.
- The original 1000-class `fc` layer is replaced with a head: `Linear(2048→256) → ReLU → Dropout(0.3) → Linear(256→1) → Sigmoid`.

## Training setup

| Setting | Value |
| --- | --- |
| Epochs | 20 |
| Batch size | 32 |
| Optimizer | Adam, lr = 1e-4 (trainable params only) |
| Scheduler | `ReduceLROnPlateau` on validation loss (factor 0.5, patience 3) |
| Precision | Mixed precision (`autocast` + `GradScaler`) |
| Device | CUDA if available, otherwise CPU |

## Per-epoch output

Each epoch runs a training pass, then a validation pass (no gradients), and prints:

- training loss and validation loss (averaged over batches)
- validation **accuracy**
- validation **sensitivity** (recall for cecum: TP / (TP + FN)), the key metric for not missing cecum images
- current learning rate

Example line:

```
Epoch 1/20 | Train loss: ... | Val loss: ... | Accuracy: ... | Sensitivity: ... | LR: 1.00e-04
```

## Running

```
python training.py
```

Requires `torch` and `torchvision`. The script currently trains and reports metrics only; it does not save the model weights.
