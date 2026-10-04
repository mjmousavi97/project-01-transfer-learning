# Fine-Grained Pet Classification with Transfer Learning

A controlled study of how transfer learning behaves on the **Oxford-IIIT Pet** dataset (37 cat and dog breeds), implemented with TensorFlow/Keras and **hand-written training and validation loops**.

> **Status:** work in progress. Results below are placeholders and will be filled in as experiments are completed.

## Research Question

How much does transfer learning help on fine-grained image classification, and how does the answer change as the amount of training data shrinks?

To answer this, the project compares different ways of adapting an ImageNet-pretrained backbone, using a fair and reproducible protocol.

## Dataset

- **Oxford-IIIT Pet**, loaded through `tensorflow_datasets` (`oxford_iiit_pet`).
- 37 classes (12 cat breeds, 25 dog breeds), 7,349 images in total, roughly 200 images per class.
- Images vary in size and aspect ratio.

### Splits

The official split (3,680 train / 3,669 test) is re-partitioned into three parts so that hyperparameters can be tuned without touching the test set:

| Split | Images | Purpose |
|-------|--------|---------|
| Train | 5,148 | Model training |
| Validation | 1,100 | Hyperparameter tuning, model selection |
| Test | 1,101 | Final evaluation only |

> **Note:** because the official test set was partly merged into training, the numbers here are not directly comparable to published results on the official split.
> *TODO: verify per-class balance in each split and document the outcome.*

## Models and Experiments

| Experiment | Description |
|------------|-------------|
| Frozen backbone | ImageNet ResNet-50 as a fixed feature extractor, train only a new classification head |
| Partial fine-tuning | Unfreeze the last N blocks, with a lower learning rate for the backbone |
| Full fine-tuning | Train all layers |
| ConvNeXt-Tiny | Same protocols, used as an architectural comparison |

Planned analyses:

- Low-data curve: training with 10%, 25%, 50% and 100% of the training data
- Ablations: augmentation, label smoothing, learning-rate schedule
- Error analysis: confusion matrix, most-confused breeds, misclassified examples
- Model inspection: Grad-CAM and embedding visualization (t-SNE / UMAP)
- Cost: parameters, FLOPs and inference time

## Evaluation Protocol

- Identical splits, image size, augmentation and training budget across models.
- All hyperparameter decisions are made on the validation set. The test set is used once at the end.
- Each experiment is repeated with multiple seeds and reported as mean ± std.
- Metrics: accuracy, macro F1, confusion matrix.

## Results

*To be added.*

| Model | Val Accuracy | Test Accuracy | Test Macro F1 | Params |
|-------|--------------|---------------|---------------|--------|
| Frozen ResNet-50 | – | – | – | – |
| Fine-tuned ResNet-50 | – | – | – | – |
| ConvNeXt-Tiny | – | – | – | – |

## Project Structure

```
project-01-transfer-learning/
├── src/            # source code and notebooks
├── notebooks/      # exploratory notebooks
├── configs/        # experiment configurations
├── results/        # metrics, figures, logs
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/mjmousavi97/project-01-transfer-learning.git
cd project-01-transfer-learning
pip install -r requirements.txt
```

The notebooks were developed on Google Colab with a GPU runtime.

## Reproducibility

- Fixed random seeds (`set_seed` in `src/utils.py`).
- Pinned dependency versions in `requirements.txt`.
- Note that bit-exact results on GPU are not guaranteed even with fixed seeds.

## Roadmap

- [x] Project structure and dataset exploration
- [ ] Verified, reproducible train/val/test split
- [ ] Data pipeline with augmentation
- [ ] Frozen-backbone training with a custom loop
- [ ] Fine-tuning experiments
- [ ] ConvNeXt-Tiny comparison
- [ ] Low-data experiments and error analysis
- [ ] Final write-up of results and lessons learned

## License

*TODO: choose a license.*
