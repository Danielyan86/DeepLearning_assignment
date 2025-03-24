# Rice Plant Disease Classification

This project implements various deep learning approaches for classifying rice plant diseases using the [Rice Plant Diseases Dataset](https://www.kaggle.com/datasets/jay7080dev/rice-plant-diseases-dataset).

## Project Structure

```
.
├── src/
│   ├── models/         # Model architectures and training code
│   ├── data/          # Data loading and preprocessing utilities
│   ├── utils/         # Helper functions and utilities
│   ├── visualization/ # Visualization tools for results
│   └── config/        # Configuration files for different experiments
├── notebooks/         # Jupyter notebooks for analysis
└── data/             # Dataset directory (not included in git)
```

## Requirements

- Python 3.8+
- TensorFlow/Keras
- PyTorch
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Implementation Approaches

The project implements four different approaches for rice plant disease classification:

1. Fine-tuned Pre-trained CNN Model
2. Fine-tuned Pre-trained Transformer Model
3. Custom CNN Architecture
4. Custom Transformer Architecture

Each approach includes:

- Hyperparameter tuning
- 80/20 train/validation split
- Learning curve visualization
- Model performance evaluation

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download the dataset from Kaggle and place it in the `data/` directory

3. Run experiments:

```bash
python src/models/train.py --model_type [cnn/transformer] --approach [pretrained/custom]
```

## Results

The project generates:

- Learning curves for each model
- Validation accuracy tables for different hyperparameter combinations
- Analysis of worst misclassified examples
- Model architecture diagrams
