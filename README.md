# Self-pollinated cannabis seeds lead to less variation in shape

This repository contains the data analysis and machine learning models for classifying cannabis cultivars based on seed morphology.

The primary objective is to determine if seed shape, defined by a set of landmark coordinates, is a sufficient trait to distinguish between different cultivars.

## 📈 Methodology

The analysis uses a dataset of 713 seeds, each described by 13 landmarks, resulting in 26 coordinates (X, Y) as features.

Four different classification models are tested and evaluated:
* Quadratic Discriminant Analysis (QDA)
* Random Forest
* Decision Tree
* Gaussian Naive Bayes (GaussianNB)

Additionally, two main experiments are conducted:
1.  **Full Training:** Models are trained on all 7 cultivars present in the dataset (`Fr`, `H`, `K`, `Le`, `Ma`, `Me`, `Pa`).
2.  **Partial Training:** Models are trained on a subset of 5 cultivars, excluding `H` and `K`, to evaluate performance without these, which might be outliers.

## 📂 Repository Structure

* **`data/raw_coordinates.txt`**: The raw dataset. Each row represents a seed, with 26 coordinate columns and one column for the cultivar.
* **`00-dataset.ipynb`**: Jupyter Notebook for initial data loading, exploration, and visualization. It reshapes the coordinates and plots the seed outlines.
* **`01-train.ipynb`**: Notebook that trains and evaluates the four classification models on the **full dataset** (7 cultivars). It includes a `GridSearchCV` for Random Forest hyperparameter tuning.
* **`01-train-without-H-K.ipynb`**: Notebook that performs the same analysis but on a **data subset** (5 cultivars), excluding 'H' and 'K'.
* **`utils.py`**: (Inferred) Utility script containing helper functions, such as `calc_confusion_matrix` for generating confusion matrices.
* **`fig_all/`**: (Inferred) Directory where generated images are saved, including the cultivar distribution and confusion matrices for each model.

## 📊 Key Results

The classifiers' results (accuracy, recall, f1-score) and their corresponding confusion matrices are detailed in the notebooks.

* On the **full dataset (7 classes)**, the optimized Random Forest model (after GridSearchCV) achieves an accuracy of ~70%.
* On the **reduced dataset (5 classes)**, excluding 'H' and 'K', the optimized Random Forest's accuracy improves to ~78%.

This suggests that the 'H' and 'K' cultivars are morphologically more difficult to distinguish or have significant overlap with the other classes.

## 🚀 How to Run the Project

### 1. Prerequisites

You will need Python 3.x installed, along with the following libraries:

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn

You can install them using `pip`:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Execution

1.  Clone the repository:
    ```bash
    git clone [https://github.com/Francisco-ft/Self-pollinated-cannabis-seeds-lead-to-less-variation-in-shape.git](https://github.com/Francisco-ft/Self-pollinated-cannabis-seeds-lead-to-less-variation-in-shape.git)
    cd Self-pollinated-cannabis-seeds-lead-to-less-variation-in-shape
    ```
2.  Open and run the Jupyter notebooks in order:
    * `00-dataset.ipynb` (to explore the data).
    * `01-train.ipynb` or `01-train-without-H-K.ipynb` (to run the models).
