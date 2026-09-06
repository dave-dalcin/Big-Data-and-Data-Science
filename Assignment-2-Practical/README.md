# Big Data and Data Science — Practical Assignment

## From Clickstream Data to Purchase Prediction: A Practical Decision Tree Application

**Student:** Dave Dalcin
**Student ID:** 93315
**Course:** Big Data and Data Science
**Institution:** Swiss School of Business and Management Geneva
**Course Lecturer:** Sabrina Šuman, PhD, Senior Lecturer

## Project Overview

This project presents a practical example of Big Data Analytics and Data Science applied to e-commerce session data. Its objective is to investigate whether behavioural, technical and contextual information recorded during a browsing session can help distinguish purchasing sessions from sessions that end without a transaction.

A Decision Tree classifier is developed in Python and evaluated against a majority-class baseline. Because purchasing sessions represent only 15.47% of the dataset, model evaluation considers balanced accuracy, precision, recall and F1-score in addition to overall accuracy.

The analysis also includes a sensitivity test in which `PageValues`, the most influential predictor in the selected model, is removed. This comparison examines how strongly predictive performance depends on the availability of that variable.

## Dataset

The project uses the **Online Shoppers Purchasing Intention Dataset** from the UCI Machine Learning Repository.

The dataset contains:

* 12,330 e-commerce sessions;
* 17 predictor variables;
* one binary target named `Revenue`;
* 10,422 sessions without a purchase;
* 1,908 purchasing sessions;
* no missing values.

Each row represents an aggregated session rather than a continuous stream of individual click events. The target indicates whether the session ended with a transaction.

Dataset reference:

> Sakar, C., & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q

## Analytical Method

The practical analysis includes the following stages:

1. Inspection of the dataset structure and variable types.
2. Evaluation of missing values and exact matching rows.
3. Descriptive analysis of numerical and categorical variables.
4. Examination of the target-class imbalance.
5. Definition of numerical measures and categorical predictors.
6. Stratified 80:20 training and test split.
7. One-hot encoding of categorical variables.
8. Construction of a majority-class baseline.
9. Decision Tree parameter selection using five-fold stratified cross-validation.
10. Final evaluation on a separate test set.
11. Interpretation of the confusion matrix and tree structure.
12. Analysis of impurity-based feature importance.
13. Sensitivity analysis without `PageValues`.

The primary model-selection metric is balanced accuracy. This metric gives equal importance to the model's performance on purchasing and non-purchasing sessions.

## Selected Decision Tree

Cross-validation selected the following configuration for the primary model:

| Parameter                               | Selected value |
| --------------------------------------- | -------------- |
| Split criterion                         | Gini           |
| Maximum depth                           | 3              |
| Minimum samples per leaf                | 100            |
| Class weighting                         | Balanced       |
| Mean cross-validation balanced accuracy | 0.855          |

The preprocessing pipeline transformed the 17 original predictors into 75 model predictors. The selected tree used six of them in its decision rules.

## Main Test Results

| Metric             | Majority baseline | Decision Tree |
| ------------------ | ----------------: | ------------: |
| Accuracy           |             84.5% |         83.3% |
| Balanced accuracy  |             50.0% |         84.3% |
| Purchase precision |              0.0% |         47.7% |
| Purchase recall    |              0.0% |         85.9% |
| Purchase F1-score  |              0.0% |         61.4% |

Although the baseline achieved slightly higher overall accuracy, it classified every session as a non-purchase and failed to identify any of the 382 purchasing sessions in the test set. The Decision Tree correctly identified 328 purchasing sessions.

## Sensitivity Analysis

`PageValues` accounted for 88.4% of the primary model's total impurity-based feature importance. A second Decision Tree was therefore selected and evaluated after removing this predictor.

| Metric             | With PageValues | Without PageValues | Difference |
| ------------------ | --------------: | -----------------: | ---------: |
| Accuracy           |           83.3% |              63.7% |   −19.5 pp |
| Balanced accuracy  |           84.3% |              66.2% |   −18.2 pp |
| Purchase precision |           47.7% |              25.5% |   −22.3 pp |
| Purchase recall    |           85.9% |              69.6% |   −16.2 pp |
| Purchase F1-score  |           61.4% |              37.3% |   −24.1 pp |

The reduced model retained some ability to identify purchasing sessions, but its performance declined substantially. This demonstrates that the main model depends strongly on `PageValues`. Before operational deployment, an organisation would need to confirm how and when this variable is calculated and whether it is available at the intended intervention point.

## Project Structure

```text
Big-Data-and-Data-Science/
├── Assignment-1-Term-Paper/
│   └── Big_Data_and_Data_Science_Part-1-Term-Paper.pdf
└── Assignment-2-Practical/
    ├── data/
    │   └── online_shoppers_intention.csv
    ├── documentation/
    │   └── Big_Data_and_Data_Science_Part-2-Practical-Documentation.pdf
    ├── notebooks/
    │   └── Big_Data_and_Data_Science_Part_2.ipynb
    ├── outputs/
    │   ├── decision_tree_confusion_matrix.png
    │   ├── decision_tree_feature_importance.png
    │   ├── pagevalues_sensitivity_comparison.png
    │   ├── revenue_class_distribution.png
    │   └── selected_decision_tree.png
    ├── screenshots/
    │   └── Figures used in the practical documentation
    ├── .gitignore
    ├── README.md
    └── requirements.txt
```

## How to Run the Project

### 1. Create a virtual environment

From the Assignment-2-Practical directory:

```bash
python3 -m venv .venv
```

### 2. Activate the environment

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Open the notebook

Open the project folder in Visual Studio Code and select the Python interpreter located inside `.venv`.

The notebook is located at:

```text
notebooks/Big_Data_and_Data_Science_Part_2.ipynb
```

Alternatively, it can be opened through Jupyter:

```bash
cd notebooks
jupyter notebook Big_Data_and_Data_Science_Part_2.ipynb
```

### 5. Execute the complete analysis

Restart the notebook kernel and select **Run All**. The notebook reads the source dataset from `data` and saves the generated figures in `outputs`.

## Software and Libraries

The analysis was developed using:

* Python 3.14;
* Jupyter Notebook;
* Visual Studio Code;
* pandas;
* NumPy;
* Matplotlib;
* Seaborn;
* scikit-learn.

The Python dependencies required to run the project are listed in requirements.txt.

## Reproducibility

A fixed random state of `42` is used for the train-test split and Decision Tree models. The training and test sets are stratified to preserve the original purchase rate.

All preprocessing and model-selection operations are performed through scikit-learn pipelines. The test set remains separate from parameter selection and is used only for final evaluation.

## Interpretation and Limitations

The reported relationships are predictive associations and should not be interpreted as causal effects. Impurity-based feature importance describes how the selected Decision Tree formed its splits but does not demonstrate that a predictor causes purchasing behaviour.

The dataset contains aggregated session observations rather than raw event streams, customer histories or multiple channels. Exact matching rows were retained because no session identifier was available to determine whether they represented accidental duplication or separate sessions with identical aggregated values.

The practical usefulness of the primary model also depends on the availability of `PageValues`. If this variable is calculated only after the intended intervention point, the primary model's performance would not represent a realistic early-prediction scenario.

## Use of AI-Assisted Development

GitHub Copilot with GPT-5.6 Sol Light was used as a coding assistant during development. Its suggestions were reviewed before use. The execution, interpretation and validation of the analysis remained the responsibility of the author.

## References

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Sakar, C. O., Polat, S. O., Katircioglu, M., & Kastro, Y. (2019). Real-time prediction of online shoppers’ purchasing intention using multilayer perceptron and LSTM recurrent neural networks. *Neural Computing and Applications, 31*(10), 6893–6908. https://doi.org/10.1007/s00521-018-3523-0

Sakar, C., & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q
