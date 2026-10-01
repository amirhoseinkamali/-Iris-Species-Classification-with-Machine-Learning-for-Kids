# Iris Species Classification with Machine Learning for Kids

## Overview
This project demonstrates the classification of Iris flower species using a machine learning model trained on the "Machine Learning for Kids" platform. The model takes four numerical features (sepal and petal measurements) to perform multiclass classification among three Iris species: Setosa, Versicolor, and Virginica. The entire pipeline, from data preparation to model evaluation, is executed within a Jupyter Notebook, which consumes the pre-trained model via a REST API.

The dataset is perfectly balanced, with 50 samples for each species.

| Species    | Samples in Dataset |
|------------|--------------------|
| Setosa     | 50                 |
| Versicolor | 50                 |
| Virginica  | 50                 |
| **Total**      | **150**                |

## Project Structure
| File                           | Description                                                                  |
|--------------------------------|------------------------------------------------------------------------------|
| `main.ipynb`                   | The main Jupyter Notebook containing the complete project pipeline.          |
| `mlforkidsnumbers.py`          | A Python class to connect to and interact with the MLforKids model API.      |
| `iris_dataset.xlsx`            | The complete Iris dataset with 150 samples.                                  |
| `iris_train.xlsx`              | The training subset (120 samples, 80% of the data).                          |
| `iris_test.xlsx`               | The testing subset (30 samples, 20% of the data).                            |
| `iris_test_with_prediction.xlsx` | The test data with the model's predictions appended in a new column.         |
| `train_setosa.csv`             | 40 training samples labeled as 'Setosa' for MLforKids.                       |
| `train_versicolor.csv`         | 40 training samples labeled as 'Versicolor' for MLforKids.                   |
| `train_virginica.csv`          | 40 training samples labeled as 'Virginica' for MLforKids.                    |

## Input Features
The model uses four numerical features, all measured in centimeters:

| Feature      | Description    |
|--------------|----------------|
| `SepalLengthC` | Sepal length   |
| `SepalWidthCm` | Sepal width    |
| `PetalLengthC` | Petal length   |
| `PetalWidthCm` | Petal width    |

## Pipeline

### 1. Dataset Loading and Splitting
The full dataset is loaded from `iris_dataset.xlsx`. It is then split into training (80%) and testing (20%) sets. A stratified split on the `Species` column ensures that the proportion of each species is maintained in both sets. `random_state=42` is used for reproducibility.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_excel('iris_dataset.xlsx', sheet_name='Iris')

data_train, data_test = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data['Species']
)

# Result: 120 training samples and 30 test samples.
```

### 2. Preparation of Labeled CSV Files
The training data is further split by species. For each species, a separate CSV file is generated containing only the feature data. These files are formatted for upload to the Machine Learning for Kids platform to train the model.

```python
for species in ['setosa', 'versicolor', 'virginica']:
    subset = data_train[data_train['Species'] == species].copy()
    subset = subset.drop(columns=['Species'])  
    subset.to_csv(f"train_{species}.csv", index=False)
```

### 3. Model Connection and Inference
The `MLforKidsNumbers` class from `mlforkidsnumbers.py` is used to connect to the deployed model. On first run, the model is downloaded and cached locally. Predictions are made by passing a dictionary of features to the `classify` method.

```python
from mlforkidsnumbers import MLforKidsNumbers

project = MLforKidsNumbers(
    modelurl="https://mlforkids-newnumbers.j8ayd8ayn23.eu-de.codeengine.appdomain.cloud/saved-models/auth0|6a48f728e746a2b505255257-2/status"
)

testvalue = {
    "SepalLengthC": 5.1,
    "SepalWidthCm":  3.5,
    "PetalLengthC": 1.4,
    "PetalWidthCm":  0.2,
}

response = project.classify(testvalue)
# Result: 'setosa' with 96% confidence
```

### 4. Batch Prediction
The model is used to predict the species for all samples in the test set. The predictions are then stored in a new `Prediction` column and saved to `iris_test_with_prediction.xlsx`.

```python
data_test = pd.read_excel('iris_test.xlsx')

predictions = []
for i in range(len(data_test)):
    row = data_test.iloc[i]
    result = project.classify({
        "SepalLengthC": float(row['SepalLengthC']),
        "SepalWidthCm":  float(row['SepalWidthCm']),
        "PetalLengthC": float(row['PetalLengthC']),
        "PetalWidthCm":  float(row['PetalWidthCm']),
    })
    predictions.append(result[0]['class_name'])

data_test['Prediction'] = predictions
data_test.to_excel("iris_test_with_prediction.xlsx", index=False)
```

## Evaluation
The model's performance is evaluated on the held-out test set of 30 samples.

### Confusion Matrix
The confusion matrix shows the relationship between the actual species and the model's predictions. The model correctly identified all 10 `setosa` samples. The only two misclassifications occurred between `versicolor` and `virginica`.

```
Confusion Matrix:
 [[10  0  0]
 [ 0  9  1]
 [ 0  1  9]]
```
(Rows: Actual `setosa`, `versicolor`, `virginica`; Columns: Predicted `setosa`, `versicolor`, `virginica`)

### Classification Report
The model achieves an overall accuracy of 93%.

```
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       0.90      0.90      0.90        10
   virginica       0.90      0.90      0.90        10

    accuracy                           0.93        30
   macro avg       0.93      0.93      0.93        30
weighted avg       0.93      0.93      0.93        30
```

## Conclusion
- The machine learning model achieves an excellent overall accuracy of **93%** on the unseen test data.
- The `Setosa` species is perfectly classified, indicating it is linearly separable from the other two species based on the provided features.
- The errors are confined to confusion between `Versicolor` and `Virginica`, which is a common challenge with the Iris dataset due to the overlap in their feature distributions.
- The project highlights the effectiveness and simplicity of using the Machine Learning for Kids platform to train and deploy models without requiring deep expertise in ML frameworks.

## How to Run
1.  **Install Dependencies**: Ensure you have Python and the required libraries installed.
    ```bash
    pip install pandas scikit-learn matplotlib seaborn requests ydf openpyxl
    ```

2.  **Run the Notebook**: Launch Jupyter and open the `main.ipynb` file.
    ```bash
    jupyter notebook main.ipynb
    ```

3.  Execute the cells in the notebook sequentially from top to bottom to replicate the entire process.

## About MLforKids
Machine Learning for Kids is an educational platform designed to introduce beginners to machine learning concepts. It provides an intuitive interface for training models on text, numbers, and images, and it allows users to use these models in creative projects, such as games and applications built with Scratch or Python.
