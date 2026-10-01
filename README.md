 Iris Species Classification with Machine Learning for Kids
Overview
This project implements a machine learning model for classifying Iris flower species based on sepal and petal measurements. The model is trained using the Machine Learning for Kids (MLforKids) platform and consumed through its REST API. It performs multiclass classification across three Iris species using four numerical features.

Species	Samples in Dataset
Setosa	50
Versicolor	50
Virginica	50
Total	150
Project Structure
File	Description
main.ipynb	Main notebook containing the complete pipeline
mlforkidsnumbers.py	MLforKidsNumbers class for connecting to the MLforKids model
iris_dataset.xlsx	Full dataset (150 samples in the Iris sheet)
iris_train.xlsx	Training split (120 samples — 80%)
iris_test.xlsx	Test split (30 samples — 20%)
iris_test_with_prediction.xlsx	Test data with an appended Prediction column
train_setosa.csv	40 labeled training samples for Setosa
train_versicolor.csv	40 labeled training samples for Versicolor
train_virginica.csv	40 labeled training samples for Virginica
Input Features
The model consumes four numerical features, all measured in centimeters:

Feature	Description
SepalLengthC	Sepal length
SepalWidthCm	Sepal width
PetalLengthC	Petal length
PetalWidthCm	Petal width
Pipeline
1. Dataset Loading and Inspection
python
import pandas as pd

data = pd.read_excel('iris_dataset.xlsx', sheet_name='Iris')
print(data['Species'].value_counts())
The dataset is fully balanced: 50 samples per species.

2. Train/Test Split
Ratio: 80% training / 20% testing

Stratified sampling on the Species column to preserve class balance

random_state=42 for reproducibility

Result: 120 training samples and 30 test samples.

3. Preparation of Labeled CSV Files
For each species, a separate CSV file is generated without the Species column, to be uploaded to MLforKids as labeled training data.

4. Model Connection
python
from mlforkidsnumbers import MLforKidsNumbers

project = MLforKidsNumbers(
    modelurl="https://mlforkids-newnumbers.j8ayd8ayn23.eu-de.codeengine.appdomain.cloud/saved-models/..."
)
On first execution, the model is downloaded and cached under saved_models/. Subsequent runs reuse the cached model.

5. Inference on Test Data
python
response = project.classify({
    "SepalLengthC": 5.1,
    "SepalWidthCm": 3.5,
    "PetalLengthC": 1.4,
    "PetalWidthCm": 0.2,
})
# Result: 'setosa' with 96% confidence
Predictions are persisted to iris_test_with_prediction.xlsx.

Evaluation
Confusion Matrix
text
                 Predicted
                 setosa   versicolor   virginica
Actual setosa       10         0            0
       versicolor    0         9            1
       virginica     0         1            9
Confusion Matrix Visualization
https://confusion_matrix.png/

Place the exported confusion matrix image in the project root and name it confusion_matrix.png. The visualization is generated in main.ipynb using seaborn.heatmap with a Blues colormap and integer annotations.

Classification Report
Class	Precision	Recall	F1-Score	Support
setosa	1.00	1.00	1.00	10
versicolor	0.90	0.90	0.90	10
virginica	0.90	0.90	0.90	10
Overall Accuracy			0.93	30
Macro Average	0.93	0.93	0.93	30
Weighted Average	0.93	0.93	0.93	30
Overall accuracy: 93%

The Setosa class is classified with perfect precision and recall.

The only two misclassifications occur between Versicolor and Virginica, which is expected given the significant overlap in their feature distributions.

Conclusions
The model achieves an overall accuracy of 93% on the held-out test set.

Petal-related features (length and width) are the primary discriminators between species.

Setosa is linearly separable from the other two classes; the Versicolor/Virginica boundary remains the primary source of error.

Leveraging the MLforKids service simplifies both model training and deployment, eliminating the need for deep machine learning expertise.

Running the Project
bash
pip install pandas scikit-learn matplotlib seaborn requests ydf openpyxl
jupyter notebook main.ipynb
Execute the notebook cells sequentially from top to bottom.

Dependencies
pandas — data manipulation

scikit-learn — data splitting and evaluation

matplotlib and seaborn — visualization

ydf — model loading

requests — API communication

openpyxl — Excel I/O

About MLforKids
Machine Learning for Kids is an educational platform that enables users, particularly beginners, to train and deploy machine learning models without writing complex code.

