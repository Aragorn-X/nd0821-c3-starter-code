# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is a Random Forest Classifier using the default hyperparameters in scikit-learn.

## Intended Use
This model should be used to infer the salary of a person given input features such as age, workclass, occupation, etc.

## Training Data
The data was obtained from the UCI Machine Learning Repository
(Census Income Data Set - https://archive.ics.uci.edu/ml/datasets/census+income).
80% of data is used for training using scikit-learn method train_test_split with random_state=42.

## Evaluation Data
The data was obtained from the UCI Machine Learning Repository
(Census Income Data Set - https://archive.ics.uci.edu/ml/datasets/census+income).
20% of data is used for evaluation using scikit-learn method train_test_split with random_state=42.

## Metrics
The model was evaluated using precision, recall, f_beta as performance metrics with the
following results:
- precision:  0.74
- recall:  0.64
- f_beta: 0.68

## Ethical Considerations
The dataset contains personal information such as race, gender that could potentially lead to data unfairness.

## Caveats and Recommendations
The datasource dates back to 1994 when the dataset was extracted from the Census database.
Therefore, an actualization of the input data would be necessary to extract more precise
and meaningful considerations. 