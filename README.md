Readmission-DL — City General Hospital 30-day Readmission Prediction
Student name: Ishika Kumari 
Student ID: 1148
Submission date: 27/03/2026

Problem
Predict whether a patient will be readmitted within 30 days of discharge using structured clinical data from City General Hospital (3,800 training records, 950 test records).


My model______
--Architecture:--
 
I used a small PyTorch MLP because the data is tabular and the problem is binary classification. The network has two hidden layers (128 and 64 neurons), with BatchNorm, ReLU, and Dropout to keep training stable and reduce overfitting. The final layer has one output neuron, and the model was trained using BCEWithLogitsLoss. I also used early stopping based on validation loss.

--Key preprocessing decisions:--
 
I removed columns like patient_id because they act more like identifiers than useful predictive features. I also converted admission_date into calendar-based features such as month, day, and day of week to capture possible time-related patterns. To keep preprocessing clean and reusable, I used a ColumnTransformer, with median imputation + scaling for numeric features and most-frequent imputation + one-hot encoding for categorical features.
 
--How I handled class imbalance:--
 
I used `pos_weight` in `BCEWithLogitsLoss`, computed from the training split, so the minority class gets a stronger learning signal. On top of that, I tuned the decision threshold on the validation set to maximize minority-class F1 instead of using a fixed `0.5`, which gives a better precision-recall tradeoff for readmission prediction.


Results on validation set
Metric	Value
AUROC	
F1 (minority class)	
Precision (minority)	
Recall (minority)	
Decision threshold used	
How to run
1. Install dependencies
pip install -r requirements.txt
2. Train the model (optional — pretrained weights included)
python notebooks/solution.ipynb  # or run cells in order
3. Run inference on the test set
python src/predict.py --input data/test.csv --output predictions.csv
The output CSV will contain two columns: patient_id and readmission_probability.

Repository structure
readmission-dl/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── solution.ipynb
├── src/
│   └── predict.py
├── DECISIONS.md
├── requirements.txt
└── README.md
