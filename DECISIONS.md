Decision log
**decision 1**
What I did:

Removed patient_id column.
Parsed admission_date into admission_month, admission_day, admission_dow.
Applied median imputation for numeric fields and most-frequent imputation for categorical fields.

Why I did it:

ID is unique and not meaningful for predictions.
Date likely contains useful timing patterns but needs to be model-friendly.
Needed a consistent cleaning path for both training and inference.

What I considered and rejected:

Keeping raw patient_id (noise/leakage risk).
Using admission_date as raw string (limits learning calendar patterns).
Manual one-off cleaning in notebook cells (hard to reproduce in deployment).

What would happen if I was wrong:

Dropping ID could remove signal if it was informative.
Date-derived features might add noise if weak.
Poor imputation could hurt metrics for minority classes (F1/recall).

**decision 2**
Decision 2: Model architecture and handling class imbalance

What I did:

Built a compact PyTorch MLP (128 -> 64) with BatchNorm, ReLU, and Dropout(0.2).
Trained using BCEWithLogitsLoss with pos_weight to up-weight the minority class.
Applied early stopping based on validation loss.

Why I did it:

Dataset (~3,800 rows) needed a model that’s expressive but not too big.
Observed class imbalance, so weighting helps the model pay attention to minority class.
Chose a simple, stable design that’s easy to explain and train quickly.

What I considered and rejected:

Bigger/deeper network (risk of overfitting).
Ignoring class imbalance (would hurt minority-class metrics).
Oversampling-only approach (kept pipeline simpler and deterministic).

What would happen if I was wrong:

Model might underfit or miss patterns if too simple.
Too strong class weighting could lower precision.
Early stopping too soon might reduce final performance.

**decision 3**
What I did:

Evaluated validation with ROC-AUC, PR-AUC, F1, precision, recall, and accuracy.
Chose decision threshold by maximizing minority-class F1 instead of using 0.5.
Saved threshold and metrics in artifacts/config.json for reproducibility.

Why I did it:

Class imbalance meant overall accuracy alone wasn’t enough.
Fixed 0.5 threshold didn’t give the best precision-recall tradeoff.
Needed a clear threshold for consistent inference and discussion.

What I considered and rejected:

Reporting accuracy only (insufficient for imbalanced data).
Using default threshold 0.5 (would reduce F1).
Optimizing only ROC-AUC (threshold decisions are operational).

What would happen if I was wrong:

Threshold tuning could overfit validation, hurting test F1.
F1 might not match the hospital’s desired precision/recall balance.
Too aggressive threshold could increase false positives in deployment.