# Model Observations (Dry Bean Dataset)

These notes are based on the actual test-split metrics in `model_metrics.csv`,
per-class F1 scores, train-vs-test accuracy gaps, and the Random Forest
confusion matrix produced by `src/train_models.py` (see the run log for exact
numbers; all values below are taken directly from that run, not assumed).

## Logistic Regression
Accuracy 0.9147, AUC 0.9945 (the highest AUC of all five models), MCC 0.8973.
Train accuracy (0.9213) is only 0.65 points above test accuracy (0.9147), the
smallest train-test gap among the three non-Bayes models, indicating good
generalization and no meaningful overfitting even with `class_weight="balanced"`
on an imbalanced target. It is the most interpretable model (coefficients map
directly to feature effects) and the cheapest to train, but its linear decision
boundary caps performance on the harder, non-linearly-separable classes:
SIRA (F1 0.867) is its weakest class, mainly confused with DERMASON.

## Decision Tree
Accuracy 0.8981, the lowest of the four non-Bayes models. Despite capping
`max_depth=10` and `min_samples_leaf=5` specifically to control overfitting,
it still shows the largest train-test gap among the non-ensemble models
(train 0.9433 vs test 0.8981, a 4.5-point drop), meaning a single tree
memorizes training-set idiosyncrasies faster than it learns generalizable
splits. It remains the easiest model to explain to a non-technical audience
(a printable if/else path per prediction) and is the fastest to both train
and score, but on this dataset it is clearly outperformed by the ensemble
built from many such trees.

## kNN (k selected by 5-fold CV on the training set)
Grid search over k in {3,5,7,9,11} selected **k=9** (best CV weighted-F1
0.9240), used with standardized features (distance-based, scaling is
required). Test accuracy 0.9129 and AUC 0.987 are close to Logistic
Regression and Random Forest. The train-test gap (0.9371 vs 0.9129, 2.4
points) sits between Logistic Regression and Decision Tree. Its main
practical drawback is computational cost at inference: unlike the other
four models it stores the full training set (10,834 rows) and must compute
distances to all of them for every prediction, which is also why its saved
model file (~1.3 MB) is far larger than Logistic Regression, Decision Tree,
or Naive Bayes.

## Naive Bayes (Gaussian)
Clearly the weakest model: accuracy 0.7630, AUC 0.967, MCC 0.7143 - all
well below the other four. Its per-class F1 collapses hardest on BARBUNYA
(0.548) and SEKER (0.683), the two classes whose geometric features (area,
perimeter, axis lengths, shape factors) are the most mutually correlated in
this dataset - a direct violation of Naive Bayes' core conditional
independence assumption. Notably its train accuracy (0.7638) and test
accuracy (0.7630) are almost identical (0.0008 gap), the smallest gap of
any model here - it is not overfitting, it is systematically underfitting
because the independence assumption itself is a poor match for
shape-derived features that are geometrically dependent on one another.

## Random Forest (Ensemble)
The overall best performer on this dataset: highest Accuracy (0.9188),
Precision, Recall, F1 (0.9188/0.9188/0.9187) and MCC (0.9017), and a very
close second on AUC (0.9932 vs Logistic Regression's 0.9945). Individual
trees fit the training data perfectly (train accuracy 1.0000), the largest
train-test gap of any model (8.1 points down to 0.9188 test accuracy) -
but bagging 300 trees over bootstrapped samples and random feature subsets
still generalizes better than any single model tested, because the
ensemble average cancels out each tree's individual overfitting noise.
From the confusion matrix, its only notable weakness mirrors every other
model's: SIRA is confused with DERMASON in 57 of 527 cases (the single
largest off-diagonal error anywhere in the matrix), while the visually
distinct BOMBAY class is classified perfectly (F1 1.000) by all five
models - Bombay beans are the largest of the seven varieties, giving them
a clearly separable Area/Perimeter/AxisLength profile regardless of the
class-imbalance sensitivity that hurts BARBUNYA and SIRA elsewhere.

## Overall Winner: Random Forest (Ensemble)
It is chosen not because of a single metric but because it wins or ties on
five of the six metrics (Accuracy, Precision, Recall, F1, MCC) and is only
0.0013 behind Logistic Regression on AUC - a difference too small to change
the ranking. The trade-off is model size (~10.8 MB after compression, vs a
few KB for Logistic Regression/Decision Tree/Naive Bayes) and reduced
interpretability compared to a single Decision Tree or Logistic Regression,
which is worth noting for any application where explainability matters more
than raw accuracy.
