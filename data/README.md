# Dataset -- COMPAS Recidivism (ProPublica)

## The problem

In 2016, ProPublica investigated COMPAS, a risk-assessment algorithm
actually used by courts in Broward County, Florida, to help inform
bail and sentencing decisions. COMPAS scores a defendant's likelihood
of reoffending on a 1-10 scale; judges could see that score when
deciding, among other things, whether someone should be released
before trial. ProPublica obtained COMPAS's scores for thousands of
defendants and matched them against what actually happened over the
following two years, then published the data.

This dataset is that data: each row is one defendant, with their
demographics and criminal history at the time of screening, COMPAS's
own risk score for them, and whether they were actually rearrested
within two years.

**Your task:** predict `two_year_recid` -- will this person be
rearrested within two years? -- from the case facts. Once you have a
model, the more interesting question is the one ProPublica actually
asked: is it equally accurate for everyone, or does it get things
wrong more often, in a particular direction, for some groups than
others? `race` is deliberately excluded from the model's own inputs
(see `config.yaml` and `src/preprocessing.py`) so it can be used
afterward purely to check this, in `src/evaluate.py`.

Before any of that: look at the data first. It comes from a real
system with real data-entry and record-keeping quirks -- don't assume
every column is clean or consistent just because it loads without
error.

## Data dictionary

| column | type | description | notable values |
|--------|------|--------------|------------------|
| `id` | identifier | internal record id | not a model feature |
| `sex` | categorical | defendant's sex | `Male`, `Female` |
| `age` | numeric | defendant's age (years) at screening | |
| `age_cat` | categorical | age bucket | `Less than 25`, `25 - 45`, `Greater than 45` |
| `race` | categorical | defendant's race, as recorded | `African-American`, `Caucasian`, `Hispanic`, `Asian`, `Native American`, `Other`; excluded from model features, used only to audit fairness |
| `juv_fel_count` | numeric | number of prior juvenile felony offenses | |
| `juv_misd_count` | numeric | number of prior juvenile misdemeanor offenses | |
| `juv_other_count` | numeric | number of other prior juvenile offenses | |
| `juvenile_total` | numeric | total juvenile offenses | |
| `priors_count` | numeric | number of prior adult offenses | |
| `prior_offenses` | numeric | number of prior offenses | |
| `age_in_months` | numeric | age expressed in months | |
| `c_charge_degree` | categorical | degree of the current charge | `F` (felony), `M` (misdemeanor) |
| `decile_score` | numeric | COMPAS's own risk score | 1 (lowest risk) to 10 (highest risk); excluded from model features, used only for comparison |
| `score_text` | categorical | COMPAS's own risk category | `Low`, `Medium`, `High`; excluded from model features, used only for comparison |
| `two_year_recid` | binary | **target** -- was this person rearrested within two years? | `0` = no, `1` = yes |

Source: derived from [propublica/compas-analysis](https://github.com/propublica/compas-analysis) (the data behind the "Machine Bias" investigation). Personally-identifying columns (name, date of birth, case numbers, charge descriptions) were removed.


# Week 1

These were the models created:
* **Logistic Regression:** with parameters `{'max_iter': 1000}`
* **Decision Tree:** without specific parameters

### Metrics Comparison

| Model | Train Accuracy | Test Accuracy | Gap (Train - Test) |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | 67.8% | 67.9% | -0.001 |
| **Decision Tree** | 82.9% | 62.8% | +0.201 |

### The main differences:

- In the **logistic regression** model, it achieved a better accuracy (68%). It is very stable because the training (67.8%) and testing (67.9%) scores are almost identical. This means it learned the patterns correctly without just memorizing the data.
- The **decision tree** model achieved a lower overall accuracy (63%). This model "memorized" the training data (scoring 82.9%) but performed poorly on new data (62.8%). This indicates overfitting. However, it did show a slightly smaller gap in false positive rates between the main racial groups.

**In summary:**
The logistic regression model was better. Even though the decision tree has slightly more balanced numbers regarding race, the fact that it overfitted makes it unreliable for real-world scenarios. The logistic regression is much more robust, consistent, and makes more accurate predictions on new cases.


# Week 2

These were the models created:
* **Logistic Regression:** with parameters `{'max_iter': 2000}`
* **Decision Tree:** without specific parameters

### Metrics Comparison

| Model | Train Accuracy | Test Accuracy | Gap (Train - Test) |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | 67.6% | 65.7% | +0.019 |
| **Decision Tree** | 79.2% | 61.5% | +0.177 |

### The main differences:

- In the **logistic regression** model, it achieved a better accuracy (66%). It is very stable because the training (67.6%) and testing (65.7%) scores are very close. This means it learned the patterns correctly without just memorizing the data.
- The **decision tree** model achieved a lower overall accuracy (62%). This model "memorized" the training data (scoring 79.2%) but performed worse on new data (61.5%). This indicates overfitting. However, it did show a smaller gap in false positive rates between the main racial groups (a 10% gap compared to the logistic regression's 14% gap). 

**In summary:**
The logistic regression model was better. Even though the decision tree has slightly more balanced numbers regarding race, the fact that it overfitted makes it unreliable for real-world scenarios. The logistic regression is much more robust, consistent, and makes more accurate predictions on new cases.


# Week 3 and Week 4

### Pipeline Improvements

This week, we upgraded the pipeline to make our tests more reliable and accurate:
* **Added a Baseline Model:** We introduced a Dummy model to see the minimum performance required to actually add value.
* **New Preprocessing:** Added `Target Encoder` (to handle categories better) and `Robust Scaler` (to handle data outliers).
* **Advanced Tuning:** Used `Optuna` to automatically find the best parameters (hyperparameter tuning) using nested cross-validation. This prevents models from cheating or memorizing data.
* **New Model:** Added a Random Forest model to the experiment.
* **Hidden Test Set:** We locked away 20% of the data to test the final model at the very end.

### Metrics Comparison

| Model | Train Accuracy | Validation Accuracy | Gap (Train - Val) |
| :--- | :--- | :--- | :--- |
| **Dummy (Baseline)** | 54.9% | 54.9% | 0.000 |
| **Decision Tree (Tuned)**| 68.4% | 67.5% | +0.009 |
| **Logistic Regression** | 67.5% | 67.2% | +0.003 |
| **Random Forest** | 73.1% | 64.6% | +0.085 |

### The main differences:

- The **Tuned Decision Tree** had a massive improvement. In Week 2, it overfitted heavily (17.7% gap). Thanks to the new tuning process, the gap dropped to just 0.9%. It is now the most accurate model (67.5%).
- The **Logistic Regression** remains very stable and consistent. It achieved 67.2% accuracy with almost zero gap (0.3%), meaning it generalizes perfectly to new data.
- The **Random Forest** suffered from overfitting. It memorized the training data (73.1%) but dropped to 64.6% on validation, creating an 8.5% gap. 
- **Fairness (False Positive Rates):** Both the Decision Tree and Logistic Regression have a 13% gap in false positives between African-American and Caucasian groups. This is a significant improvement over the original COMPAS system, which has a much worse 22% gap.

**In summary:**
The pipeline upgrades successfully fixed the Decision Tree's overfitting problem from Week 2. Both the **Tuned Decision Tree** and **Logistic Regression** are now excellent, stable choices that perform much better than the baseline and are fairer than the original COMPAS tool.

j
# Week 5 and Week 6

### Pipeline Improvements

This week, we expanded our model benchmark while keeping the same preprocessing (`Target Encoder` + `Robust Scaler`), 5-fold stratified cross-validation setup, and the 20% locked test set from Week 3:
* **Added AdaBoost:** Introduced an adaptive boosting ensemble (`adaboost`) using default parameters to test if sequential error correction improves validation accuracy.
* **Added Oblique Random Forest:** Introduced an `oblique_random_forest` model (which uses linear combinations of features at each split instead of single-feature axis-aligned splits) to evaluate more complex decision boundaries.
* **Standardized Fixed Configurations:** Evaluated all 6 models with fixed configurations (`Tuning: off`) to compare baseline architectures directly.

### Metrics Comparison

| Model | Train Accuracy | Validation Accuracy | Gap (Train - Val) |
| :--- | :--- | :--- | :--- |
| **Dummy (Baseline)** | 54.9% | 54.9% | -0.000 |
| **Random Forest** | 70.3% | 67.7% | +0.025 |
| **Decision Tree** | 68.5% | 67.5% | +0.011 |
| **Logistic Regression** | 67.0% | 66.9% | +0.001 |
| **AdaBoost (New)** | 67.1% | 66.8% | +0.003 |
| **Oblique Random Forest (New)** | 75.8% | 64.6% | +0.113 |

### The main differences:

- **Random Forest & Decision Tree** lead the overall performance. With constrained depth (`max_depth=10`), the **Random Forest** achieved the highest validation accuracy (67.7%) with a controlled gap (2.5%), closely followed by the **Decision Tree** (67.5% accuracy, 1.1% gap).
- **AdaBoost (New)** performed very similarly to **Logistic Regression**. It reached 66.8% validation accuracy with an almost nonexistent gap (0.3%), showing strong stability and zero overfitting right out of the box.
- **Oblique Random Forest (New)** suffered from heavy overfitting. Because it ran without depth constraints (`params={}`), it memorized the training data (75.8%) and dropped to 64.6% on validation, resulting in an 11.3% gap—the worst generalization among all models.
- **Fairness (False Positive Rates):** All of our models remain fairer than the original COMPAS score (which has a 22% FPR gap between African-American [0.45] and Caucasian [0.23] groups). Among the new models, **AdaBoost** and **Oblique Random Forest** both recorded a **13% FPR gap** (0.33 vs. 0.20 for AdaBoost; 0.35 vs. 0.23 for Oblique RF). This matches the 13% gap of the **Decision Tree** (0.28 vs. 0.15) and **Random Forest** (0.30 vs. 0.17), and outperforms **Logistic Regression** (18% gap: 0.40 vs. 0.22).

**In summary:**
Adding the two new models showed that **AdaBoost** is a very stable runner-up that generalizes well (66.8% accuracy, 0.3% gap) and improves fairness over Logistic Regression. Meanwhile, the unconstrained **Oblique Random Forest** overfitted heavily (11.3% gap) and needs hyperparameter tuning (like limiting `max_depth`) to compete with the **Random Forest** (67.7%) and **Decision Tree** (67.5%).


# Week 7

### Pipeline Improvements

This week, we upgraded our evaluation framework and expanded our model benchmark to get a more complete and honest assessment of model performance:
* **Expanded Metrics & New Selection Target:** Shifted our primary optimization and selection metric from simple `accuracy` to `roc_auc` (evaluating across `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`, `fbeta_2`, `roc_auc`, and `average_precision`). This evaluates how well models rank risk across all thresholds rather than relying on a single fixed cutoff.
* **Honest Nested Cross-Validation:** Turned on `Optuna` hyperparameter tuning (30 trials × 5 inner folds) inside a 5-fold outer Nested CV loop, including an **optimism check** (`inner_best` vs. `outer_validation`) to ensure zero data leakage or overfitting during tuning.
* **Added 3 New Models:** Introduced `gradient_boosting`, `hist_gradient_boosting`, and `naive_bayes` alongside our existing tree and ensemble models.

### Metrics Comparison (Nested CV Honest Estimates)

| Model | Untuned ROC AUC | Tuned ROC AUC (Val) | ROC AUC Gap (Train - Val) | Tuned Accuracy | Tuned F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy (Floor)** | 0.500 | 0.500 | 0.000 | 54.9% | 0.000 |
| **Hist Gradient Boosting (New)** | 0.717 | **0.728** | +0.010 | **68.5%** | **0.618** |
| **Gradient Boosting (New)** | 0.727 | **0.728** | +0.013 | 68.2% | 0.611 |
| **AdaBoost** | 0.718 | 0.727 | +0.015 | 68.2% | 0.613 |
| **Random Forest** | 0.683 | 0.725 | +0.022 | 67.9% | 0.612 |
| **Oblique Random Forest** | 0.683 | 0.725 | +0.021 | 67.7% | 0.609 |
| **Decision Tree** | 0.609 | 0.716 | +0.015 | 67.3% | 0.593 |
| **Naive Bayes (New)** | 0.685 | 0.705 | +0.001 | 62.1% | 0.368 |

### The main differences:

- **Boosting Models Take the Lead:** **Hist Gradient Boosting** and **Gradient Boosting** tied for the highest honest validation `roc_auc` (0.728), closely followed by **AdaBoost** (0.727). **Hist Gradient Boosting** also achieved the highest validation accuracy (68.5%) and F1-score (0.618) while keeping the train-validation gap remarkably low (+0.010).
- **Massive Recovery for Overfitting Models:** Without tuning, **Decision Tree** (0.609 AUC, +0.110 gap), **Random Forest** (0.683 AUC, +0.118 gap), and **Oblique Random Forest** (0.683 AUC, +0.149 gap) overfitted heavily. Optuna's nested tuning (especially constraining `min_samples_leaf` and `max_depth`) eliminated this overfitting, shrinking their gaps to ~1.5%–2.2% and boosting both Random Forest variants to a competitive 0.725 `roc_auc`.
- **Zero Tuning Optimism:** Across all models, the optimism check (`inner_best` minus `outer_validation`) was between `-0.001` and `+0.001`, proving that our Nested CV pipeline completely prevented hyperparameter overfitting.
- **Naive Bayes Trade-off:** While **Naive Bayes** generalized with virtually zero gap (+0.001) and improved from 0.685 to 0.705 `roc_auc` via `var_smoothing`, its default 0.5 threshold is overly conservative—yielding high precision (0.743) but very low recall (0.245), resulting in the lowest accuracy (62.1%).
- **Fairness (False Positive Rates):** Every model substantially improved upon COMPAS's original 22% FPR gap between African-American (0.45) and Caucasian (0.23) groups:
  - **Gradient Boosting** achieved a **12% gap** (0.28 vs. 0.16).
  - **AdaBoost** (0.29 vs. 0.17) and **Decision Tree** (0.29 vs. 0.16) achieved a **12%–13% gap**.
  - **Hist Gradient Boosting** (0.29 vs. 0.15), **Random Forest** (0.30 vs. 0.16), and **Oblique Random Forest** (0.30 vs. 0.16) all recorded a **14% gap**.
  - **Naive Bayes** had the lowest absolute FPRs (**7% gap**: 0.11 vs. 0.04), though this is a byproduct of its low overall positive prediction rate (24% recall).

**In summary:**
Switching to `roc_auc` and enabling Nested CV with `Optuna` cured the severe overfitting seen in our tree and forest models while confirming zero tuning leakage. **Hist Gradient Boosting** and **Gradient Boosting** are now our top performers (0.728 `roc_auc`, ~68.5% accuracy), offering the best balance of predictive power, generalization, and reduced racial disparity compared to the original COMPAS score.
