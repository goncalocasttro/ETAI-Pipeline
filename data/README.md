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


# Week 3

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