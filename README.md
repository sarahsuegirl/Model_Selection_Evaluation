
# Facebook Ads Click Prediction & Model Optimization

An end-to-end Machine Learning pipeline that predicts whether a user will click on a Facebook ad based on site engagement (`Time Spent on Site`) and demographic data (`Salary`).

This project demonstrates a rigorous model benchmark pipeline using **Scikit-Learn's `Pipeline**` and **`GridSearchCV`** to systematically evaluate multiple classification algorithms and optimize hyperparameter configurations using ROC-AUC scoring.

<img width="598" height="432" alt="image" src="https://github.com/user-attachments/assets/2c54cc11-a855-41cb-ba52-5b3f535ed1ae" />



##  Business Problem & Impact

Digital ad targeting requires identifying high-intent users to optimize ad spend and maximize Click-Through Rates (CTR). By modeling user behavior against demographic features, marketing teams can:

* **Target High-Probability Convertors:** Allocate budget toward users most likely to engage.
* **Reduce Ad Fatigue:** Avoid serving ads to demographics with low click intent.
* **Optimize Campaign ROI:** Leverage automated classification to streamline audience segmentation.

## Data Preprocessing & Pipeline Architecture


[ Raw Data ] ──► [ Drop Identifier Cols ] ──► [ MinMaxScaler ] ──► [ Pipeline + GridSearchCV ] ──► [ Best Model ]



1. **Feature Cleansing:** Irrelevant PII identifiers (`Names`, `emails`, `Country`) were dropped to prevent noise and data leakage.
2. **Feature Scaling:** `MinMaxScaler` was applied to normalize numerical features (`Time Spent on Site` and `Salary`) into the $[0, 1]$ range, ensuring fair distance calculations for models like KNN and Logistic Regression.
3. **Automated Model Benchmarking:** Implemented a single, extensible Scikit-Learn `Pipeline` combined with `GridSearchCV` (3-fold cross-validation) to evaluate **5 distinct classifier families** across **95 hyperparameter combinations**.

---

##  Model Comparison & Hyperparameter Tuning

The pipeline evaluated the following algorithms:

* **Support Vector Classifier (SVC):** Tuned `C` regularization parameters $[0.01, 0.1, 1, 10, 100]$.
* **Logistic Regression:** Tuned L1 and L2 penalty norms.
* **k-Nearest Neighbors (KNN):** Tuned $k$ values $[2, 5, 10, 15, 20]$.
* **Bernoulli Naive Bayes:** Tuned smoothing parameter $\alpha$.
* **Random Forest Classifier:** Evaluated combinations of `n_estimators`, `max_depth`, and `max_leaf_nodes`.

### 🏆 Best Model Results

| Metric | Winning Configuration |
| --- | --- |
| **Top Model** | **Random Forest Classifier** |
| **Best Hyperparameters** | `max_depth=3`, `max_leaf_nodes=7`, `n_estimators=50` |
| **Validation ROC-AUC Score** | **`0.9572` (95.72%)** |

> **Key Finding:** Ensemble tree methods outperformed distance and linear models, capturing non-linear interactions between `Time Spent on Site` and `Salary` without overfitting (controlled via low `max_depth`).

---

##  Repository Structure

```text
├── facebook_ads.csv                 # Raw dataset (499 records, 6 features)
├── facebook_ads_classification.ipynb # Complete Jupyter Notebook analysis
├── README.md                        # Documentation
└── requirements.txt                 # Project dependencies



##  Key Takeaways for Production

* **Pipeline Modularity:** Using Scikit-Learn `Pipeline` guarantees clean cross-validation by preventing data leakage between train/test splits.
* **Hyperparameter Regularization:** Constraining `max_depth=3` in Random Forest prevented the model from memorizing training data while achieving a peak **0.957 ROC-AUC**.

---

