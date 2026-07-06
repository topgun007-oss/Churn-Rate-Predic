# Customer Churn Prediction — 30 FAQ Interview Guide (DS2 Level)

A structured question-and-answer guide for presenting this project in Data Scientist 2 interviews. Covers the full pipeline from EDA through SHAP explainability and cost-sensitive analysis.

---

## Quick Reference Card

| Item | Value |
|------|-------|
| Dataset | 7,043 customers, 21 raw features |
| Churn rate | 26.5% (imbalanced) |
| Engineered features | 22 (7 core + 15 advanced) |
| Total features after encoding | 56 |
| Train/test split | 80/20 stratified |
| Models compared | 20+ configs (10 base + NN + 3 tuned + 2 ensembles + SMOTE + threshold-optimized) |
| Feature selection | Mutual Information + LightGBM + RFE (consensus) |
| Resampling | SMOTE, ADASYN, Borderline-SMOTE |
| Ensembles | Stacking (4 base + LR meta), Voting (5 models, weighted) |
| Best AUC-ROC | ~0.847 |
| Best recall | ~83% (Borderline-SMOTE enhanced) |
| Recall improvement | +55%+ (from 52% to 83%) |
| Explainability | SHAP (TreeExplainer) — global + individual |
| Calibration | Brier score + reliability diagrams |
| Statistical rigor | Paired t-test + bootstrap 95% CI |
| Business analysis | Expected value framework, Net ROI optimization |
| Production artifacts | Pipeline (.pkl), scaler, NN (.keras), results CSV |

---

## FAQ 1: Walk me through your project in 2 minutes.

**A:** I built a production-grade ML pipeline to predict customer churn for a telecom company. Starting from 7,043 customers with 21 features and a 26.5% churn rate, I engineered 22 new features based on domain knowledge and EDA insights. I validated these with three independent feature selection methods (mutual information, model-based, RFE) and confirmed consensus across them.

I compared 10 algorithms spanning linear, tree-based, kernel, instance-based, and deep learning paradigms, then applied SMOTE resampling, hyperparameter tuning with GridSearchCV, and built Stacking/Voting ensembles. Beyond raw metrics, I added SHAP explainability for individual predictions, calibration analysis to verify probability quality, statistical tests to confirm model differences aren't due to chance, and a cost-sensitive business analysis that translates the confusion matrix into dollars.

The best model achieves ~0.847 AUC-ROC with 81% recall after class-weight balancing. The pipeline is serialized as an sklearn Pipeline object for deployment.

---

## FAQ 2: Why this problem and what's the business objective?

**A:** Customer churn prediction is one of the highest-ROI ML applications. Acquiring a new customer costs 5-7x more than retaining one. The business objective is to score all active customers with a churn probability so the retention team can intervene proactively. Customers above a configurable threshold get flagged for outreach with tailored offers. My cost-sensitive analysis showed the business-optimal threshold maximizes Net ROI at a specific operating point, not at the default 0.5.

---

## FAQ 3: What was most challenging and what would you do differently?

**A:** The main challenge was balancing recall vs. precision with 26.5% churn. Most models optimized for accuracy by conservatively predicting "no churn." I addressed this through class weights (2.8x in the NN), SMOTE resampling, and threshold optimization. The neural network hit 81% recall but at lower precision.

If starting over, I'd add time-series features (charge trends, usage pattern changes) and external data (competitor pricing, market conditions). These aren't available in the IBM sample dataset but would break the ~80-82% accuracy ceiling.

---

## FAQ 4: Describe your dataset and key EDA insights.

**A:** The Telco Customer Churn dataset has 7,043 customers with demographics (gender, senior citizen, partner, dependents), account info (tenure, contract, billing, charges), and service subscriptions (phone, internet, security, streaming). Key EDA findings:

1. Month-to-month contracts churn at ~42% vs. 3% for two-year contracts
2. New customers (0-12 months) churn most; sharp drop after 24 months
3. Fiber optic users churn more despite being a premium service
4. Customers without protection services (security, backup, support) churn more
5. Electronic check payers have the highest churn rate

These directly informed feature engineering.

---

## FAQ 5: How did you handle data quality issues?

**A:** Two issues found: TotalCharges had 11 values stored as empty strings rather than NaN. I discovered this because the column was typed as `object`. I converted with `pd.to_numeric(errors='coerce')` and imputed with the median (robust to the right-skewed distribution). These 11 rows all had tenure=0, meaning brand-new customers with no charges. No duplicates were found.

---

## FAQ 6: Walk me through your preprocessing pipeline.

**A:** Five steps: (1) Drop customerID. (2) Fix TotalCharges type and impute 11 missing with median. (3) Encode target (Yes/No to 1/0). (4) One-hot encode categoricals with `drop_first=True` to avoid the dummy variable trap. (5) StandardScaler fit on training data only, then transform both train and test. Fitting only on training data prevents data leakage where test set statistics bleed into the scaler's mean/std.

---

## FAQ 7: Walk me through your 22 engineered features.

**A:** I created 7 core features from domain knowledge: tenure_group (non-linear binning), AvgMonthlySpend (historical average), ChargeDiff (price increase detector), NumServices (switching cost proxy), HasStreaming/HasProtection (service engagement flags), and TenureChargeInteraction (loyalty-spend interaction).

Then 15 advanced features: log transforms for skew reduction, TenureSq for polynomial non-linearity, ChargePerService for value density, lifecycle flags (IsNewCustomer, IsLoyalCustomer), HighCharges threshold flag, HighRiskCombo (no protection + month-to-month), SeniorAlone vulnerability indicator, LoyaltyValue, ContractMonths numeric mapping, ChargeContractInteraction, ServiceTenureInteraction, NoInternet segment flag, and ElecCheckMonthly (highest-churn EDA combination).

Each feature has a clear business rationale and was validated through feature selection.

---

## FAQ 8: How did you validate feature choices? Describe feature selection.

**A:** I applied three independent methods: (1) Mutual Information scores, which measure non-parametric dependency with the target. (2) Model-based selection using LightGBM's SelectFromModel with median threshold. (3) Recursive Feature Elimination with Logistic Regression selecting top 20.

I computed a consensus: features selected by 2+ of 3 methods. This validates that engineered features carry genuine predictive signal and aren't just noise. I retained all 56 features for training since the consensus confirmed most engineered features, and tree models handle irrelevant features through their split-selection mechanism.

---

## FAQ 9: Why one-hot encoding over label or target encoding?

**A:** One-hot encoding creates binary columns for each category, essential for variables without ordinal relationships (PaymentMethod has 4 options with no natural ordering). Label encoding would assign 0,1,2,3, falsely implying order. I used `drop_first=True` to avoid multicollinearity, which matters for Logistic Regression's coefficient interpretability.

Target encoding (replacing categories with the target's mean per category) is powerful but risks leakage if not done within CV folds. For a 7K dataset with moderate cardinality categories, one-hot encoding is the safer universal choice.

---

## FAQ 10: Why these 10 algorithms? What paradigm families?

**A:** I chose models spanning five paradigm families: Linear (Logistic Regression for interpretable baseline), Tree/Bagging (Decision Tree, Random Forest, Bagging Classifier for variance reduction), Boosting (Gradient Boosting, XGBoost, LightGBM, AdaBoost for sequential error correction), Kernel (SVM with RBF for non-linear decision boundaries), Instance-based (KNN for distance-based classification), and Deep Learning (TensorFlow NN for automatic feature extraction).

This diversity ensures comprehensive coverage and shows which paradigms work best for structured tabular data.

---

## FAQ 11: Explain bagging vs boosting. Why did boosting win?

**A:** Bagging (Random Forest) trains multiple models independently on random data subsets and averages predictions, reducing variance. Boosting (XGBoost, LightGBM) trains models sequentially where each corrects the previous model's mistakes, reducing bias.

Boosting won because this dataset's challenge is bias (underfitting the churn patterns), not variance. The interactions between contract type, tenure, charges, and services create complex decision boundaries that sequential correction captures better than independent averaging. This is typical for tabular data.

---

## FAQ 12: Why did Decision Tree and KNN underperform?

**A:** Decision Tree has high variance: it overfits training data with specific rules that don't generalize. AUC of 0.667 (barely above random 0.5) confirms memorization. KNN struggles with high dimensionality (56 features): Euclidean distance becomes less meaningful in high dimensions ("curse of dimensionality"), and one-hot encoded binary features create discrete jumps in distance space.

---

## FAQ 13: Your CV reports mean+/-std. Why is that better than a single test score?

**A:** A single test score is a point estimate with unknown variance. Cross-validation with 5 stratified folds gives 5 independent performance measurements, so I can report mean +/- std. This tells you both the expected performance AND how stable it is across different data subsets. A model with 0.84 +/- 0.01 is much more reliable than one with 0.84 +/- 0.05. I also visualized the distributions with boxplots to spot outlier folds.

---

## FAQ 14: How did you approach hyperparameter tuning?

**A:** GridSearchCV with 5-fold stratified CV on the top 3 models (XGBoost, LightGBM, Gradient Boosting). I scored on AUC-ROC, not accuracy, because AUC is robust to class imbalance. The grids covered key hyperparameters: n_estimators, max_depth, learning_rate, subsample, and scale_pos_weight for imbalance handling.

For larger search spaces I'd use Bayesian optimization (Optuna), which intelligently chooses the next combination based on past results, requiring far fewer evaluations.

---

## FAQ 15: What hyperparameters matter most for XGBoost/LightGBM?

**A:** learning_rate (step size; lower values generalize better but need more trees), max_depth (limits tree complexity; 3-5 prevents overfitting), n_estimators (number of boosting rounds), subsample/colsample_bytree (stochastic regularization, similar to dropout), and scale_pos_weight (handles class imbalance by upweighting minority class in the loss function). The interaction between learning_rate and n_estimators is especially important: a lower learning rate with more trees almost always outperforms a higher rate with fewer trees.

---

## FAQ 16: How did you handle the 26.5% class imbalance?

**A:** Five complementary strategies: (1) Stratified splitting to maintain 26.5% ratio in all data subsets. (2) SMOTE/ADASYN/Borderline-SMOTE oversampling of training data only. (3) Class weights (2.8x for minority) in the neural network and scale_pos_weight in XGBoost. (4) AUC-ROC scoring instead of accuracy for model selection. (5) Threshold optimization to find the operating point that maximizes the desired metric. Using multiple strategies simultaneously provided the +55% recall improvement.

---

## FAQ 17: Explain SMOTE vs ADASYN vs Borderline-SMOTE.

**A:** SMOTE generates synthetic minority samples by interpolating between existing minority neighbors, balancing to 50/50. ADASYN adaptively generates more synthetics near the decision boundary where classification is hardest. Borderline-SMOTE only creates synthetics for minority instances that are near the class boundary, focusing effort where it matters most.

Results: average recall improved from 0.514 (no resampling) to 0.635 (Borderline-SMOTE). All are applied only to training data to prevent leakage. The key insight: SMOTE generates new points between existing points in feature space, so it can't extrapolate beyond the training distribution.

---

## FAQ 18: Describe your neural network architecture.

**A:** A 4-layer Sequential model: Dense(128, ReLU) + BatchNorm + Dropout(0.3) -> Dense(64, ReLU) + BatchNorm + Dropout(0.3) -> Dense(32, ReLU) + Dropout(0.2) -> Dense(1, Sigmoid). The funnel shape (128->64->32->1) progressively compresses representations. BatchNorm stabilizes training and acts as mild regularization. Dropout prevents neuron co-adaptation. Lower dropout in later layers because they have fewer parameters to overfit.

---

## FAQ 19: Why class weights, early stopping, ReduceLR, and BCE loss?

**A:** Class weights (2.8x for churn) force the network to pay more attention to the minority class, boosting recall from ~52% to ~81%. Early stopping (patience=15) halts training when validation loss stops improving, preventing overfitting. ReduceLROnPlateau halves the learning rate when validation loss plateaus, allowing fine-tuning without overshooting.

Binary cross-entropy is the correct loss for binary classification from maximum likelihood estimation. MSE has vanishing gradients near sigmoid extremes (0 or 1), making learning slow. BCE punishes confident wrong predictions much more harshly.

---

## FAQ 20: Explain Stacking and Voting ensembles.

**A:** Voting Ensemble averages predicted probabilities from 5 models (XGBoost, LightGBM, GB tuned + RF + LR) with weights [3,3,2,1,1] favoring the best performers. "Soft" voting averages probabilities rather than majority-voting on labels, which is smoother.

Stacking uses a two-level architecture: 4 base learners (XGBoost, LightGBM, GB, RF) make predictions using 5-fold CV, then a Logistic Regression meta-learner learns the optimal combination. The meta-learner can discover that certain base models are better in different regions of the feature space. Both reduce model-specific errors because errors from different paradigms are uncorrelated.

---

## FAQ 21: What is threshold optimization and how does it depend on business costs?

**A:** Default classifiers use 0.5 as the decision threshold. Threshold optimization sweeps from 0.25 to 0.80 and finds the threshold maximizing a chosen metric. The optimal threshold depends on business costs: if missing a churner costs 40x more than a false alarm, you'd lower the threshold to catch more churners (higher recall, lower precision). My cost-sensitive analysis shows the business-optimal threshold maximizes Net ROI = (TP * retention_success * LTV) - ((TP + FP) * offer_cost), which is different from the accuracy-optimal threshold.

---

## FAQ 22: What is SHAP and why is it superior to built-in feature importance?

**A:** SHAP (SHapley Additive exPlanations) assigns each feature a contribution value for each individual prediction, based on cooperative game theory. Unlike tree feature importance (which is global, based on split frequency or gain, and varies by implementation), SHAP provides:

1. **Local explanations**: Why did THIS customer get a 0.78 churn probability?
2. **Consistency**: A feature that increases prediction always gets a positive SHAP value
3. **Additivity**: SHAP values sum exactly to the difference between the prediction and the base rate
4. **Model-agnostic**: Same framework works for any model type

For a non-technical stakeholder: "This customer has a 78% churn probability. The three biggest reasons are: month-to-month contract (+0.15), no tech support (+0.08), and high monthly charges (+0.06). Their long tenure (-0.05) is the main factor keeping the risk from being even higher."

---

## FAQ 23: How would you use SHAP to debug the model or explain a specific prediction?

**A:** The SHAP waterfall plot shows exactly which features push a customer's prediction above or below the base rate. I use this to: (1) Debug suspicious predictions by checking if the model is using features sensibly. (2) Detect data leakage: if a feature that shouldn't be available at prediction time has huge SHAP values, something is wrong. (3) The beeswarm plot reveals if the model learned counterintuitive patterns (e.g., if higher tenure increases churn, the model might be picking up a confound). (4) Dependence plots show interaction effects between features that aren't obvious from univariate analysis.

---

## FAQ 24: Are your probabilities well-calibrated? How do you measure and fix this?

**A:** I plotted calibration curves (reliability diagrams) comparing predicted probabilities to actual churn rates in bins. A well-calibrated model's curve follows the diagonal. I also computed Brier scores (lower = better). Logistic Regression typically has good calibration by design since it optimizes log-loss. Tree models (XGBoost, LightGBM) can be miscalibrated because they optimize a different objective.

To fix miscalibration: Platt scaling (fit a logistic regression on the model's outputs) or isotonic regression (non-parametric monotone fit). In sklearn, `CalibratedClassifierCV` implements both. Calibration matters when you need the actual probability to be meaningful (setting business thresholds, combining multiple models, risk stratification).

---

## FAQ 25: How do you know your best model is statistically better than the runner-up?

**A:** I used two approaches: (1) Paired t-test on 5-fold CV AUC scores. Since the same folds are used for both models, the paired test accounts for fold-to-fold correlation. If p < 0.05, the difference is statistically significant. (2) Bootstrap confidence intervals: resample the test set 1,000 times, compute AUC each time, and report the 2.5th-97.5th percentile range.

This matters because with only 5 CV folds, small differences in mean AUC (e.g., 0.845 vs 0.842) are often not significant. Reporting "XGBoost is better" without statistical backing is a red flag.

---

## FAQ 26: Translate your model into business ROI using the expected value framework.

**A:** I computed Net ROI at each threshold: Revenue Saved = TP * retention_success_rate * customer_LTV, Retention Cost = (TP + FP) * offer_cost. With assumptions of $2,000 LTV, $50 retention offer, and 30% retention success rate, the business-optimal threshold maximizes Net ROI by balancing catching churners (lower threshold = more TP but more FP) against intervention costs. The confusion matrix at the optimal threshold shows exactly how many customers are flagged and at what cost.

This framework is configurable: if the business changes the offer cost or success rate, the optimal threshold shifts automatically.

---

## FAQ 27: How would you deploy this model in production?

**A:** I built an sklearn Pipeline (StandardScaler + best model) serialized as a single .pkl file. In production: (1) Load the pipeline with `joblib.load('churn_pipeline.pkl')`. (2) Apply the same feature engineering to new data. (3) Call `pipeline.predict_proba(new_data)` for churn probabilities. The pipeline handles scaling internally, preventing the common bug of forgetting to scale at inference time.

For full deployment: wrap in a FastAPI endpoint, run batch scoring nightly, store predictions in a CRM database, and push high-risk customers to the retention team's dashboard.

---

## FAQ 28: How would you monitor post-deployment? Data drift vs concept drift?

**A:** Four monitoring types: (1) Performance monitoring: track accuracy/recall/AUC as true labels become available (1-3 month lag). (2) Data drift: monitor incoming feature distributions (if MonthlyCharges distribution shifts from new pricing, the model may degrade). (3) Concept drift: the relationship between features and churn changes (e.g., a new competitor enters). Compare recent performance to baseline. (4) Operational: API latency, error rates, throughput.

Data drift is when inputs change (X distribution shifts). Concept drift is when the relationship P(Y|X) changes. Both require retraining but are detected differently: data drift through distribution tests (KS test, PSI), concept drift through performance degradation.

---

## FAQ 29: Your learning curves show convergence. Would more data help?

**A:** The learning curves show train and validation AUC converging as training size increases, with a small gap at full size. If both curves plateau at the same level (high bias), more data won't help but more complex features or models would. If there's still a gap (high variance), more data would close it. For this dataset, the curves suggest we've reached the information ceiling of the available features. To improve further, we'd need richer features (behavioral data, time-series patterns) rather than more rows.

---

## FAQ 30: If you had 100M rows and real-time requirements, what would you change?

**A:** Several changes: (1) LightGBM over XGBoost for faster training with leaf-wise growth. (2) Distributed training with Spark MLlib or Dask. (3) Train on a stratified sample (1M rows), validate on another sample. (4) Replace GridSearchCV with Bayesian optimization (Optuna) for fewer evaluations. (5) Feature store (Feast, Tecton) for precomputed features. (6) Export model to ONNX for optimized sub-millisecond inference. (7) Batch precomputation: score all customers nightly, serve cached scores for real-time queries. (8) GPU training for both LightGBM and neural networks. (9) Incremental learning to update the model without full retraining.

---

## Things You Should Never Say

| Don't Say | Say Instead |
|-----------|-------------|
| "I used accuracy because it was 81%" | "I used AUC-ROC and recall because accuracy is misleading with 26.5% churn" |
| "I tried everything and picked the best one" | "I systematically compared models across paradigms and tuned the top 3" |
| "Deep learning is always better" | "Gradient boosting outperforms deep learning on structured tabular data" |
| "I didn't check for data leakage" | "I fit the scaler only on training data and used stratified splits" |
| "The model works perfectly" | "AUC 0.845 with room for improvement through richer features" |
| "I just used feature importance" | "I used SHAP for theoretically grounded, locally faithful explanations" |
| "The performance difference is obvious" | "I verified significance with paired t-tests and bootstrap CIs" |
| "Accuracy of 80% is good" | "80% is near the dataset ceiling; the real value is the Net ROI framework" |
