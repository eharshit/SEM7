# PMTSA Exam Notes

## 1. Discuss the process and significance of cross-validation for time series (e.g., rolling window). Why is standard k‑fold not valid?

**Process (Rolling Window)**
*   **Time series data is ordered**, so the model is always trained on past data and tested on future data.
*   **How it works**:
    *   Start with an initial training window (for example, Jan–Jun).
    *   Test the model on the next time period (for example, Jul).
    *   Move the window forward: train on Jan–Jul, test on Aug.
    *   Repeat this process until the end of the dataset.

**Significance**
*   Preserves the temporal order of observations.
*   Gives a realistic estimate of future forecasting performance.
*   Helps identify concept drift (changes in data patterns over time).
*   Reduces the risk of overfitting compared to a single train–test split.

**Why Standard k-Fold is Not Valid**
*   **Random Splitting**: Standard k-fold uses random splitting, which mixes past and future data.
*   **Data Leakage**: The model indirectly learns from future values.
*   **Violates Assumption**: It violates the core time series assumption that future data must not influence the past.
*   **Over-optimistic**: Leads to over-optimistic accuracy, which fails in real-world forecasting.

## 2. What is normalization? Describe Min–Max Normalization with an example.

**Normalization**
*   **Definition**: A data preprocessing technique used to scale numerical features to a common range.
*   **Purpose**: Ensures that features with large values do not dominate features with smaller values.
*   **Importance**: Critical for distance-based algorithms like KNN, K-means, and gradient descent–based models.
*   **Benefit**: Helps improve training stability and model performance.

**Min–Max Normalization**
*   **Definition**: Rescales data to a fixed range, usually [0, 1].
*   **Formula**: $X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$
*   **Example**:
    *   Given values: 10, 20, 30 ($X_{min} = 10, X_{max} = 30$)
    *   **For 10**: $\frac{10 - 10}{30 - 10} = 0$
    *   **For 20**: $\frac{20 - 10}{30 - 10} = 0.5$
    *   **For 30**: $\frac{30 - 10}{30 - 10} = 1$

## 3. Compare Bagging and Boosting with examples of algorithms using them.

**Bagging (Bootstrap Aggregating)**
*   **Process**: Multiple models are trained independenty on random samples (with replacement) and aggregated via voting/averaging.
*   **Goal**: Reduces **variance** and controls overfitting.
*   **Characteristics**: Works well with high-variance models; less sensitive to noise.
*   **Algorithms**: Random Forest, Bagged Decision Trees.

**Boosting**
*   **Process**: Models are trained sequentially; each new model corrects errors (misclassified points) of previous ones.
*   **Goal**: Reduces **bias** and improves overall accuracy.
*   **Characteristics**: More sensitive to noise/outliers; focuses on hard-to-predict points.
*   **Algorithms**: AdaBoost, Gradient Boosting, XGBoost.

**Comparison Summary**
*   **Training**: Bagging = Parallel; Boosting = Sequential.
*   **Main Objective**: Bagging = Reduce Variance; Boosting = Reduce Bias.

## 4. Describe the working principles of Fourier Transform, Wavelet Transform, and EMD for time series decomposition. Compare their use‑cases.

**1. Fourier Transform (FT)**
*   **Principle**: Decomposes signal into a sum of sine and cosine waves.
*   **Limitation**: Assumes stationarity (statistical properties don't change); loses time info.
*   **Use-case**: Identifying dominant periodic patterns in **stationary signals** (e.g., constant seasonal cycles).

**2. Wavelet Transform (WT)**
*   **Principle**: Uses small "wavelets" to decompose signal at multiple scales.
*   **Advantage**: Provides **both** time and frequency information.
*   **Use-case**: Detecting sudden changes, trends, or spikes in **non-stationary signals** (e.g., financial shocks).

**3. Empirical Mode Decomposition (EMD)**
*   **Principle**: Data-driven method that breaks signal into Intrinsic Mode Functions (IMFs).
*   **Advantage**: Adaptive; no predefined basis functions; handles non-linear data well.
*   **Use-case**: Analyzing complex, **non-linear, non-stationary** real-world signals (e.g., biomedical data).

## 5. Explain any two methods of handling missing values in a dataset.

**Method 1: Deletion Method**
*   **Process**: Removing rows or columns that contain missing values.
*   **Suitability**: Best when the amount of missing data is very small.
*   **Pros**: Simple and fast.
*   **Cons**: Can cause loss of important information; may introduce bias if missingness is not random.

**Method 2: Imputation Method**
*   **Process**: Replacing missing values with estimated values.
*   **Techniques**:
    *   **Mean/Median/Mode**: For general data.
    *   **Forward/Backward Fill**: Specifically for time series (preserves order).
*   **Pros**: Preserves dataset size and structure.
*   **Cons**: Can distort variance or weaken relationships if done blindly.

## 6. Describe the assumptions of Linear Regression and explain why they matter.

**Assumptions**
1.  **Linearity**: The relationship between independent and dependent variables is linear. (Violation = inaccurate predictions).
2.  **Independence of Errors**: Residuals are not correlated with each other. (Violation = misleading confidence intervals).
3.  **Homoscedasticity**: The variance of residuals is constant across all predictor values. (Violation = unreliable standard errors).
4.  **Normality of Errors**: Residuals follow a normal distribution. (Important for hypothesis testing).
5.  **No Multicollinearity**: Independent variables are not highly correlated. (Violation = unstable coefficients).

**Why They Matter**
*   Ensures coefficient estimates are **unbiased and reliable**.
*   Violations lead to reduced model accuracy and **make statistical inference (p-values) unreliable**.

## 7. Explain Hybrid Forecasting Models (Classical + ML). Give an example of an additive and a multiplicative hybrid approach.

**Hybrid Forecasting Models**
*   **Concept**: Combines classical time series models (captured trend/seasonality) with machine learning models (captures nonlinear patterns/residual structure).
*   **Goal**: Improve accuracy by leveraging strengths of both matrices.

**Additive Hybrid Approach**
*   **Formula**: Series = Trend + Seasonality + Residual
*   **Method**:
    *   Classical model (e.g., ARIMA) models Trend & Seasonality.
    *   ML model (e.g., Random Forest) models Residuals.
*   **Final Forecast**: ARIMA Forecast + ML Residual Forecast.

**Multiplicative Hybrid Approach**
*   **Formula**: Series = Trend × Seasonality × Residual
*   **Use Case**: When seasonality scales with the level of the series.
*   **Method**:
    *   Classical model captures multiplicative structure.
    *   ML model learns patterns in scaled residuals.
*   **Final Forecast**: Classical Forecast × ML Correction Factor.

## 8. Define Trend and Seasonality in time series with simple examples.

**Trend**
*   **Definition**: Long-term upward or downward movement in the data showing overall direction over time.
*   **Example**: Monthly sales increasing every year due to business growth.

**Seasonality**
*   **Definition**: Repeating patterns at fixed intervals (daily, monthly, yearly) caused by calendar effects or human behavior.
*   **Example**: Higher electricity usage every summer or increased retail sales during festivals.

## 9. Explain the concept of Gradient Boosting. How does XGBoost improve traditional boosting?

**Gradient Boosting**
*   **Concept**: An ensemble technique that builds models sequentially.
*   **Process**: Each new model tries to correct the errors (residuals) of the previous models using gradient descent to minimize a loss function.
*   **Base Learner**: Typically uses shallow decision trees.

**XGBoost Improvements**
*   **Regularization**: Uses L1 and L2 regularization to reduce overfitting.
*   **Speed**: Supports parallel processing for faster training.
*   **Robustness**: Handles missing values automatically.
*   **Efficiency**: Includes early stopping for better generalization and is scalable for large datasets.

## 10. A dataset exhibits complex seasonality, trend shifts, and noise. Propose a complete forecasting solution mixing decomposition, ML, and statistical models, and justify each step.

**Step 1: Data Preprocessing**
*   **Action**: Handle missing values, normalize if needed, and split data using rolling validation.
*   **Justification**: Ensures clean input and prevents data leakage, critical for time series.

**Step 2: Time Series Decomposition**
*   **Action**: Use **STL or EMD** to separate Trend, Seasonality, and Residuals (Noise).
*   **Justification**: Simplifies the problem by separating structured patterns from random noise.

**Step 3: Statistical Modeling**
*   **Action**: Apply **SARIMA or ETS** to model the Trend and Seasonality components.
*   **Justification**: Statistical models excel at capturing linear and periodic structures interpretably.

**Step 4: Machine Learning on Residuals**
*   **Action**: Train **XGBoost or Random Forest** on the Residuals using lag features and rolling stats.
*   **Justification**: ML captures non-linear relationships and complex interactions that statistical models miss.

**Step 5: Hybrid Combination & Evaluation**
*   **Action**: Add Statistical Forecast + ML Residual Forecast. Evaluate using RMSE/MAE via walk-forward validation.
*   **Justification**: Combines strengths of both approaches for robust accuracy under shifting conditions.

## 11. What is a Random Forest? How does it reduce overfitting?

**Random Forest**
*   **Definition**: Random Forest is an ensemble algorithm that builds many decision trees on random samples and features, then combines their outputs to reduce overfitting and improve accuracy. 
*   **Process**: Builds multiple decision trees on different bootstrapped samples of the data.
*   **Prediction**: The final output is the majority vote (classification) or average (regression) of all trees.

**How It Reduces Overfitting**
*   **Bagging**: Each tree sees a different random subset of data.
*   **Feature Randomness**: At each split, only a random subset of features is considered, reducing correlation between trees.
*   **Averaging**: Aggregating many diverse trees smooths out individual errors and noise, preventing any single tree from dominating.

## 12. Discuss the steps involved in SARIMA model construction and when it is preferred.

**Steps to Build SARIMA**
*   **Stationarity**: Check using plots and ADF test; apply differencing ($d$).
*   **Identification**: Identify non-seasonal parameters ($p, d, q$) and seasonal parameters ($P, D, Q$) using ACF/PACF plots.
*   **Fitting**: Fit the SARIMA $(p,d,q)(P,D,Q)_m$ model on training data.
*   **Diagnostics**: Check residuals for white-noise behavior.
*   **Validation**: Validate using rolling forecast and select best model via AIC/BIC.

**When SARIMA Is Preferred**
*   Data shows **clear seasonality** with a fixed period.
*   Relationship is mostly **linear**.
*   Strong need for **interpretability**.
*   Works well for short to medium-term forecasts.

## 13. Explain Gini Index and Information Gain used for attribute selection in decision trees.

**Gini Index**
*   **Concept**: Measures how often a randomly chosen sample would be misclassified. (Lower is purer).
*   **Usage**: Main criterion for **CART** trees.
*   **Formula**: $Gini = 1 - \Sigma(p_i^2)$

**Information Gain**
*   **Concept**: Measures the reduction in entropy (uncertainty) after a split. (Higher is better).
*   **Usage**: Main criterion for **ID3 and C4.5** algorithms.
*   **Formula**: $Gain = Entropy(parent) - Weighted Entropy(children)$

## 14. What is Logistic Regression? Explain its use in binary classification.

**Logistic Regression**
*   **Definition**: A classification algorithm used to predict binary outcomes (0 or 1).
*   **Mechanism**: Uses a **sigmoid function** to map outputs to a probability between 0 and 1.
*   **Modeling**: It models the probability of the positive class ($P(Y=1|X)$).

**Use in Binary Classification**
*   **Thresholding**: Applies a threshold (usually 0.5) to probability to assign class labels.
*   **Applications**: Spam detection, disease prediction, churn prediction.
*   **Pros**: Easy to interpret and efficient for linearly separable data.

## 15. Explain how Trend, Seasonality, and Cyclic components can be visually analyzed in time series data.

**Trend**
*   **Visual**: Observed as a long-term upward or downward movement in line plots.
*   **Analysis**: Identified using smoothing techniques like moving averages.

**Seasonality**
*   **Visual**: Regular repeating patterns at fixed intervals (peaks and troughs).
*   **Analysis**: Visualized using **seasonal subseries plots** or monthly box plots.

**Cyclic Component**
*   **Visual**: Irregular long-term fluctuations without fixed periods (economic cycles).
*   **Analysis**: Detected by observing peaks and troughs over multiple years (longer duration than seasonality).

## 16. Discuss the challenges of applying Random Forests and Boosting on temporal data, and propose solutions such as sliding windows and feature engineering.

**Challenges**
*   **No Time Awareness**: Standard ML models (RF/Boosting) shuffle data, breaking temporal dependency.
*   **Extrapolation**: They cannot predict trends outside the range of training values.
*   **Seasonality**: They struggle if seasonality is not explicitly engineered as features.
*   **Leakage Risk**: Random splitting allows future data to leak into the training set.

**Solutions**
*   **Sliding Windows**: Convert series into supervised learning format using lags ($t-1, t-2$).
*   **Feature Engineering**: Add rolling statistics (moving mean/std) and calendar features (month, day of week).
*   **Validation**: Use **Walk-Forward Validation** designed for time series, not k-fold.
*   **Hybrid Approach**: Use decomposition to model the residuals with ML after removing trend/seasonality.

## 17. What is a Decision Tree? Mention any two advantages of using it.

**Decision Tree**
*   **Definition**: A supervised learning algorithm that splits data into branches based on feature conditions, forming a tree-like structure for decision making.

**Advantages**
1.  **Interpretability**: Easy to understand and visualize; mimics human logic.
2.  **Versatility**: Handles both numerical and categorical data with minimal preprocessing (no normalization needed).

## 18. Describe the model selection criteria AIC & BIC and their role in ARIMA modeling.

**AIC (Akaike Information Criterion)**
*   **Metric**: Balances goodness of fit (likelihood) vs. model complexity (number of parameters $k$).
*   **Goal**: Penalizes excessive parameters to prevent overfitting. Lower AIC is better.
*   **Formula**: $AIC = 2k - 2\ln(L)$

**BIC (Bayesian Information Criterion)**
*   **Metric**: Similar to AIC but applies a **stronger penalty** for complexity ($k \ln(n)$).
*   **Goal**: Prefers simpler models, especially for large datasets. Lower BIC is better.
*   **Formula**: $BIC = k \ln(n) - 2\ln(L)$

**Role in ARIMA**
*   Used to select the optimal combination of $(p, d, q)$ parameters by choosing the model with the **lowest AIC/BIC scores**.
