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
