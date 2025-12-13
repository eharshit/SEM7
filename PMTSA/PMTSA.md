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
