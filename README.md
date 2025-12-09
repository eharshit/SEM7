# Assignment 1

## Part A – Short Answer Questions

### 1. Define Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL). Give one real-world example of each.

**Answer:**
*   **Artificial Intelligence (AI):** AI is a broad branch of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence, such as reasoning, problem-solving, and perception. It is the simulation of human intelligence processes by machines.
    *   *Example:* Smart Assistants like **Alexa** or **Siri** that understand voice commands and respond.
*   **Machine Learning (ML):** ML is a subset of AI that involves training algorithms to learn from data and improve their accuracy over time without being explicitly programmed for every specific rule. It focuses on identifying patterns in data.
    *   *Example:* **Email Spam Filters** that learn to identify junk mail based on user actions and email content.
*   **Deep Learning (DL):** DL is a specialized subset of ML based on Artificial Neural Networks with multiple layers (hence "deep"). It attempts to simulate the behavior of the human brain to learn from large amounts of data.
    *   *Example:* **Self-Driving Cars** using computer vision to recognize pedestrians, traffic lights, and road signs in real-time.

### 2. Differentiate between AI Services, ML Services, and ML Frameworks in AWS.

**Answer:**
The AWS AI/ML stack is organized into three layers based on abstraction and control:

| Feature | AI Services | ML Services | ML Frameworks & Infrastructure |
| :--- | :--- | :--- | :--- |
| **Level** | Top Layer (High Abstraction) | Middle Layer (Managed Platform) | Bottom Layer (Low Abstraction) |
| **Target Audience** | Software Developers (No ML skills needed) | Data Scientists & ML Engineers | Expert Researchers & practitioners |
| **Function** | Pre-trained APIs for vision, speech, etc. | Managed platform to build, train, deploy models. | Raw frameworks and compute power |
| **Examples** | Amazon Rekognition, Polly, Lex, Translate | Amazon SageMaker | TensorFlow, PyTorch, Apache MXNet on EC2 |

### 3. What is the role of Amazon Rekognition and Amazon Polly in AI applications?

**Answer:**
*   **Amazon Rekognition:** It is an image and video analysis service. Its role is to add computer vision capabilities to applications. It can identify objects, people, text, scenes, and activities in images and videos, as well as detect inappropriate content.
*   **Amazon Polly:** It is a service that turns text into lifelike speech. Its role is to enable applications to "speak" by converting written text into realistic human voices (Text-to-Speech), allowing for the creation of speech-enabled products.

### 4. List three features of Amazon SageMaker and explain how it simplifies the ML lifecycle.

**Answer:**
**Three Features:**
1.  **SageMaker Studio:** An integrated development environment (IDE) specifically for ML.
2.  **SageMaker Autopilot:** Automatically explores data, selects algorithms, and trains the best model with minimal user input.
3.  **SageMaker Experiments:** Tracks and manages different iterations of model training to compare results.

**How it simplifies the ML Lifecycle:**
SageMaker abstracts away the heavy lifting of infrastructure management. It provides a single platform that covers the entire lifecycle—from **labeling data** (Ground Truth) and **building models** (Notebooks) to **training** (managed distributed clusters) and **deployment** (one-click endpoints). Developers don't need to manually provision servers or configure networking, which speeds up the process from idea to production.

### 5. What is the difference between TensorFlow, PyTorch, and Apache MXNet as ML frameworks?

**Answer:**
*   **TensorFlow:** Created by Google. It is highly popular for production environments and supports both mobile and large-scale deployments. It historically used static computation graphs (though eager execution was added later).
*   **PyTorch:** Created by Facebook (Meta). It is favored by researchers for its flexibility and intuitive "Pythonic" style. It uses dynamic computation graphs, making it easier to debug and experiment with.
*   **Apache MXNet:** A high-performance, scalable framework often chosen for AWS services integration. It is known for its efficiency in training deep learning models across multiple GPUs and machines.

---

## Part B – Descriptive Questions

### 6. Explain the layered structure of the AWS AI/ML stack with suitable examples.

**Answer:**
The AWS AI/ML stack is designed to cater to different levels of expertise, structured into three distinct layers:

1.  **Top Layer: AI Services (Application Services)**
    *   *Description:* These are pre-trained services available via simple APIs. They are designed for app developers who want to add intelligence without building their own models.
    *   *Examples:*
        *   **Vision:** Amazon Rekognition (Image analysis).
        *   **Speech:** Amazon Polly (Text-to-speech), Amazon Transcribe (Speech-to-text).
        *   **Language:** Amazon Lex (Chatbots), Amazon Comprehend (NLP).

2.  **Middle Layer: ML Services (Platform Services)**
    *   *Description:* This layer centers around **Amazon SageMaker**. It is a fully managed platform that removes the "heavy lifting" of machine learning. It is for data scientists who want to build custom models but don't want to manage the underlying infrastructure (servers, clusters).
    *   *Examples:* Amazon SageMaker (Build, Train, Deploy), SageMaker Studio, SageMaker Ground Truth.

3.  **Bottom Layer: ML Frameworks & Infrastructure**
    *   *Description:* This is for expert practitioners who need full control over the environment. It provides raw compute power and supported deep learning frameworks.
    *   *Examples:*
        *   **Infrastructure:** EC2 instances with GPUs (P3, P4 instances), AWS Neuron SDK.
        *   **Frameworks:** Deep Learning AMIs (Amazon Machine Images) pre-loaded with TensorFlow, PyTorch, and MXNet.

### 7. Discuss the different AWS pricing models. Compare Pay-as-you-go and Reserved Instances with examples.

**Answer:**
AWS offers flexible pricing to suit different usage patterns. The two primary models compared here are:

1.  **Pay-as-you-go (On-Demand):**
    *   *Concept:* You pay only for the compute capacity or services you actually consume, similar to an electricity bill. There are no upfront commitments.
    *   *Pros:* Maximum flexibility, ideal for short-term or spiky workloads.
    *   *Example:* If you launch an EC2 instance to train a model for 3 hours and then terminate it, you are charged only for those 3 hours.

2.  **Reserved Instances (RI) / Savings Plans:**
    *   *Concept:* You commit to a specific usage term (1 or 3 years) in exchange for a significant discount compared to On-Demand prices.
    *   *Pros:* Lower cost for steady-state, predictable workloads.
    *   *Example:* A database server that needs to be running 24/7 for the next year. By purchasing a Reserved Instance for it, you might save up to 72% compared to the hourly On-Demand rate.

**Comparison:**
*   **Flexibility:** Pay-as-you-go is high (cancel anytime); Reserved is low (committed for years).
*   **Cost:** Pay-as-you-go is more expensive per hour; Reserved offers the lowest hourly rate.
*   **Use Case:** Pay-as-you-go for testing/experiments; Reserved for core production systems.

### 8. What are IAM Users, Roles, and Policies? How do they help in securing cloud resources?

**Answer:**
**Recall: "Identity and Access Management" (IAM)**
*   **IAM User:** Represents a specific person or application that interacts with AWS. A user has permanent credentials (username/password or Access Keys).
*   **IAM Role:** An identity with specific permissions that is not associated with a specific person. It does not have permanent credentials. Instead, it is "assumed" by a user, service, or application to obtain temporary security credentials.
*   **IAM Policy:** A JSON document that explicitly defines permissions (what actions are allowed or denied). Policies are attached to Users, Groups, or Roles.

**How they secure resources:**
They act as the security guard for AWS accounts. Instead of giving everyone "Admin" access, you verify **identity** (Authentication) via Users and enforce **permissions** (Authorization) via Policies. Roles ensure that services (like EC2 accessing S3) don't need hardcoded passwords, reducing the risk of credential leakage.

### 9. Explain the principle of least privilege in IAM. Why is it considered a best practice?

**Answer:**
*   **Principle:** The principle of least privilege states that a user or service should be granted *only* the permissions necessary to perform their specific job and nothing more.
*   **Why it is best practice:**
    *   **Security:** If a user's credentials are compromised, the attacker can only damage the specific narrow scope that the user had access to, rather than the entire account.
    *   **Accident Prevention:** It prevents legitimate users from accidentally deleting or modifying critical resources they shouldn't be touching.
    *   *Example:* A developer working on an S3 storage bucket doesn't need start/stop access to EC2 servers.

### 10. Describe how AWS Regions and Availability Zones ensure fault tolerance.

**Answer:**
*   **AWS Region:** A separate geographic area (e.g., US-East-1, EU-West-1) that acts as a completely independent failure domain. Regions are isolated from one another.
*   **Availability Zones (AZs):** Within each Region, there are multiple (usually 3 or more) Availability Zones. Each AZ is essentially a physically distinct data center (or cluster of them) with its own power, cooling, and networking.

**Ensuring Fault Tolerance:**
By deploying an application across multiple AZs within a Region, you ensure resilience. If a fire or power outage strikes one AZ (Data Center A), the application continues to run in the other AZs (Data Center B and C) without interruption. This physical separation is the cornerstone of high availability in the cloud.

---

## Part C – Case Study / Application-based Questions

### 11. A company wants to build a fraud detection system using AWS AI/ML services. Which services would you recommend, and why?

**Answer:**
**Recommendation:**
1.  **Amazon Fraud Detector:** This is the primary recommendation. It is a fully managed AI service specifically built for detecting fraud (fake accounts, payment fraud) using AWS's own expertise. It requires no ML experience.
2.  **Amazon SageMaker:** If the company has unique, highly specific data requiring a custom algorithm not covered by the managed service.

**Why:**
*   **Speed to Market:** Amazon Fraud Detector uses pre-built models that learn from historical data, allowing the company to deploy a solution in days rather than months.
*   **Accuracy:** It leverages data patterns seen by AWS (consumer fraud patterns) to boost accuracy.
*   **Integration:** It can easily integrate with real-time transaction flows.

### 12. Suppose you are developing a student performance prediction system. How can Amazon SageMaker help in training and deploying the ML model?

**Answer:**
Amazon SageMaker streamlines the workflow for this predictive system:
1.  **Data Preparation (SageMaker Data Wrangler):** You can import student data (grades, attendance, demographics) and clean/transform it visually.
2.  **Training (SageMaker Training Jobs):** You select an algorithm (e.g., Linear Regression or XGBoost). SageMaker spins up the necessary compute instances, trains the model on the student data, and then shuts down the instances automatically to save cost.
3.  **Tuning (Hyperparameter Tuning):** SageMaker can automatically run multiple training runs with slightly different settings to find the most accurate model.
4.  **Deployment (SageMaker Endpoints):** Once trained, you can deploy the model as a real-time API endpoint. The school's application can send a student's profile to this endpoint, and SageMaker will return the predicted performance score instantly.

### 13. Design a small use case where Rekognition and Polly can work together in an application. Explain step by step.

**Answer:**
**Use Case: Smart Assistant for the Visually Impaired**
An application on a mobile phone helps visually impaired users understand their surroundings.

**Step-by-Step Workflow:**
1.  **Capture:** The user points their phone camera at an object (e.g., a bottle of water on a table) and taps the screen.
2.  **Analyze (Rekognition):** The app sends the captured image to **Amazon Rekognition**.
    *   Rekognition analyzes the image and returns labels: "Table," "Water Bottle," "Plastic."
3.  **Process:** The application logic selects the most confident and relevant label: "Water Bottle."
4.  **Synthesize (Polly):** The text "Water Bottle" is sent to **Amazon Polly**.
    *   Polly converts the text into a realistic audio stream (MP3).
5.  **Output:** The phone plays the audio, allowing the user to "hear" what is in front of them.

### 14. Your organization needs to give temporary access to a contractor for uploading files into an S3 bucket. How would you configure IAM roles and policies for this situation?

**Answer:**
**Configuration Steps:**
1.  **Create a Policy:** Write an IAM Policy that allows only specific actions (like `s3:PutObject`) on that specific S3 bucket resource. This strictly follows the principle of least privilege.
2.  **Create a Role:** Create an IAM Role (e.g., `ContractorUploadRole`) and attach the policy created in step 1 to it.
3.  **Trust Relationship:** Configure the Role's "Trust Policy" to specify who can assume this role (e.g., a specific external AWS account ID if the contractor has one, or via an Identity Provider).
4.  **Access:**
    *   Instead of giving the contractor long-term Access Keys (which are risky), guide them to use AWS STS (Security Token Service) to **assume** the role.
    *   The contractor receives temporary credentials (valid for a few hours) to perform the upload. Once the session expires, access is automatically revoked.

### 15. Compare traditional on-premise ML development with Cloud-based ML development (AWS SageMaker) in terms of cost, scalability, and ease of use.

**Answer:**
| Feature | Traditional On-Premise ML | Cloud-based ML (AWS SageMaker) |
| :--- | :--- | :--- |
| **Cost** | **High CapEx (Capital Expenditure).** Requires buying expensive GPUs and servers upfront, regardless of utilization. | **OpEx (Operating Expenditure).** Pay-as-you-go. You only pay for the minutes/hours the training instances are running. |
| **Scalability** | **Difficult.** Adding more power involves purchasing and installing physical hardware, taking weeks or months. | **Elastic.** Scale up or down instantly. You can switch from 1 server to 50 servers for a large job and back to 0 in minutes. |
| **Ease of Use** | **Low.** Requires managing cooling, hardware updates, drivers, and networking manually. | **High.** Managed services handle the OS, patching, and hardware management. Development features (Studio, Autopilot) are built-in. |

---

# Assignment 2: on Amazon SageMaker

## Part A – Short Answer Questions

### 1. What is Amazon SageMaker and how does it simplify the machine learning lifecycle?

**Answer:**
*   **Definition:** Amazon SageMaker is a fully managed service that provides every developer and data scientist with the ability to build, train, and deploy machine learning (ML) models quickly.
*   **Simplification:** It removes the heavy lifting from each step of the ML process:
    *   **Build:** Provides hosted Jupyter notebooks (SageMaker Notebooks) or an IDE (SageMaker Studio) with pre-configured kernels.
    *   **Train:** Manages distributed training clusters, automatically handling infrastructure provisioning and tear-down.
    *   **Deploy:** Offers one-click deployment for hosting models on auto-scaling clusters or specifically for batch processing.

### 2. Explain the role of Amazon S3 in SageMaker workflows.

**Answer:**
Amazon Simple Storage Service (S3) acts as the primary storage layer for SageMaker.
*   **Input Data:** Training datasets (images, CSVs) are stored in S3 buckets for the model to access during training.
*   **Model Artifacts:** After training, the model methodology (the actual "brain" file, e.g., `model.tar.gz`) is outputted and saved back to a specified S3 bucket.
*   **Checkpointing:** During long training jobs, SageMaker saves intermediate checkpoints to S3 to prevent data loss.

### 3. Differentiate between SageMaker Studio and SageMaker Notebooks.

**Answer:**
*   **SageMaker Notebook Instances:** These are traditional, individual EC2 instances running JupyterLab/Jupyter Notebooks. They are great for quick experiments but are tied to a specific instance type and need manual start/stop management.
*   **SageMaker Studio:** This is a fully integrated development environment (IDE) for ML. It is a unified interface that offers more than just notebooks: it includes visual tools for debugging (Debugger), tracking experiments (Experiments), managing pipelines (Pipelines), and detecting bias (Clarify). It allows for easier switching of compute resources without restarting the server.

### 4. What are the advantages of using built-in algorithms in SageMaker compared to custom algorithms?

**Answer:**
*   **Speed & Optimization:** Built-in algorithms (like XGBoost, Linear Learner) are highly optimized by AWS engineers to run efficiently on AWS infrastructure, providing up to 10x faster training than open-source implementations.
*   **Ease of Use:** You don't need to write the model code from scratch. You only need to provide the data and set hyperparameters.
*   **Out-of-the-Box:** They come pre-packaged as Docker containers, ready for training and deployment immediately.

### 5. Define hyperparameters. Why is hyperparameter tuning important?

**Answer:**
*   **Definition:** Hyperparameters are the external configuration variables that you set *before* training a model (e.g., Learning Rate, Batch Size, Max Depth of a tree). They control the training process itself.
*   **Importance of Tuning:** Choosing the wrong values can lead to a model that fails to learn (underfitting) or memorizes the data (overfitting). Tuning (finding the optimal combination of these values) acts like "fine-tuning" a radio frequency to get the clearest signal (highest accuracy) for your model.

### 6. List and explain any three preprocessing steps commonly performed before training a model.

**Answer:**
1.  **Handling Missing Values (Imputation):** Data often has gaps. Strategies include filling them with the mean/median of the column or removing the rows entirely to ensure the algorithm doesn't error out.
2.  **Encoding Categorical Variables:** ML algorithms require numbers, not text. Processes like *One-Hot Encoding* convert categories (e.g., "Red", "Blue") into binary vectors (1, 0) for the model to understand.
3.  **Feature Scaling (Normalization/Standardization):** If features vary wildly in scale (e.g., Age: 20-60 vs Income: 30000-100000), the model might be biased towards larger numbers. Scaling adjusts all features to a common range (e.g., 0 to 1).

### 7. What are the different deployment options in SageMaker?

**Answer:**
*   **Real-time Inference:** Deploys a persistent HTTPS endpoint for low-latency, real-time predictions (e.g., instant credit card fraud check).
*   **Batch Transform:** Processes large datasets offline. It spins up a cluster, processes the whole file, saves results to S3, and shuts down (e.g., scoring a million customers for marketing emails at night).
*   **Asynchronous Inference:** For requests that take a long time to process (minutes) or have large payloads. It queues requests and processes them when resources are available.
*   **Serverless Inference:** Auto-scaling endpoints that scale to zero when not in use, ideal for intermittent traffic.

### 8. Mention at least two AWS services that integrate with SageMaker for security and automation.

**Answer:**
1.  **AWS Identity and Access Management (IAM):** Controls *who* can access SageMaker resources and *what* permissions the SageMaker instances themselves have (e.g., permission to read from S3).
2.  **AWS Key Management Service (KMS):** Encrypts the data at rest in S3 and in the application storage volumes attached to SageMaker notebooks to ensure data security.

### 9. Explain the importance of model monitoring after deployment.

**Answer:**
Models are trained on historical data, but the real world changes. **Model Monitoring** is crucial to detect:
*   **Data Drift:** A change in the input data distribution (e.g., incomes rise due to inflation, but the model was trained on old salaries).
*   **Model Quality Drift:** A degradation in the model's accuracy over time.
Monitoring ensures you are alerted when the model becomes stale so you can retrain it before it impacts business decisions.

### 10. What is Boto3 and how is it used in SageMaker?

**Answer:**
*   **Definition:** Boto3 is the AWS Software Development Kit (SDK) for Python.
*   **Usage in SageMaker:** While the high-level `sagemaker` Python SDK is often used for ease, Boto3 (`boto3.client('sagemaker')`) allows for low-level, granular control over AWS services. It is used to programmatically create buckets, upload data to S3, invoke SageMaker endpoints, and manage AWS resources directly from Python scripts.

---

## Part B – Long Answer Questions

### 1. Draw and explain the high-level architecture diagram of Amazon SageMaker.

**Answer:**
*(Note: As this is a text description, imagine a diagram with three main pillars)*

**Architecture Components:**

1.  **Preparation (Notebook Instance):**
    *   **User** interacts with a **SageMaker Notebook Instance** (EC2) running Jupyter.
    *   This instance connects to **Amazon S3** to upload/download datasets.

2.  **Training (Training Cluster):**
    *   When training starts, SageMaker spins up a separate, ephemeral **Training Cluster** (EC2 instances).
    *   It pulls the **Data** from S3 and the **Algorithm Container** (Docker image) from **Amazon ECR** (Elastic Container Registry).
    *   After training, the **Model Artifacts** are pushed back to **S3**, and the cluster is terminated.

3.  **Inference (hosting Cluster):**
    *   For deployment, SageMaker reads the model from S3 and spins up an **Inference Endpoint** (hosting Instance).
    *   Client Applications send API requests (HTTPS) to this endpoint to get predictions.

### 2. Describe the workflow of an ML project in SageMaker from data collection to monitoring.

**Answer:**
The SageMaker workflow follows the standard ML lifecycle:

1.  **Data Collection & Storage:** Raw data is gathered and stored in an **Amazon S3 bucket**.
2.  **Data Preprocessing:** A data scientist uses **SageMaker Studio** or **Notebooks** to clean, explore, and visualize the data (using Pandas/Matplotlib).
3.  **Model Training:** A `Training Job` is configured. SageMaker provisions the requested compute instances, loads the data from S3, and runs the training algorithm.
4.  **Model Evaluation:** The trained model is evaluated against a validation dataset. Metrics (accuracy, RMSE) are logged to **CloudWatch**.
5.  **Deployment:** If effective, the model is deployed to an **Endpoint** for real-time predictions.
6.  **Monitoring:** **SageMaker Model Monitor** continuously watches the endpoint's inputs/outputs. If data drifts beyond a baseline, it triggers an alert (via CloudWatch) to prompt retraining.

### 3. Explain with examples how data preprocessing is performed in SageMaker notebooks.

**Answer:**
Preprocessing is typically done using standard Python libraries like Pandas and Scikit-Learn within the notebook before training.

**Example Scenario:** preparing a customer dataset.

*   **Step 1: Load Data**
    ```python
    import pandas as pd
    df = pd.read_csv('s3://my-bucket/data.csv')
    ```
*   **Step 2: Handle Missing Values**
    ```python
    # Fill missing 'Age' with the mean age
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    ```
*   **Step 3: Encoding**
    ```python
    # Convert 'Gender' (Male/Female) to 0/1
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    ```
*   **Step 4: Splitting**
    ```python
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=0.2)
    ```
*   **Step 5: Upload for Training**
    After processing, the clean dataframes are saved as CSVs and uploaded back to S3 for the SageMaker Training Job to use.

### 4. Compare real-time endpoints, batch transform, and asynchronous inference in model deployment.

**Answer:**

| Feature | Real-time Inference | Batch Transform | Asynchronous Inference |
| :--- | :--- | :--- | :--- |
| **Use Case** | Instant predictions needed immediately (e.g., fraud check during swipe). | Processing huge datasets offline (e.g., nightly risk scoring). | Large payloads or long processing times (e.g., processing a 1GB image). |
| **Latency** | Milliseconds to Seconds. | Hours (depends on dataset size). | Seconds to Minutes (queued). |
| **Architecture** | Persistent server (always on). | Ephemeral cluster (starts, runs, stops). | Persistent server with a queue. |
| **Cost** | 24/7 hourly cost (unless serverless). | Pay only for the duration of the job. | Hourly cost for the instance. |

### 5. Discuss advantages and challenges of using SageMaker in enterprise-scale ML projects.

**Answer:**

**Advantages:**
*   **Scalability:** Can train on hundreds of GPUs simultaneously without manual cluster setup.
*   **Security:** Integrated with AWS IAM, VPC, and KMS, meeting strict enterprise compliance standards (HIPAA, PCI).
*   **MLOps:** Features like SageMaker Pipelines allow for automated, reproducible workflows (CI/CD for ML).

**Challenges:**
*   **Cost Management:** It is easy to accidentally leave expensive instances (like P3/P4 GPU instances) running, leading to "cloud bill shock."
*   **Complexity:** The sheer number of features (Studio, Wrangler, Clarify, JumpStart) can be overwhelming for beginners.
*   **Vendor Lock-in:** Heavy reliance on specific SageMaker APIs can make it difficult to migrate models to a different cloud provider later.

---

## Part C – Lab 3 and 4

**(Note: This section outlines the code steps required to complete the lab tasks.)**

### 1. Data Ingestion
*   **Task:** Upload dataset to S3.
*   **Implementation:**
    ```python
    import boto3
    s3 = boto3.client('s3')
    bucket_name = 'my-sagemaker-lab-bucket'
    s3.upload_file('local_dataset.csv', bucket_name, 'raw/data.csv')
    ```

### 2. Preprocessing in Notebooks
*   **Task:** Load, encode, split.
*   **Implementation:**
    ```python
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(f's3://{bucket_name}/raw/data.csv')
    # Preprocessing
    df.dropna(inplace=True)
    df = pd.get_dummies(df) # One-hot encoding

    # Split: Train/Val
    train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save and upload back to S3 for training
    train_data.to_csv('train.csv', index=False, header=False)
    val_data.to_csv('validation.csv', index=False, header=False)
    # (Use upload code from Step 1)
    ```

### 3. Model Training (XGBoost)
*   **Task:** Train model using built-in XGBoost.
*   **Implementation:**
    ```python
    import sagemaker
    from sagemaker.xgboost.estimator import XGBoost

    role = sagemaker.get_execution_role()
    xgb_estimator = XGBoost(
        entry_point='script.py', # or use image_uri for built-in
        role=role,
        instance_count=1,
        instance_type='ml.m5.large',
        framework_version='1.5-1',
        hyperparameters={'max_depth': 5, 'eta': 0.2, 'objective': 'binary:logistic'}
    )
    
    xgb_estimator.fit({'train': f's3://{bucket_name}/train', 'validation': f's3://{bucket_name}/validation'})
    ```

### 4. Hyperparameter Tuning
*   **Task:** Automatic tuning.
*   **Implementation:**
    ```python
    from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner

    hyperparameter_ranges = {
        'max_depth': IntegerParameter(3, 10),
        'eta': ContinuousParameter(0.1, 0.5)
    }

    tuner = HyperparameterTuner(
        xgb_estimator,
        objective_metric_name='validation:rmse',
        hyperparameter_ranges=hyperparameter_ranges,
        max_jobs=10,
        max_parallel_jobs=2
    )

    tuner.fit({'train': ..., 'validation': ...})
    ```

### 5. Model Evaluation
*   **Task:** Evaluate performance.
*   **Implementation:** Download the test set predictions and use `sklearn.metrics`.
    ```python
    from sklearn.metrics import accuracy_score, precision_score
    # Assuming predictions are obtained
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")
    ```

### 6. Deployment
*   **Task:** Real-time endpoint.
*   **Implementation:**
    ```python
    predictor = xgb_estimator.deploy(initial_instance_count=1, instance_type='ml.m5.large')
    response = predictor.predict(test_data_sample)
    ```

### 7. Monitoring
*   **Task:** Enable Data Capture.
*   **Implementation:**
    ```python
    from sagemaker.model_monitor import DataCaptureConfig
    
    capture_config = DataCaptureConfig(
        enable_capture=True, 
        sampling_percentage=100, 
        destination_s3_uri=f's3://{bucket_name}/monitoring'
    )
    
    predictor.update_data_capture_config(capture_config)
    ```

---

## Part D – Case Study

**Scenario: Financial Company Loan Default Prediction**

### 1-3. Build, Train, Tune (Implementation Summary)
The workflow for the bank dataset follows the patterns in Part C.
*   **Data:** The "Bank Loan" CSV (Income, Credit Score, Default History) is cleaned. Missing incomes are imputed.
*   **Models:** We use the `binary:logistic` objective with XGBoost since this is a Yes/No classification problem.
*   **Tuning:** We tune `scale_pos_weight` to handle class imbalance (since defaults are rare compared to non-defaults).

### 4. Deploy and Test
*   **Deployment:** The best model from the Tuning job is deployed to an `ml.t2.medium` instance to save costs during dev.
*   **Test:** A dummy customer profile is sent: `{Income: 50000, CreditScore: 600, ...}`. The model returns score `0.85` (High portability of default).

### 5. Monitor for Drift
Over 6 months, economic conditions change. The `Income` distribution of applicants lowers.
*   **Action:** SageMaker Model Monitor detects the statistical difference in `Income` (baseline vs. current).
*   **Alert:** A CloudWatch alarm triggers an email to the Data Science team to retrain the model on newer data.

### Additional Questions

#### 4. How does SageMaker compare with Google Vertex AI or Azure ML Studio?
*   **SageMaker (AWS):** Best for users already deep in the AWS ecosystem. extremely granular control (coders love it). Massive marketplace of algorithms.
*   **Vertex AI (Google):** Excellent integration with MLOps and TensorFlow/TFX. Often felt to be more "opinionated" or streamlined for Auto-ML.
*   **Azure ML Studio (Microsoft):** Strongest visual interface ("Designer" drag-and-drop). Best for enterprise integration with Office/Windows environments.

#### 5. Explain how serverless architecture in AWS Lambda can integrate with SageMaker.
Lambda acts as the "glue" or the "trigger."
*   **Scenario:** A user uploads a photo to S3.
*   **Integration:**
    1.  **S3 Event:** The upload triggers an **AWS Lambda function**.
    2.  **Logic:** The Lambda function has a script that calls the **SageMaker Endpoint** (using Boto3 `invoke_endpoint`).
    3.  **Result:** SageMaker returns the prediction to Lambda, which then saves the result to a database (DynamoDB).
*   **Benefit:** You don't need a permanently running web server to handle the application logic; Lambda manages the execution only when an event occurs.

---

# Assignment 3: AWS Language AI Services

## Part A – Conceptual Questions

### 1. Explain the core architecture and working mechanism of Amazon Comprehend.

**Answer:**
**Working Mechanism:**
Amazon Comprehend is a Natural Language Processing (NLP) service that uses machine learning to find insights and relationships in text. It works on a pre-trained model basis (and supports custom models) to process unstructured text.
*   **Input:** Raw text documents (emails, support tickets, reviews) ingested from S3.
*   **Processing:** The text passes through internal deep learning models trained for specific tasks (Entity Recognition, Sentiment Analysis, Key Phrase Extraction, Topic Modeling).
*   **Output:** Returns a JSON structure containing the detected entities, sentiment scores (Positive, Negative, Neutral, Mixed), language codes, and key phrases, along with confidence scores for each.

### 2. Use Case: Propose how Amazon Comprehend could be used by an e-commerce company to automatically extract and analyze customer feedback from product reviews.

**Answer:**
**Scenario:** An online electronic store wants to know why a specific headphone model is getting returning customers.
**Implementation:**
1.  **Ingestion:** Scrape customer reviews and store them in Amazon S3.
2.  **Analysis:** Use Amazon Comprehend's **Sentiment Analysis** and **Entity Extraction** APIs.
3.  **Insights:**
    *   **Sentiment:** Filter reviews to isolate "Negative" sentiment.
    *   **Topic Modeling:** Comprehend identifies that the phrase "Battery Life" often appears in negative reviews, while "Sound Quality" appears in positive ones.
4.  **Action:** The company realizes the battery is the pain point and forwards this data to the manufacturing team.

### 3. Describe the main components of Amazon Lex (Intents, Utterances, Slots, Prompts).

**Answer:**
*   **Intents:** A particular goal that the user wants to achieve (e.g., "BookHotel", "OrderPizza").
*   **Utterances:** Phrases that the user speaks or types to trigger an intent (e.g., "I want to book a room", "Can I get a pizza?", "Reserve a hotel").
*   **Slots:** Data required to fulfill the intent. These are parameters the bot needs to collect (e.g., `{City}`, `{CheckInDate}`, `{PizzaSize}`, `{Toppings}`).
*   **Prompts:** Questions the bot asks the user to fill the slots (e.g., "What city are you staying in?", "What size pizza would you like?").

### 4. What role does Automatic Speech Recognition (ASR) and Natural Language Understanding (NLU) play in Lex?

**Answer:**
Amazon Lex allows you to build conversational interfaces using the same deep learning engine as Alexa.
*   **ASR (Automatic Speech Recognition):** This is the **Input** layer. It converts the spoken audio waves from the user into text. If the user speaks "Book a flight," ASR converts that sound into the string "Book a flight".
*   **NLU (Natural Language Understanding):** This is the **Logic** layer. It analyzes the text to determine the user's *Intent* (what they want) and extracts *Slots* (specific data). It understands that "Book a flight" maps to the `BookFlight` intent.

### 5. Use Case: Design a banking chatbot using Amazon Lex that helps customers check their account balance and transfer funds. Identify intents, utterences, slots.

**Answer:**
**Bot Name:** BankBot

**Intent 1: CheckBalance**
*   **Utterances:** "What is my balance?", "Check checking account", "How much money do I have?"
*   **Slots:** `{AccountType}` (Checking, Savings, Credit).
*   **Prompt:** "Which account would you like to check? Checking or Savings?"

**Intent 2: TransferFunds**
*   **Utterances:** "Transfer money", "Send cash to Mom", "Make a transfer".
*   **Slots:**
    *   `{SourceAccount}` (e.g., Checking)
    *   `{DestinationAccount}` (e.g., Savings)
    *   `{Amount}` (e.g., $500)
*   **Prompt:** "Which account are you transferring from?", "Where should the money go?", "How much do you want to transfer?"

### 6. Compare the different voice generation technologies: Standard TTS, Neural TTS (NTTS), and Generative Engine.

**Answer:**
| Feature | Standard TTS | Neural TTS (NTTS) | Generative Engine |
| :--- | :--- | :--- | :--- |
| **Technology** | Concatenative synthesis (gluing recorded sounds together). | Deep Learning models (LSTM/RNN) that generate speech features. | Large Language Model (LLM) based storage of voice characteristics. |
| **Quality** | Robotic, unnatural phrasing. Good for basic alerts. | Highly natural, human-like intonation and breathing pauses. | Extremely realistic, adaptive to context, and emotionally expressive. |
| **Adaptability** | Limited. | Supports styles (Newscaster, Conversational). | Can adapt tone and style dynamically based on the text context. |

### 7. Use Case: Imagine you are building an e-learning platform for visually impaired students. Explain how Amazon Polly can make the platform more inclusive and effective.

**Answer:**
**Implementation:**
1.  **Text-to-Speech:** Integrate Polly to convert all written course material (PDFs, quizzes, articles) into audio.
2.  **Language Support:** Use Polly's multilingual capabilities to offer content in the student's native language.
3.  **Neural Voices:** Use **Neural TTS** for long-form reading (Chapter descriptions) to reduce listener fatigue, as it sounds more natural than standard robotic voices.
4.  **Speech Marks:** Use Polly's "Speech Marks" metadata to highlight text on the screen as it is being read (if the student has partial vision), aiding in following along.

### 8. Outline the key steps involved in Amazon Transcribe’s processing pipeline.

**Answer:**
1.  **Input:** Audio file (MP3, WAV, FLAC) acts as input from an S3 bucket or a real-time stream.
2.  **Acoustic Modeling:** The service analyzes the audio waveform to identify phonemes (basic units of sound).
3.  **Language Modeling:** It breaks phonemes into words and sequences based on probability (predicting the next word).
4.  **Punctuation & Formatting:** It automatically adds capitalization and punctuation to make the text readable.
5.  **Output:** A JSON transcript is generated containing the text, confidence scores, and timestamps for every word.

### 9. How does Speaker Diarization and Custom Vocabulary improve transcription accuracy?

**Answer:**
*   **Speaker Diarization:** This feature distinguishes *who* is speaking. It labels the text as "Speaker 0: Hello", "Speaker 1: Hi there". This is crucial for meetings or interviews to understand the context of the conversation.
*   **Custom Vocabulary:** Standard ASR struggles with domain-specific terms (e.g., medical drug names like "Hydroxychloroquine" or brand names like "SupraCL"). Custom Vocabulary allows you to upload a list of these specific words and their specific pronunciations, dramatically improving accuracy for technical or industry-specific audio.

### 10. Use Case: Describe how a media company can use Amazon Transcribe to automatically generate subtitles for live news broadcasts.

**Answer:**
1.  **Streaming:** The live broadcast audio is fed into **Amazon Transcribe Streaming**.
2.  **Processing:** Transcribe converts the speech to text in real-time with low latency.
3.  **Formatting:** The JSON output (with timestamps) is converted into a subtitle format (like WebVTT or SRT).
4.  **Display:** The subtitles are overlaid on the video stream for viewers.
5.  **Customization:** The company uses a "Custom Vocabulary" to ensure names of politicians, cities, and current events are spelled correctly.

---

## Part B – Analytical & Application Questions

### 11. Mini Project: Education - Lecture Video Transcription & Summarization

**Project Title:** Automating Classroom Accessibility and Study Aids

**Problem Statement:**
Students often struggle to take detailed notes during fast-paced lectures. Additionally, searching for specific concepts inside a 1-hour video is difficult.

**AWS Services Used:**
1.  **Amazon S3:** To store the recorded lecture videos.
2.  **Amazon Transcribe:** To convert the video audio into text.
3.  **Amazon Comprehend:** To extract key phrases and topics for the summary.
4.  **Amazon Polly:** To generate an audio summary for revision on the go.

**Data Flow & Outputs:**
1.  **Upload:** Professor uploads `lecture_week1.mp4` to S3.
2.  **Trigger:** An S3 event triggers a Lambda function.
3.  **Transcribe:** Lambda calls **Amazon Transcribe** to generate a full text transcript of the lecture.
    *   *Output:* A searchable text file of the entire class.
4.  **Analyze:** The transcript is sent to **Amazon Comprehend**.
    *   *Action:* Comprehend extracts "Key Phrases" (e.g., "Quantum Entanglement", "Schrodinger's Equation").
5.  **Summarize:** A script compiles these key points into a bulleted summary.
6.  **Narrate:** **Amazon Polly** reads this summary, creating a 2-minute "Recap Audio" file.

**Challenges & Ethical Considerations:**
*   **Privacy:** Ensuring student questions/voices recorded in the lecture are not analyzed without consent.
*   **Accuracy:** Scientific terms might differ from standard speech; Custom Vocabularies would be needed.

---

## Part C – Research & Real-World Analysis

### 12. Case Study: BMW Group using AWS Language AI Services

**Overview:**
The BMW Group, a world-leading premium manufacturer of automobiles, operates globally with data coming in from over 100 countries.

**AWS Service Used:**
**Amazon Translate** and **Amazon Transcribe**.

**Impact on Operations:**
*   **The Challenge:** BMW processes huge amounts of technical data from field engineers and workshops worldwide. Much of this data is in local languages (Japanese, Russian, Chinese) but needs to be analyzed centrally in German or English.
*   **The Solution:** BMW integrated **Amazon Translate** into their internal data platform.
*   **Result:**
    *   **Efficiency:** They automatically translate millions of documents and support tickets in real-time.
    *   **Speed:** Reduced the time to analyze international vehicle data from days to seconds.
    *   **Cost:** Eliminated the massive cost of manual human translation for routine technical data.
    *   This allows BMW's central engineering teams to spot global trends (e.g., a specific part failing in humid climates) instantly, regardless of the language the report was written in.

---

# Assignment 4: Amazon Rekognition

## Part A – Conceptual Questions

### 1. Explain the architecture of AWS Rekognition.

**Answer:**
Amazon Rekognition is a fully managed, serverless deep learning image and video analysis service.
*   **Input Layer:** It accepts direct image uploads or integrates with **Amazon S3** (storage) and **Kinesis Video Streams** (real-time video).
*   **Deep Learning Layer:** It runs on pre-trained Convolutional Neural Networks (CNNs) managed by AWS. These models are constantly trained on billions of images.
*   **API Interface:** Users interact via simple APIs (e.g., `DetectLabels`, `DetectFaces`) without managing specific servers or models.
*   **Output Layer:** Returns probabilistic predictions in JSON format, including bounding boxes, labels, and confidence scores.

### 2. List Rekognition’s key features.

**Answer:**
1.  **Object and Scene Detection:** Identifies thousands of objects (e.g., bicycle, phone) and scenes (e.g., parking lot, beach).
2.  **Facial Analysis:** Detects faces and attributes like emotion, age range, eyes open, and glasses.
3.  **Face Comparison (Identity Verification):** Compares a face in one image with a face in another to verify identity.
4.  **Text in Image (OCR):** Detects and extracts text from images, helpful for reading license plates or street signs.
5.  **Content Moderation:** Automatically detects unsafe, explicit, or suggestive content in images and videos.

### 3. How does Rekognition integrate with S3 and Lambda?

**Answer:**
This is the standard event-driven architecture for image processing:
1.  **Trigger:** An image is successfully uploaded to an **Amazon S3** bucket.
2.  **Event Notification:** S3 allows you to configure an event notification that triggers an **AWS Lambda** function upon upload (`s3:ObjectCreated`).
3.  **Processing:** The Lambda function executes code (using Boto3) to call **Amazon Rekognition** APIs, passing the S3 bucket name and file key.
4.  **Result:** Rekognition processes the image and returns the JSON result to Lambda, which can then save metadata to DynamoDB or send an alert via SNS.

### 4. Explain Rekognition Custom Labels.

**Answer:**
Standard Rekognition detects generic objects (e.g., "Car", "Flower"). **Custom Labels** allows you to extend this capability to specific business needs.
*   **Function:** You can train the model to identify specific objects unique to your industry using a small dataset (as few as 10 images).
*   **Example:** Warning lights on a specific machine dashboard, your company's specific logo, or distinguishing between a "Rose" and a "Tulip" instead of just "Flower".

### 5. Discuss its advantages and limitations.

**Answer:**
*   **Advantages:**
    *   **Serverless:** No infrastructure to manage or patch.
    *   **Scalability:** Can process millions of images automatically.
    *   **Integration:** Seamlessly works with other AWS services (S3, Lambda).
*   **Limitations:**
    *   **Customization Limits:** You cannot tweak the underlying neural network layers of the base models.
    *   **Privacy Concerns:** Facial recognition requires strict ethical adherence and data privacy controls.
    *   **Connectivity:** Requires internet access (unless using specific edge deployments), which might add latency compared to local processing.

---

## Part B – Case Studies

### Case Study 1: Smart Campus Security System
**Scenario:** A university wants to upgrade its security system using AI-powered facial recognition to monitor entry points, detect unauthorized visitors, and ensure student safety.

**1. Describe how AWS Rekognition can be used to design this smart security system.**
*   Rekognition can automate entry management. By using **Face Comparison**, the system can match faces at the turnstile against a database of registered student IDs stored in a "Rekognition Collection".

**2. Explain the role of S3, Lambda, and DynamoDB in automating the image processing workflow.**
*   **S3:** Stores the reference images of students.
*   **Lambda:** Orchestrates the flow. When a camera captures a face, Lambda sends it to Rekognition to compare with the S3/Collection data.
*   **DynamoDB:** Stores the access logs (Time, Student ID, Confidence Score) and metadata.

**3. What Rekognition APIs would you use for face detection and comparison?**
*   `IndexFaces`: To add students to the authorized collection.
*   `SearchFacesByImage`: To compare a captured live face against the authorized collection to find a match.

**4. Discuss how you would ensure data privacy and regulatory compliance (e.g., GDPR, student consent).**
*   **Consent:** Explicit opt-in from students is required.
*   **Encryption:** Use **KMS** to encrypt image data in S3 and DynamoDB.
*   **Retention:** Set up S3 Lifecycle policies to auto-delete images after their retention period expires.

**5. Suggest one improvement using Custom Labels to detect suspicious objects (like bags or helmets).**
*   Train a **Custom Labels** model specifically on dataset images of "Unattended Bags" or "Motorcycle Helmets". This allows the security cameras to flag these specific security risks which generic models might miss.

### Case Study 2: Retail Analytics & Customer Emotion Tracking
**Scenario:** A retail chain wants to understand customer behavior inside stores. Cameras capture real-time footage, and the company wants to analyze customer count, gender distribution, and emotions at checkout counters.

**1. How can AWS Rekognition’s facial analysis and emotion detection capabilities help in this scenario?**
*   It can analyze video feeds to determine the demographic mix (Age, Gender) of shoppers and their sentiment (Happy, Confused, Sad) at specific product aisles or checkout lines.

**2. Design a data flow using S3 (video storage), Lambda (trigger), and CloudWatch (monitoring).**
*   **S3:** Ingest video chunks.
*   **Lambda:** Triggered to call Rekognition Video analysis.
*   **CloudWatch:** Lambda sends the aggregated metrics (e.g., "CustomerCount: 50", "AvgEmotion: Happy") to CloudWatch Metrics for dashboarding.

**3. Which APIs will you use to detect emotions and track customers in video streams?**
*   `StartFaceDetection`: To track faces in stored videos.
*   `CreateStreamProcessor` (with Kinesis): For real-time face and emotion analysis.

**4. How can the data be visualized in a business intelligence dashboard?**
*   The metadata (Age, Emotion) stored in a database (like Redshift or DynamoDB) can be connected to **Amazon QuickSight** to create pie charts of Gender distribution or heatmaps of customer sentiment.

**5. What ethical considerations should be addressed when analyzing customer faces?**
*   **Anonymity:** Data should be aggregated. Avoid storing Personally Identifiable Information (PII) linked to the facial analysis unless necessary.
*   **Notice:** Stores must display clear signage informing customers that video analytics are in use.

### Case Study 3: Healthcare Patient Monitoring System
**Scenario:** A hospital wants to implement a system that automatically detects patient activity in hospital rooms (e.g., patient lying, sitting, or falling) and alerts nurses in emergencies.

**1. Explain how object and activity detection in AWS Rekognition Video can automate this process.**
*   Rekognition Video can detect activities and poses. By analyzing the geometry of the person, it can distinguish between "Sitting", "Standing", and "Lying Down" (potential fall).

**2. Outline how SNS and Lambda can trigger real-time alerts to medical staff.**
*   If Rekognition detects a "Fall" confidence score > 90%, Lambda publishes a message to an **Amazon SNS** topic. SNS then sends an SMS or Pager notification to the nursing station instantly.

**3. Which Rekognition features (e.g., Custom Labels) can help recognize domain-specific actions like “patient fall”?**
*   **Custom Labels:** While generic models detect people, Custom Labels can be trained specifically on images of "Patient Falling" vs "Patient Sleeping" to reduce false positives.

**4. How can you maintain HIPAA compliance when handling video footage?**
*   **Encryption:** Enable Server-Side Encryption (SSE) on S3 buckets.
*   **Access Control:** Strict IAM Policies ensuring only authorized medical applications (not individual devs) can access the raw footage.
*   **BAA:** Ensure the AWS account has a Business Associate Agreement (BAA) active.

**5. Suggest how the hospital could analyze data trends to improve patient care.**
*   Analyze long-term data to find patterns: e.g., "Falls happen most often between 2 AM and 4 AM". This allows the hospital to increase staffing during those specific high-risk windows.

### Case Study 4: Media and Entertainment – Automated Video Tagging
**Scenario:** A media production company wants to automatically tag scenes in videos for faster searching and editing — e.g., identifying “beach scenes,” “sports activities,” and “specific celebrities.”

**1. Which Rekognition APIs and features would you use for automatic tagging and celebrity identification?**
*   `StartLabelDetection`: For tagging scenes (Beach, Car chase).
*   `StartCelebrityRecognition`: To identify famous actors in the footage.

**2. Create a workflow combining S3 (storage), Rekognition Video (analysis), and DynamoDB (metadata storage).**
*   Upload raw footage to **S3** -> Trigger **Lambda** to start Rekognition Job -> Rekognition writes results to an SNS topic -> Another Lambda parses results and writes tags with timestamps to **DynamoDB**.

**3. How can Custom Labels improve tagging for specific genres (e.g., sports, nature)?**
*   Generic models might just see "Sport". Custom Labels can be trained to identify specific moves like "Touchdown" (Football) or "Slam Dunk" (Basketball) for automated highlight generation.

**4. Discuss how JSON output from Rekognition can be integrated into an editing tool or media database.**
*   The JSON contains timestamps for every label. A script can convert this into an XML or EDL (Edit Decision List) file importable by Adobe Premiere, placing markers on the timeline where specific celebrities appear.

**5. What challenges might arise when processing large video datasets, and how can AWS handle scalability?**
*   **Challenge:** Processing hours of 4K video is computationally heavy.
*   **Scalability:** Rekognition is fully managed. You don't need to provision servers; AWS automatically allocates resources to process jobs in parallel, handling widely varying loads without bottlenecks.

### Case Study 5: Financial Services – KYC (Know Your Customer) Verification
**Scenario:** A bank needs to verify customer identity during remote onboarding by comparing the uploaded photo with the ID card image provided by the user.

**1. Explain how AWS Rekognition’s compare_faces() API can be used for KYC automation.**
*   The API takes two images: the Selfie captured live and the ID card photo. It returns a **Similarity Score**. If the score is above a threshold (e.g., 95%), the identity is verified.

**2. Describe how Lambda functions and DynamoDB can store and verify identity results securely.**
*   Lambda receives the API response. It stores the *result* (Verified/Failed) and the *confidence score* in DynamoDB. To protect privacy, it should *not* store the raw biometric vectors, just the verification status and a reference to the secure S3 image location.

**3. How can detect_text() be used to extract ID details (like name, DOB, and ID number)?**
*   `DetectText` performs OCR on the ID card image to extract lines of text. Regex patterns can then parse out the Name, Date of Birth, and ID Number to auto-fill the form for the user.

**4. Discuss how Rekognition ensures data security through encryption and IAM policies.**
*   **Transit:** All API calls are secured via SSL/HTTPS.
*   **At Rest:** Rekognition does not persist images permanently for processing; it processes and discards (unless using Collections). Stored metadata is encrypted via AWS KMS.

**5. Identify potential limitations of using Rekognition for KYC verification (e.g., image quality, lighting, privacy).**
*   **Image Quality:** Blurry ID photos or glare can cause false negatives.
*   **Lighting:** Poor lighting on the selfie can reduce confidence scores.
*   **Bias:** Facial recognition algorithms must be constantly monitored to ensure they perform equally well across all demographics and skin tones.

---

# Assignment 5: Serverless Computing for AI

## Part A – Conceptual Questions

### 1. Explain serverless computing in your own words. Describe how it works and list any three advantages.

**Answer:**
**Definition:** Serverless computing is a cloud execution model where the cloud provider (AWS) runs the server and dynamically manages the allocation of machine resources. It doesn't mean "no servers," but rather that the developer doesn't have to provision, scale, or maintain them.
**How it works:** Developers write code (functions) and define "triggers" (like an HTTP request or an S3 upload). The code runs only when triggered.
**Advantages:**
1.  **No Server Management:** No OS patching or administrative overhead.
2.  **Automatic Scaling:** It scales up from 1 request to 1,000 automatically.
3.  **Cost Efficiency:** You pay only for the compute time you consume (milliseconds), not for idle time.

### 2. Draw or describe a simple AI workflow using AWS services (Lambda, API Gateway, Step Functions, S3). Explain how data flows through each stage.

**Answer:**
**Workflow Description:**
1.  **Input:** User sends a request (e.g., "Analyze this text") to **API Gateway**.
2.  **Routing:** API Gateway triggers a **Step Function** (State Machine) to orchestrate the workflow.
3.  **Process A:** Step 1 in the workflow triggers a **Lambda** function to clean/preprocess the data.
4.  **Process B (AI):** Step 2 calls **Amazon Comprehend** or **SageMaker** to perform inference.
5.  **Storage:** Step 3 uses a **Lambda** function to save the final result into **Amazon S3**.
6.  **Response:** The system returns the result ID to the user.

### 3. What is the role of AWS Lambda in an AI workflow? Give one real-world example of how Lambda is used for AI tasks.

**Answer:**
**Role:**
Amazon Lambda acts as the "glue" or the "event handler." It executes light-weight logic to connect services, transform data, or trigger inference endpoints without running a permanent server.
**Real-world Example:**
**Chatbot Fulfillment:** When a user asks a chatbot "What's the weather?", **Amazon Lex** triggers a **Lambda** function. This function calls a third-party Weather API, formats the response ("It is 75°F in New York"), and returns it to Lex to speak back to the user.

### 4. Describe how AWS CloudWatch helps in monitoring serverless AI workflows. Explain logs, metrics, and alarms with examples.

**Answer:**
*   **Logs (CloudWatch Logs):** Stores the text output from standard output/error.
    *   *Example:* If your Lambda crashes, you check the logs to see the Python traceback error.
*   **Metrics (CloudWatch Metrics):** Numerical data points tracking performance.
    *   *Example:* Checking the "Duration" metric to see if your AI inference is taking 200ms or 2 seconds.
*   **Alarms (CloudWatch Alarms):** Rules that trigger notifications based on metrics.
    *   *Example:* Setting an alarm to send an email if "ThrottledRequests" > 5 (meaning you are hitting your concurrency limit).

### 5. List and explain any three security practices used to secure AI APIs in AWS (e.g., IAM, KMS, API throttling, CloudTrail).

**Answer:**
1.  **IAM (Least Privilege):** Ensure your Lambda functions have strictly scoped IAM Roles. A function that analyzes images should have permission only for `rekognition:DetectLabels` and `s3:GetObject`, not `s3:DeleteObject`.
2.  **API Throttling (API Gateway):** Set usage plans and throttling limits on your API Gateway. This prevents attackers from flooding your expensive AI APIs with millions of junk requests (DDoS protection and Cost control).
3.  **AWS KMS (Encryption):** Use Key Management Service to manage keys for encrypting sensitive data passed through the workflow (in S3 or environment variables), ensuring data is unreadable if intercepted.
