# Assignment 1: AI/ML on Cloud

## Part A – Short Answer Questions

### 1. Define Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL). Give one real-world example of each.
*   **Artificial Intelligence (AI):** AI is the broad concept of creating smart machines that can think and solve problems like humans. It's about making computers "smart".
    *   *Example:* Virtual Assistants like Siri or Alexa that understand what you say.
*   **Machine Learning (ML):** ML is a subset of AI where computers learn from data instead of following strict rules. We don't program them with every answer; we teach them how to find the answer.
    *   *Example:* Netflix recommending movies you might like based on what you watched before.
*   **Deep Learning (DL):** DL is a specialized type of ML inspired by the human brain (using "neural networks"). It is used for very complex tasks like understanding images or speech.
    *   *Example:* Unlocking your smartphone with your face (FaceID).

### 2. Differentiate between AI Services, ML Services, and ML Frameworks in AWS.
*   **AI Services (Top Layer):** These are ready-to-use tools. You don't need to know any coding or math. You just send data (like a photo) and get an answer (like "This is a cat").
*   **ML Services (Middle Layer):** These are tools for developers to build their own models. It makes the process easier by handling the messy infrastructure work. (e.g., Amazon SageMaker).
*   **ML Frameworks (Bottom Layer):** These are the raw building blocks for experts. It's for people who want to build a model from scratch and control every detail. (e.g., TensorFlow, PyTorch).

### 3. What is the role of Amazon Rekognition and Amazon Polly in AI applications?
*   **Amazon Rekognition:** It gives applications "eyes". It allows an app to analyze images and video to identify objects, people, text, and scenes.
*   **Amazon Polly:** It gives applications a "voice". It converts written text into lifelike spoken audio.

### 4. List three features of Amazon SageMaker and explain how it simplifies the ML lifecycle.
1.  **SageMaker Studio:** A single "dashboard" where you can do all your work (building, training, deploying).
2.  **Autopilot:** A feature that automatically finds the best AI model for your data without you doing the hard work.
3.  **Model Monitor:** A security guard that watches your model after it's live to make sure it keeps working correctly.
*   **Simplification:** SageMaker handles all the servers and technical setup, so you can focus entirely on your code and data.

### 5. What is the difference between TensorFlow, PyTorch, and Apache MXNet as ML frameworks?
*   **TensorFlow:** Created by Google. It is very popular for large-scale production (big companies) and running AI on mobile phones.
*   **PyTorch:** Created by Meta (Facebook). It is loved by researchers and students because it is flexible and the code looks like normal Python.
*   **Apache MXNet:** Chosen by AWS. It is known for being very fast and efficient, often used on smaller devices.

---

## Part B – Descriptive Questions

### 6. Explain the layered structure of the AWS AI/ML stack with suitable examples.
Think of the AWS AI Stack like a 3-layer cake:
1.  **Top Layer (AI Services):** For app developers. No ML skills needed. "Instant AI".
    *   *Examples:* Amazon Rekognition (Vision), Amazon Polly (Speech).
2.  **Middle Layer (ML Services):** For data scientists. A managed platform that makes building models easier.
    *   *Example:* **Amazon SageMaker**. It manages the servers for you.
3.  **Bottom Layer (Frameworks & Hardware):** For expert practitioners. Full control over the hardware and math.
    *   *Examples:* **Deep Learning AMIs**, EC2 instances with powerful GPUs (like P3 instances), using TensorFlow or PyTorch directly.

### 7. Discuss the different AWS pricing models. Compare Pay-as-you-go and Reserved Instances with examples.
*   **Pay-as-you-go (On-Demand):**
    *   *Concept:* You pay only for what you use, by the second or hour. It's like a taxi meter—no commitment.
    *   *Example:* Running a server for 2 hours to test a script. You only pay for those 2 hours.
*   **Reserved Instances (Savings Plans):**
    *   *Concept:* You commit to using a server for a long time (like 1 year) in exchange for a big discount (up to 72% cheaper). It's like a yearly gym membership.
    *   *Example:* A company running a database 24/7. Reserving it for a year saves a lot of money compared to paying the hourly rate.

### 8. What are IAM Users, Roles, and Policies? How do they help in securing cloud resources?
*   **IAM User:** Represents a specific person or service (like "Developer1") with a permanent password/key.
*   **IAM Role:** A "hat" or badge that gives temporary permission. You put it on to do a job, and then take it off. It doesn't have a permanent password.
*   **IAM Policy:** The "Rulebook". It is a document that lists exactly what actions are Allowed or Denied.
*   **Security:** These tools ensure that only the right people can touch your data. It stops strangers from accessing your files.

### 9. Explain the principle of least privilege in IAM. Why is it considered a best practice?
**Principle of Least Privilege:** Giving someone *only* the permissions they strictly need to do their job, and nothing more.
*   *Example:* If an intern needs to upload a file to one specific folder, you give them permission for *only* that folder, not the whole database.
*   *Why Best Practice:* Safety. If the intern's account gets hacked, the hacker is stuck in that one folder and can't destroy the entire system.

### 10. Describe how AWS Regions and Availability Zones ensure fault tolerance.
*   **Region:** A large geographic location (like "US East Virginia") where AWS has data centers.
*   **Availability Zone (AZ):** Separate, isolated data centers within a Region. They are physically apart and have their own power/cooling.
*   **Fault Tolerance:** If you run your application in multiple AZs and one data center has a power outage, your application stays online because the other AZs keep working.

---

## Part C – Case Study / Application-based Questions

### 11. A company wants to build a fraud detection system using AWS AI/ML services. Which services would you recommend, and why?
**Recommendation:** **Amazon Fraud Detector** or **Amazon SageMaker**.
*   **Why Amazon Fraud Detector?** Use this if you want the "Easy Button". It is a ready-made service trained by Amazon to spot fraud. It's fast to set up and requires no ML expertise.
*   **Why SageMaker?** Use this if you want the "Custom Button". If your company has very specific types of fraud data, SageMaker lets you build a custom model that learns exactly how *your* fraud looks.

### 12. Suppose you are developing a student performance prediction system. How can Amazon SageMaker help in training and deploying the ML model?
*   **Training:** You upload student data (grades, attendance) to S3. Then you use **SageMaker Autopilot**, which looks at the data and automatically tries different formulas to find the one that best predicts grades.
*   **Deploying:** Once the model is learned, SageMaker creates a web link (Endpoint). Your school website can send student info to this link, and it will instantly answer with the predicted grade.

### 13. Design a small use case where Rekognition and Polly can work together in an application. Explain step by step.
**Use Case:** **"Smart Sight" App for the Visually Impaired.**
1.  **Capture:** The user points their phone camera at a scene (e.g., a park).
2.  **Analyze (Rekognition):** The app sends the photo to **Amazon Rekognition**. Rekognition sees "Tree", "Bench", and "Dog".
3.  **Process:** The app creates a sentence: "I see a park with a bench and a dog."
4.  **Speak (Polly):** This sentence is sent to **Amazon Polly**.
5.  **Output:** Polly turns the text into spoken audio, describing the scene to the user.

### 14. Your organization needs to give temporary access to a contractor for uploading files into an S3 bucket. How would you configure IAM roles and policies for this situation?
1.  **Create an IAM Role:** Make a "ContractorRole". Do not give them a permanent username/password.
2.  **Define Policy:** Write a rule that says "Allow Upload" *only* to the specific bucket `company-upload`. Deny everything else.
3.  **Assume Role:** Give the contractor a way to "assume" (put on) this role.
4.  **Session Duration:** Set a time limit (e.g., 4 hours). After 4 hours, their access automatically expires, keeping your system safe.

### 15. Compare traditional on-premise ML development with Cloud-based ML development (AWS SageMaker) in terms of cost, scalability, and ease of use.
| Feature | On-Premise ML (Traditional) | Cloud-Based ML (SageMaker) |
| :--- | :--- | :--- |
| **Cost** | **High Upfront:** You must buy expensive servers/GPUs before you even start. | **Pay-as-you-go:** You only pay for the minutes you are investigating/training. Cheap to start. |
| **Scalability** | **Hard:** Adding more power takes weeks (ordering, installing). | **Easy:** Click a button to get 100x more power instantly. |
| **Ease of Use** | **Complex:** You have to fix hardware, install drivers, and manage cooling. | **Simple:** Amazon handles the hardware. You just focus on the code. |



-----------------

# Assignment 2: Amazon SageMaker

## Part A – Short Answer Questions

### 1. What is Amazon SageMaker and how does it simplify the machine learning lifecycle?
**Amazon SageMaker** is a complete toolbox for Machine Learning. It helps developers do everything in one place: build the model, train it, and launch it. It simplifes the process because you don't have to worry about managing servers or complex infrastructure; SageMaker handles the heavy lifting for you.

### 2. Explain the role of Amazon S3 in SageMaker workflows.
**Amazon S3** is the storage unit.
*   **Input:** It holds the data (like photos or spreadsheets) that SageMaker needs to learn from.
*   **Output:** Once SageMaker finishes learning, it saves the final "brain" (the trained model) back to S3.

### 3. Differentiate between SageMaker Studio and SageMaker Notebooks.
*   **SageMaker Studio:** The "Cockpit". It is a full-featured dashboard where you can control every aspect of your ML project visually.
*   **SageMaker Notebooks:** These are like individual digital workbooks (running Jupyter) where you write and test code. They are simpler but less integrated than Studio.

### 4. What are the advantages of using built-in algorithms in SageMaker compared to custom algorithms?
*   **Speed:** They are optimized by Amazon to run very fast.
*   **Easy:** You don't have to write the math code from scratch. Just plug in your data.
*   **Cost:** Faster training means you pay less.

### 5. Define hyperparameters. Why is hyperparameter tuning important?
**Hyperparameters** are the "settings" or "knobs" you adjust before training begins (like how fast the model should learn).
*   **Importance:** The default settings are rarely the best. Tuning them (finding the perfect combination) is critical to getting the most accurate results possible.

### 6. List and explain any three preprocessing steps commonly performed before training a model.
1.  **Handling Missing Values:** Filling in blank spots in your data (like missing ages) so the computer doesn't get confused.
2.  **Encoding:** Converting text into numbers (e.g., turning "Red" and "Blue" into "1" and "2") because computers only understand numbers.
3.  **Feature Scaling:** Adjusting numbers so they are all on the same scale (0 to 1). This ensures that big numbers (like Salary) don't overpower small numbers (like Age).

### 7. What are the different deployment options in SageMaker?
1.  **Real-Time Inference:** Instant answers. Good for apps that need a reply immediately (like a chatbot).
2.  **Serverless Inference:** Good for apps that are rarely used. You only pay when someone actually uses it.
3.  **Batch Transform:** Processing a huge list of data all at once (like a nightly report).
4.  **Asynchronous Inference:** For big jobs that take a long time (like analyzing a long video).

### 8. Mention at least two AWS services that integrate with SageMaker for security and automation.
*   **AWS IAM:** The Security Guard. It controls who is allowed to touch your models.
*   **AWS KMS:** The Vault. It locks (encrypts) your data so no one can steal it.

### 9. Explain the importance of model monitoring after deployment.
Models can get "stale". For example, a model trained on data from 2020 might not understand the world in 2024. **Monitoring** watches the model to see if its accuracy is dropping (Drift) so you know when it's time to retrain it.

### 10. What is Boto3 and how is it used in SageMaker?
**Boto3** is a Python tool that acts like a remote control for AWS. It allows your python script to talk to AWS services, like creating buckets or starting training jobs automatically.

---

## Part B – Long Answer Questions

### 1. Draw and explain the high-level architecture diagram of Amazon SageMaker.
*(Concept Diagram: Build -> Train -> Deploy)*

1.  **Build (Notebooks):** You write your code in a notebook. You modify and prepare your data here.
2.  **Train (The Gym):** When you are ready, SageMaker spins up a clear, powerful group of computers just for training. They learn from the data and save the result to S3. Once done, they turn off to save money.
3.  **Deploy (The Job):** SageMaker takes the trained model and puts it on a permanent server (Endpoint) so apps can ask it questions 24/7.

### 2. Describe the workflow of an ML project in SageMaker from data collection to monitoring.
1.  **Data Collection:** Gather data and store it in S3. Clean it up.
2.  **Build:** Write code to visualize and understand the data.
3.  **Train:** Start a training job. SageMaker learns from the data.
4.  **Tune:** Try different settings to see which one works best.
5.  **Deploy:** Launch the best model to a web endpoint.
6.  **Monitor:** Keep watching the live model to make sure it stays accurate.

### 3. Explain with examples how data preprocessing is performed in SageMaker notebooks.
Preprocessing is about cleaning data before the AI sees it. We use Python code for this.

*   **Step 1: Loading Data**
    ```python
    # Theory: We use 'pandas' to read the CSV file from S3 so we can work on it.
    df = pd.read_csv('s3://my-bucket/titanic.csv')
    ```
*   **Step 2: Fixing Missing Data**
    ```python
    # Theory: Does the 'Age' column have blanks? We fill them with the average age.
    df['Age'].fillna(df['Age'].median(), inplace=True)
    ```
*   **Step 3: Encoding (Text to Numbers)**
    ```python
    # Theory: Computers hate text. We change 'male/female' to '0/1'.
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    ```
*   **Step 4: Save & Upload**
    ```python
    # Theory: Save the clean data and push it back to S3 for training.
    s3.upload_file('train.csv', 'my-bucket', 'processed/train.csv')
    ```

### 4. Compare real-time endpoints, batch transform, and asynchronous inference in model deployment.
| Feature | Real-Time | Batch Transform | Asynchronous |
| :--- | :--- | :--- | :--- |
| **Use Case** | Instant answers (Shopping). | Nightly reports. | Big files (Videos). |
| **Speed** | Initial response in Milliseconds. | Takes minutes/hours. | Seconds to minutes. |
| **Cost** | Runs 24/7 (Higher cost). | Runs only for the job (Lower cost). | Scales to zero when idle. |

### 5. Discuss advantages and challenges of using SageMaker in enterprise-scale ML projects.
**Advantages:**
*   **Scale:** You can use hundreds of computers at once without setting them up.
*   **Governance:** It keeps records of who did what, which is important for big companies.
*   **Automation:** You can automate the whole process so it runs by itself.

**Challenges:**
*   **Cost:** If you forget to turn off a big server, it can get expensive.
*   **Complexity:** There are so many tools in SageMaker, it can be overwhelming for beginners.

---

## Part C – Lab 3 and 4

*(Instructions for performing the lab steps. Explanations included for understanding.)*

### 1. Data Ingestion
**Theory:** First, we need to move our data from our local computer to the cloud (S3) so SageMaker can access it.
```python
import sagemaker
import boto3

# Create a session (connection) to SageMaker
sess = sagemaker.Session()
# Find the default storage bucket
bucket = sess.default_bucket() 
prefix = 'lab-data/bank-loan'

# Upload the file 'local_dataset.csv' to the S3 bucket
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'data.csv')).upload_file('local_dataset.csv')
print(f"Uploaded to s3://{bucket}/{prefix}")
```

### 2. Preprocessing in Notebooks
**Theory:** Raw data is messy. We load it, identify missing info, and convert categories into numbers that the algorithm can understand (One-Hot Encoding). Finally, we split the data: one part to teach the model (Train), and one part to test it (Validation).
```python
import pandas as pd
import numpy as np

# Load data from S3
data_location = f's3://{bucket}/{prefix}/data.csv'
df = pd.read_csv(data_location)

# Remove rows with missing data
df = df.dropna()

# Convert categorical text columns into numeric columns (e.g., Apple -> 1, Banana -> 0)
df = pd.get_dummies(df)

# Split data: 70% for training, 20% for validation, 10% for testing
train_data, validation_data, test_data = np.split(df.sample(frac=1, random_state=1729), [int(0.7 * len(df)), int(0.9 * len(df))])

# Save the files without headers (XGBoost requirement)
train_data.to_csv('train.csv', header=False, index=False)
validation_data.to_csv('validation.csv', header=False, index=False)
```

### 3. Model Training (XGBoost)
**Theory:** We define the "Estimator" (the model builder). We tell it which algorithm to use (XGBoost), what computer power we need (ml.m4.xlarge), and where to find the training data.
```python
from sagemaker.inputs import TrainingInput

# Get the official XGBoost software container
container = sagemaker.image_uris.retrieve("xgboost", boto3.Session().region_name, "latest")

# Set up the trainer
xgb = sagemaker.estimator.Estimator(container,
                                    role=sagemaker.get_execution_role(), 
                                    instance_count=1, 
                                    instance_type='ml.m4.xlarge',
                                    output_path=f's3://{bucket}/{prefix}/output')

# Set Hyperparameters (The settings for learning)
xgb.set_hyperparameters(max_depth=5, eta=0.2, gamma=4, min_child_weight=6, subsample=0.8, objective='binary:logistic', num_round=100)

# Start training using the data in S3
xgb.fit({'train': TrainingInput(f's3://{bucket}/train.csv', content_type='csv'),
         'validation': TrainingInput(f's3://{bucket}/validation.csv', content_type='csv')})
```

### 4. Hyperparameter Tuning
**Theory:** Instead of guessing the settings, we ask SageMaker to try many different combinations automatically to find the best one. This is "Tuning".
```python
from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner

# Define the range of values to test
hyperparameter_ranges = {'eta': ContinuousParameter(0, 1), 
                        'min_child_weight': IntegerParameter(1, 10), 
                        'max_depth': IntegerParameter(1, 10)}

# Setup the automated tuner
tuner = HyperparameterTuner(xgb,
                            objective_metric_name='validation:accuracy',
                            hyperparameter_ranges=hyperparameter_ranges,
                            max_jobs=10,
                            max_parallel_jobs=3)

# Start tuning
tuner.fit({'train': ..., 'validation': ...})
```

### 5. Model Evaluation
**Theory:** We check the "Accuracy" score. We compare what the model predicted against the real answers to see how smart it is.

### 6. Deployment
**Theory:** We take our trained model and put it on a live server so we can use it.
```python
# Create a live web endpoint
xgb_predictor = xgb.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

# Send some test data to get a prediction
result = xgb_predictor.predict(test_data_payload)
print(result)
```

### 7. Monitoring
**Theory:** We turn on "Data Capture" to record every question and answer the model processes. This lets us investigate later if something goes wrong.

---

## Part D – Case Study: Financial Loan Default Prediction

### 1. Problem Statement
**Goal:** Build an AI to predict if a customer will pay back a loan (Repay) or not (Default).
**Data:** Financial history (Income, Credit Score, Loan Amount).

### 2. Implementation Steps

#### Step 1: Data Preprocessing
*   **Cleaning:** We filled in missing credit scores with the average.
*   **Formatting:** We moved the answer column (`Loan_Status`) to the front because the algorithm requires it there.
*   **Result:** Clean files were saved to S3.

#### Step 2: Model Training
*   **Tool:** We used the **XGBoost** algorithm on SageMaker.
*   **Result:** The first try gave us **78% accuracy**.

#### Step 3: Hyperparameter Tuning
*   **Optimization:** We used the Tuner to try different settings (`alpha` and `max_depth`).
*   **Result:** The accuracy improved to **84.5%**. We found the perfect settings.

#### Step 4: Deployment & Testing
*   **Launch:** We launched the model as an endpoint `bank-loan-pred`.
*   **Test:** We sent a fake customer with a high credit score (720). The model predicted `0.04`, which means a 4% chance of default. This is low, so we **Approve** the loan.

#### Step 5: Monitoring Drift
*   **Simulation:** We sent weird data (Credit Score = 0) to the model.
*   **Outcome:** SageMaker Model Monitor noticed that this data looked "wrong" compared to what it learned from. It triggered an alarm to warn us.

### 4. How does SageMaker compare with Google Vertex AI or Azure ML Studio?
*   **AWS SageMaker:** Best if you are already using AWS services. Very powerful "Studio" interface.
*   **Google Vertex AI:** Great integration with Google data tools. Often cited as having easier-to-use AutoML for images.
*   **Azure ML:** Works great if you use Microsoft products like Excel or PowerBI.

### 5. Explain how serverless architecture in AWS Lambda can integrate with SageMaker.
Lambda is the "connector".
*   **Scenario:** A mobile app uploads a check image.
*   **Flow:** The upload triggers **Lambda**. Lambda wakes up, sends the image to the **SageMaker** model, gets the prediction, saves it to the database, and then goes back to sleep.
*   **Benefit:** You don't need a dedicated server running 24/7 just to wait for checks.

---------------------
# Assignment 3: AWS Language AI Services

## Part A – Conceptual Questions

### 1. Explain the core architecture and working mechanism of Amazon Comprehend.
**Amazon Comprehend** is a service that understands text. It uses Pre-trained models to read documents and find insights.
*   **How it works:**
    1.  **Input:** You send it text (like an email or a review).
    2.  **Process:** It looks for entities (names/places), Key Phrases (important topics), Sentiment (Happy/Sad), and Language (English/Spanish).
    3.  **Output:** It gives you a structured report of what it found.

### 2. Use Case: Propose how Amazon Comprehend could be used by an e-commerce company to automatically extract and analyze customer feedback from product reviews.
**Scenario:** Analyzing thousands of product reviews.
1.  **Solution:** Feed all reviews into Comprehend.
2.  **Analysis:**
    *   **Sentiment:** Check if reviews are mostly Positive or Negative.
    *   **Keywords:** Find out *why*. If "Battery" appears often with "Negative" sentiment, you know the battery is the problem.
3.  **Action:** The system automatically alerts the engineering team about the battery issue.

### 3. Describe the main components of Amazon Lex (Intents, Utterances, Slots, Prompts).
Amazon Lex builds chatbots.
*   **Intent:** The goal. What does the user want to do? (e.g., "Book Hotel").
*   **Utterance:** What the user actually says. (e.g., "I need a room" or "Book me a place").
*   **Slot:** The specific details needed. (e.g., "Which city?" or "What date?").
*   **Prompt:** The question the bot asks to get those details. (e.g., "Sure, what city are you traveling to?").

### 4. What role does Automatic Speech Recognition (ASR) and Natural Language Understanding (NLU) play in Lex?
*   **ASR (The Ears):** It listens to your voice and converts sound into text.
*   **NLU (The Brain):** It reads the text and understands what you *mean* (your Intent).

### 5. Use Case: Design a banking chatbot using Amazon Lex that helps customers check their account balance and transfer funds. Identify intents, utterances, slots.
**Bot:** BankAssist
*   **Task 1: Check Balance**
    *   **Use says (Utterance):** "How much money do I have?"
    *   **Bot needs (Slot):** "Which account? Savings or Checking?"
*   **Task 2: Transfer Money**
    *   **User says (Utterance):** "Send money to Mom."
    *   **Bot needs (Slot):** "How much?" and "From which account?"

### 6. Compare the different voice generation technologies: Standard TTS, Neural TTS (NTTS), and Generative Engine.
*   **Standard TTS:** Sounds a bit robotic. Stitches recorded sounds together.
*   **Neural TTS:** Uses AI to sound very smooth and natural.
*   **Generative:** The newest tech. It sounds incredibly human, with emotion and adaptive style.

### 7. Use Case: Imagine you are building an e-learning platform for visually impaired students. Explain how Amazon Polly can make the platform more inclusive and effective.
**Amazon Polly** reads text out loud.
*   It can read textbooks and quizzes to students who cannot see the screen.
*   It helps them navigate the website by reading menu buttons.
*   It can read in many languages, helping students worldwide.

### 8. Outline the key steps involved in Amazon Transcribe’s processing pipeline.
1.  **Input:** You upload an audio file.
2.  **Clean:** It filters out background noise.
3.  **Identify:** It listens to the sounds to find words.
4.  **Format:** It adds punctuation (commas, periods) to make it readable.
5.  **Output:** It gives you the full written text.

### 9. How does Speaker Diarization and Custom Vocabulary improve transcription accuracy?
*   **Speaker Diarization:** This tells you *who* is speaking. Instead of a block of text, it says "Speaker A: Hello", "Speaker B: Hi there". Great for interviews.
*   **Custom Vocabulary:** This teaches the AI special words it might not know, like your company name or medical slang, so it spells them correctly.

### 10. Use Case: Describe how a media company can use Amazon Transcribe to automatically generate subtitles for live news broadcasts.
1.  **Live Stream:** Feed the live TV audio to Transcribe.
2.  **Process:** Transcribe listens in real-time.
3.  **Caption:** It turns speech to text instantly and sends it back.
4.  **Display:** The TV station shows this text at the bottom of the screen as closed captions for viewers.

---

## Part B – Analytical & Application Questions

### 11. Mini Project: Healthcare Automation
**Project:** Automating Doctor Notes.
*   **Problem:** Doctors spend too much time typing notes.
*   **Solution:**
    1.  Doctor speaks into an app (Voice Note).
    2.  **Amazon Transcribe Medical** turns the voice into text. It understands complex drug names.
    3.  **Amazon Comprehend Medical** reads the text and picks out the "Dosage", "Medicine", and "Diagnosis".
    4.  The system saves this organized info into the patient's file automatically.
*   **Ethics:** We must be very careful with privacy (HIPAA) and always have a human double-check the notes because a mistake could be dangerous.

---

## Part C – Research & Real-World Analysis

### 12. Case Study: BMW Group
**Overview:** BMW uses AWS AI to help its global factories work together.
*   **Translation:** They use **Amazon Translate** so engineers in Germany can read reports from Mexico instantly. No language barrier.
*   **GenAI:** They use **Amazon Bedrock** (Generative AI) to read huge supplier contracts and summarize them in seconds, saving hours of reading time.

---------------------------------------------------

# Assignment 4: AWS Rekognition

## Part A – Conceptual Questions

### 1. Explain the architecture of AWS Rekognition.
**AWS Rekognition** is a "Vision" service. It uses deep learning models that Amazon has already trained on billions of images.
*   **Serverless:** You don't need servers. You just send an image to the API.
*   **Pre-trained:** It already knows what a "car" or "person" looks like.
*   **Storage:** It works directly with images stored in Amazon S3.

### 2. List Rekognition’s key features.
*   **Object Detection:** Finds things like "Dog", "Car", "Tree".
*   **Facial Analysis:** Detects emotions (Happy/Sad) and attributes (Glasses/Beard).
*   **Face Compare:** Checks if two photos show the same person.
*   **Text OCR:** Reads text inside images (like street signs).
*   **Moderation:** Detects unsafe or inappropriate images.

### 3. How does Rekognition integrate with S3 and Lambda?
*   **S3:** Holds the images.
*   **Lambda:** Acts as the trigger. When you upload a photo to S3, Lambda wakes up and tells Rekognition "Hey, go check this photo". Rekognition checks it and gives the answer back to Lambda.

### 4. Explain Rekognition Custom Labels.
Sometimes the standard model isn't enough. **Custom Labels** lets you teach Rekognition new objects.
*   *Example:* Teaching it to recognize your specific *Company Logo* or a specific *Engine Part*. You upload examples, and it learns to spot them.

### 5. Discuss its advantages and limitations.
*   **Advantages:** It's super easy (no coding ML models), cheap (pay per image), and fast.
*   **Limitations:** It's not magic—blurry photos give bad results. Also, facial recognition has privacy concerns and must be used carefully.

---

## Part B: Case Study–Based Assignment

### Case Study 1: Smart Campus Security System
**Goal:** Automate entry using Face ID.
1.  **Design:** Store student photos in a "Collection". When a student walks up, the camera takes a picture. Rekognition compares the live picture to the stored collection.
2.  **Workflow:** Camera -> S3 -> Lambda -> Rekognition (SearchFaces) -> Unlock Door.
3.  **Privacy:** Encrypt all data. Get student permission first. Delete old data automatically.

### Case Study 2: Retail Analytics
**Goal:** Understand customers.
1.  **Usage:** Use Rekognition to see if customers are "Happy" or "Angry" at checkout.
2.  **Data Flow:** CCTV Video -> Lambda -> Rekognition -> Count "Angry Faces" -> Send Alert if too many.
3.  **Visualization:** Show a graph of "Customer Happiness by Hour" on a dashboard.

### Case Study 3: Hospital Patient Monitoring
**Goal:** Detect falls.
1.  **Solution:** Use Rekognition Video to monitor the room.
2.  **Alert:** If it sees the action "Falling", it triggers an alert (via SNS) to the nurse's pager immediately.
3.  **Compliance:** Use privacy masking and strong encryption (KMS) to protect patient dignity and data.

### Case Study 4: Media Video Tagging
**Goal:** Tag celebrities and scenes in movies.
1.  **APIs:** Use `StartCelebrityRecognition` to find actors. Use `StartLabelDetection` to find scenes (like "Beach").
2.  **Workflow:** Upload Video -> Rekognition scans it -> Saves tags ("Brad Pitt", "00:10:05") to a database.
3.  **Benefit:** Editors can search "Show me all shots of Brad Pitt" and find them instantly.

### Case Study 5: KYC Verification (Identity Check)
**Goal:** Verify a user's identity online.
1.  **Method:** Ask user to upload a photo of their ID card and a selfie.
2.  **Process:** Use `CompareFaces` to check if the Selfie matches the ID Card photo.
3.  **Safety:** Use `DetectText` to read the name on the ID card and check if it matches the application form.

-------------------------------------

# Assignment 5: Serverless Computing for AI Workflows

### 1. Explain serverless computing in your own words. Describe how it works and list any three advantages.
**Serverless Computing** means you focus only on your code, and Amazon handles the servers. It's like staying in a hotel—you use the room, but you don't fix the plumbing or pay the mortgage.
*   **How it works:** You upload a function (code). It sits there doing nothing (and costing nothing) until an "Event" (like a click) triggers it. It runs, does the job, and turns off.
*   **Advantages:**
    1.  **No Management:** No servers to fix or patch.
    2.  **Auto-Scaling:** It can handle 1 user or 1 million users automatically.
    3.  **Cost:** You never pay for idle time. If no one uses it, you pay $0.

### 2. Draw or describe a simple AI workflow using AWS services. Explain how data flows through each stage.
**Workflow:** Image Analyzer.
*   **User** uploads photo -> **API Gateway** (The Front Door) receives it -> **Lambda** (The Manager) takes it -> **Step Functions** (The Checklist) organizes the steps -> **AI Service** analyzes it -> **S3** saves the result.

### 3. What is the role of AWS Lambda in an AI workflow? Give one real-world example of how Lambda is used for AI tasks.
*   **Role:** Lambda is the "Glue". It connects different services. It takes data from A, cleans it, gives it to the AI, gets the answer, and puts it in B.
*   **Example:** In a chatbot, Lambda is the code that actually checks the database to find your account balance when you ask "What's my balance?".

### 4. Describe how AWS CloudWatch helps in monitoring serverless AI workflows. Explain logs, metrics, and alarms with examples.
**CloudWatch** is the dashboard for health.
*   **Logs (The Diary):** Detailed records of what happened. "Error: File not found."
*   **Metrics (The Scoreboard):** Numbers. "500 requests today." "Average speed: 200ms."
*   **Alarms (The Siren):** Alerts. "Warning! The system is failing too often!" -> Sends an email to the IT team.

### 5. List and explain any three security practices used to secure AI APIs in AWS.
1.  **IAM (Permissions):** Give pieces of your app only the permission they need. (Least Privilege).
2.  **KMS (Encryption):** Encrypt your data so it looks like nonsense to hackers.
3.  **Throttling:** Set a speed limit on your API so bad guys can't crash your system by sending too many requests at once.
