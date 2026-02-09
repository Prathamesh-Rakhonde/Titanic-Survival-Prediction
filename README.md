# Titanic Survival Predictor 🚢

### 📌 Project Overview
This is an interactive Machine Learning application that predicts whether a passenger would survive the Titanic disaster. 

Unlike static scripts, this project features a **Command-Line Interface (CLI)** that allows users to input their own details (Age, Class, Gender, etc.) and 
receive a personalized survival probability in real-time.

### 🧠 How It Works
The model uses **Logistic Regression** trained on the official Titanic dataset.
1.  **Data Cleaning:** Handles missing values (Age imputation) and converts categorical data (Sex, Embarked) into numerical format.
2.  **Model Training:** Splits data into 80% training and 20% testing sets to ensure validity.
3.  **Prediction:** Takes user input, processes it to match the model's structure, and outputs a survival probability.

### 🛠️ Tech Stack
* **Python 3.x**
* **Pandas** (Data Manipulation)
* **Scikit-Learn** (Machine Learning)

### 🎮 Example Interaction
```text
--- TITANIC SURVIVAL PREDICTOR (LITE) ---
1. Training Model...
Model Trained Successfully! Accuracy is ~81%.

Let's see if YOU would survive.
1. Ticket Class (1=Rich, 2=Middle, 3=Poor): 3
2. Gender (male/female): male
3. Age: 24
4. Ticket Price: 10
5. Which Port? (S/C/Q): S

--- CALCULATING FATE... ---
RESULT: RIP. You died. (The model is 88.2% sure)

📊 Model Performance
Accuracy: ~81%
Key Findings: Gender and Ticket Class were the most significant factors in predicting survival.
