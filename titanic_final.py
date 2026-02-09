import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

print("Loading Data...")
df = pd.read_csv('train.csv')

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId','SibSp', 'Parch'], inplace=True)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print("Training Model...")
X = df.drop('Survived', axis=1)
y = df['Survived'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Model Trained Successfully! Accuracy is ~81%.")
print("--------------------------------------------------")

print("\nLet's see if YOU would survive.")
print("Please answer the following questions:")

pclass = int(input("1. What Class ticket? (1 = Rich, 2 = Middle, 3 = Poor) : "))

sex_input = input("2. What is your Gender? (male/female) : ").strip().lower()
sex = 1 if sex_input == 'female' else 0

age = float(input("3. What is your Age?: "))

fare = float(input("4. How much did you pay? (e.g., 10 for poor, 100+ for rich): "))

embarked_input = input("5. Which Port? (S=Southampton, C=Cherbourg, Q=Queenstown): ").strip().upper()

embarked_Q = 1 if embarked_input == 'Q' else 0
embarked_S = 1 if embarked_input == 'S' else 0

user_data = pd.DataFrame([[pclass, sex, age, fare, embarked_Q, embarked_S]], columns=X.columns)

prediction = model.predict(user_data)
probability = model.predict_proba(user_data)

if prediction[0] == 1: 
    chance = probability[0][1] * 100
    print(f"RESULT: YOU SURVIVED! (The model is {chance:.1f}% sure)")
else: 
    chance = probability[0][0] * 100
    print(f"RESULT: RIP. You died. (The model is {chance:.1f}% sure)")
