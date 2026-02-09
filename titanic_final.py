import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

print("Loading Data...")
df = pd.read_csv('train.csv')

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'], inplace=True)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print("Training Model...")
X = df.drop('Survived', axis=1)
y = df['Survived'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Predicting for YOU...")

# my_data = [[3, 0, 24, 0, 0, 10, 0, 1]]
# my_data = [[1, 0, 24, 0, 0, 10, 0, 1]]
# my_data = [[3, 1, 24, 0, 0, 10, 0, 1]]
my_data = [[2, 1, 21, 0, 0, 10, 0, 1]]
my_passenger = pd.DataFrame(my_data, columns=X.columns)

my_prediction = model.predict(my_passenger)

if my_prediction[0] == 1: print("RESULT: YOU SURVIVED!")
else: print("RESULT: RIP. You died.")