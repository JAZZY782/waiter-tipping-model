import pandas as pd
import numpy as np
data=pd.read_csv("tips.csv")
print(data.head())
data["sex"] = data["sex"].map({"Female": 0, "Male": 1})
data["smoker"] = data["smoker"].map({"No": 0, "Yes": 1})
data["day"] = data["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})
data["time"] = data["time"].map({"Lunch": 0, "Dinner": 1})
print(data.head())
x = np.array(data[["total_bill", "sex", "smoker", "day",
"time", "size"]])
y = np.array(data["tip"])
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y,
test_size=0.2,
random_state=42)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtrain, ytrain)
features = np.array([[24.50, 1, 0, 0, 1, 4]])
print(model.predict(features))
