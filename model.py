# import pandas as pd
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import LabelEncoder

# # Load dataset
# df = pd.read_csv("medical.csv")

# # Encode categorical columns
# le = LabelEncoder()

# df["sex"] = le.fit_transform(df["sex"])
# df["smoker"] = le.fit_transform(df["smoker"])
# df["region"] = le.fit_transform(df["region"])

# # Features and target
# X = df.drop("charges", axis=1)
# y = df["charges"]

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Train model
# model = RandomForestRegressor(random_state=42)
# model.fit(X_train, y_train)

# # Save model
# pickle.dump(model, open("model.pkl", "wb"))

# print("Model saved successfully as model.pkl")

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("medical.csv")

print(df.head())

df["sex"] = df["sex"].map({
    "male":1,
    "female":0
})

df["smoker"] = df["smoker"].map({
    "yes":1,
    "no":0
})

df["region"] = df["region"].map({
    "northeast":0,
    "northwest":1,
    "southeast":2,
    "southwest":3
})

X = df.drop("charges", axis=1)

y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

lr = LinearRegression()

lr.fit(X_train, y_train)

prediction = lr.predict(X_test)

print("R2 Score :", r2_score(y_test, prediction))

print("MAE :", mean_absolute_error(y_test, prediction))

print("RMSE :", mean_squared_error(y_test, prediction) ** 0.5)

ridge = Ridge(alpha=1.0)

ridge.fit(X_train, y_train)

ridge_pred = ridge.predict(X_test)

print("Ridge R2 :", r2_score(y_test, ridge_pred))

lasso = Lasso(alpha=0.1)

lasso.fit(X_train, y_train)

lasso_pred = lasso.predict(X_test)

print("Lasso R2 :", r2_score(y_test, lasso_pred))

joblib.dump(lr, "model.pkl")

joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully!")