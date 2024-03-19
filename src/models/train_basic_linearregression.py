import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from models.model_utils import save_model_as_pickle
from models.pipeline import base_pipeline
from utils import load_data

df = load_data()

pipeline = Pipeline([
    ('base_pipeline', base_pipeline),
])

df = pipeline.transform(df)

print(df.info())

X = df.drop(columns=['Price'])
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=41, test_size=0.2)

reg_model = LinearRegression()

reg_model.fit(X_train, y_train)

# prediction
y_pred = reg_model.predict(X_test)

# score of reg model
score = reg_model.score(X_test, y_test)
print(f"{os.path.basename(__file__)} - score: {score}")

print(y_test)
print(y_pred)
# mse score
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
print(f"{os.path.basename(__file__)} - r_squared: {r_squared:.2%}")

# save the model
save_model_as_pickle(reg_model, os.path.basename(__file__))
