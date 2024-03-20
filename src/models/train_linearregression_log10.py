import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from features.transformers import Log10Transformer, MyMinMaxScaler, MyKNNImputer
from models.model_utils import save_model_as_pickle
from features.pipeline import base_pipeline, base_after_split_pipeline
from utils import load_data, visualize_metrics

df = load_data()

base_pipeline = Pipeline([
    ('base_pipeline', base_pipeline),
    ('Log Scale',
     Log10Transformer(columns=['Bathroom Count', 'Bedroom Count', 'Habitable Surface', 'Land Surface', 'Price'])),
])

after_split_pipeline = Pipeline([
    ('base_after_split_pipeline', base_after_split_pipeline),
    # add more to pipeline
])

df = base_pipeline.transform(df)

X = df.drop(columns=['Price'])
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=41, test_size=0.2)

X_train = after_split_pipeline.fit_transform(X_train)

reg_model = LinearRegression()

reg_model.fit(X_train, y_train)

# prediction
X_test = after_split_pipeline.transform(X_test)
y_pred = reg_model.predict(X_test)
y_pred_train = reg_model.predict(X_train)

# scoring the model
r_squared = r2_score(y_train, y_pred_train)
print(f"{os.path.basename(__file__)} - train r_squared: {r_squared:.2%}")
visualize_metrics(r_squared, y_train, y_pred_train, comments="Train set")

r_squared = r2_score(y_test, y_pred)
print(f"{os.path.basename(__file__)} - test r_squared: {r_squared:.2%}")
visualize_metrics(r_squared, y_test, y_pred, comments="Test set")

# save the model
save_model_as_pickle(reg_model, os.path.basename(__file__))
