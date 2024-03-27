import os

from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

from features.transformers import Log10Transformer, MyMinMaxScaler, MyKNNImputer, MyStandardScaler
from models.model_utils import save_model_as_pickle
from features.pipeline import base_pipeline, base_after_split_pipeline, pre_pipeline
from utils import load_data, visualize_metrics

df = load_data()

df = pre_pipeline.transform(df)

X = df.loc[:, df.columns != "Price"]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=50, test_size=0.2)

numerical_pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('imputer', KNNImputer(n_neighbors=5)),
])
# Find the numerical columns
numerical_columns = X_train.select_dtypes(include=['int', 'float']).columns

# Find the categorical columns
categorical_columns = X_train.select_dtypes(include=['object']).columns

X_train = X_train.drop(columns=categorical_columns, inplace=True)
X_test = X_test.drop(columns=categorical_columns, inplace=True)

preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, ['Postal Code', 'Longitude', 'Latitude']),
])

regression_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('My standard Scaler', MyStandardScaler()),
    ('model', LinearRegression())
])

regression_pipeline.fit(X_train, y_train)

y_pred = regression_pipeline.predict(X_test)

print(regression_pipeline.score(X_test, y_test))