import os
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from features.transformers import Log10Transformer, MyMinMaxScaler, MyKNNImputer
from models.model_utils import save_model_as_pickle
from models.pipeline import base_pipeline
from utils import load_data

df = load_data()

print(df.info())

base_pipeline = Pipeline([
    ('base_pipeline', base_pipeline),
    ('Log Scale', Log10Transformer(columns=['Bathroom Count', 'Bedroom Count', 'Habitable Surface', 'Land Surface', 'Price'])),
])

df = base_pipeline.transform(df)

pipeline = Pipeline([
    ('Min Max scaler', MyMinMaxScaler(columns=['Habitable Surface'])),
    ('KNN toilets', MyKNNImputer(columns=['Habitable Surface', 'Bathroom Count', 'Toilet Count'])),
    ('KNN Lon, Lat', MyKNNImputer(columns=['Postal Code', 'Longitude', 'Latitude'])),
    ('KNN Facade', MyKNNImputer(columns=['Facades', 'Land Surface', 'Habitable Surface', 'Subtype'])),
    ('KNN Consumption', MyKNNImputer(columns=['Consumption', 'State of Building', 'Kitchen Type'])),
])

X = df.drop(columns=['Price'])
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=41, test_size=0.2)

X_train = pipeline.fit_transform(X_train)

reg_model = LinearRegression()

reg_model.fit(X_train, y_train)

# prediction
X_test = pipeline.transform(X_test)
y_pred = reg_model.predict(X_test)

# score of reg model
score = reg_model.score(X_test, y_test)
print(f"{os.path.basename(__file__)} - score: {score}")

# mse score
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
print(f"{os.path.basename(__file__)} - r_squared: {r_squared:.2%}")

# save the model
save_model_as_pickle(reg_model, os.path.basename(__file__))
