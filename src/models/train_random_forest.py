import os

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from features.transformers import Log10Transformer, MyMinMaxScaler, MyKNNImputer
from models.model_utils import save_model_as_pickle
from models.pipeline import base_pipeline
from utils import load_data, visualize_metrics

df = load_data()

print(df.info())

base_pipeline = Pipeline([
    ('base_pipeline', base_pipeline),
    ('Log Scale',
     Log10Transformer(columns=['Bathroom Count', 'Bedroom Count', 'Habitable Surface', 'Land Surface', 'Price'])),
])

df = base_pipeline.transform(df)

pipeline = Pipeline([
    ('Min Max scaler', MyMinMaxScaler(
        columns=['Land Surface', 'Habitable Surface', 'Bathroom Count', 'Toilet Count', 'Postal Code', 'Longitude',
                 'Latitude', 'Facades', 'Subtype', 'Consumption', 'State of Building', 'Kitchen Type',
                 'cd_munty_refnis', 'PopDensity', 'MedianPropertyValue', 'NetIncomePerResident'],
        multipliers={'Subtype': 100}  # make Subtype dominant for KNN
    )),
    ('KNN Toilets', MyKNNImputer(columns=['Habitable Surface', 'Bathroom Count', 'Toilet Count', 'Subtype'])),
    ('KNN Lon, Lat', MyKNNImputer(columns=['Postal Code', 'Longitude', 'Latitude'])),
    ('KNN Facade', MyKNNImputer(columns=['Facades', 'Land Surface', 'Habitable Surface', 'Subtype'])),
    ('KNN Consumption', MyKNNImputer(columns=['Consumption', 'State of Building', 'Kitchen Type'])),
    ('KNN REFNIS blanks', MyKNNImputer(
        columns=['Longitude', 'Latitude', 'cd_munty_refnis', 'PopDensity', 'MedianPropertyValue',
                 'NetIncomePerResident']
    )),
])
X = df.drop(columns=['Price'])
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=41, test_size=0.2)

X_train = pipeline.fit_transform(X_train)

random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=20)

random_forest_model.fit(X_train, y_train)

# prediction
X_test = pipeline.transform(X_test)
y_pred = random_forest_model.predict(X_test)

y_pred_train = random_forest_model.predict(X_train)

r_squared = r2_score(y_train, y_pred_train)
print(f"{os.path.basename(__file__)} - train r_squared: {r_squared:.2%}")

r_squared = r2_score(y_test, y_pred)
print(f"{os.path.basename(__file__)} - test r_squared: {r_squared:.2%}")

# save the model
save_model_as_pickle(random_forest_model, os.path.basename(__file__))

visualize_metrics(r_squared, y_test, y_pred)
