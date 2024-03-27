import os

import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from features.build_features import AddAvgRegionPrice
from features.pipeline import pre_pipeline
from features.transformers import MyKNNImputer, CatBoostRegressorWrapper, MyMinMaxScaler, NaNToCategoryTransformer, \
    ResetIndexTransformer
from models.model_utils import save_model_as_pickle
from utils import load_data, visualize_metrics

pd.set_option('future.no_silent_downcasting', True)

df = load_data()

df = pre_pipeline.transform(df)

X = df.drop(columns=['Price'])
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=41, test_size=0.2)

X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

categorical_features_indices = [i for i, dtype in enumerate(df.dtypes) if dtype not in ['int64', 'float64']]

print(X_train.columns)

pipeline = Pipeline([
    (
        'Reset Index', ResetIndexTransformer()
    ),
    (
        "Impute Nan in 'Kitchen Type' with 'NOT_INSTALLED'",
        NaNToCategoryTransformer(columns=['Kitchen Type'], replacement='NOT_INSTALLED')
    ),
    (
        "Impute Nan with 'MISSING' for categorical columns",
        NaNToCategoryTransformer(replacement='MISSING')
    ),
    (
        "MinMaxScale 'Postal Code', 'Longitude', 'Latitude'",
        MyMinMaxScaler(columns=['Postal Code', 'Longitude', 'Latitude'])
    ),
    (
        "KnnImpute 'Postal Code', 'Longitude', 'Latitude'",
        MyKNNImputer(columns=['Postal Code', 'Longitude', 'Latitude'], n_neighbors=2)
    ),
    (
        'Add RegionPricePerSqm',
        AddAvgRegionPrice(neighbors=5)
    ),
    (
        'Cat Boost Regressor',
        CatBoostRegressorWrapper(
            iterations=10000,
            learning_rate=0.01,
            depth=6,
            eval_metric='RMSE',
            random_seed=42,
            early_stopping_rounds=50,
            verbose=100,
            fit_params={'cat_features': categorical_features_indices, 'use_best_model': True}
        )
    ),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# scoring the model
r_squared = r2_score(y_test, y_pred)
print(r_squared)
visualize_metrics(r_squared, y_test, y_pred, comments="Train set")

# # save the model
save_model_as_pickle(pipeline, os.path.basename(__file__))
