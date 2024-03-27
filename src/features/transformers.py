import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from catboost import CatBoostRegressor

class RowFilter(BaseEstimator, TransformerMixin):
    """
    filter rows based on a condition.
    """

    def __init__(self, condition):
        # Condition should be a function that takes a DataFrame and returns a boolean Series
        self.condition = condition

    def fit(self, X, y=None):
        # Nothing to fit, so just return self
        return self

    def transform(self, X):
        filtered_X = X[self.condition(X)].copy()
        filtered_X.reset_index(drop=True, inplace=True)
        return filtered_X

    def inverse_transform(self, X):
        return X


class DropNARows(BaseEstimator, TransformerMixin):
    """
    drop rows with missing values for the specified columns.
    """

    def __init__(self, columns):
        # List of columns to check for missing values
        self.columns = columns

    def fit(self, X, y=None):
        # Nothing to fit, so just return self
        return self

    def transform(self, X):
        # Drop rows with missing values in the specified columns
        filtered_X = X.dropna(subset=self.columns).copy()
        filtered_X.reset_index(drop=True, inplace=True)
        return filtered_X

    def inverse_transform(self, X):
        return X


class ResetIndexTransformer(BaseEstimator, TransformerMixin):
    """resets the index of the DataFrame."""

    def __init__(self, drop=True):
        self.drop = drop  # Whether to drop the old index column or not

    def fit(self, X, y=None):
        return self  # Nothing to do here

    def transform(self, X):
        X_copy = X.copy()
        X_copy.reset_index(drop=self.drop, inplace=True)
        return X_copy

    def inverse_transform(self, X):
        return X


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Drops or keeps the specified columns."""

    def __init__(self, keep_columns=None, drop_columns=None):
        self.keep_columns = keep_columns if keep_columns else []
        self.drop_columns = drop_columns if drop_columns else []
        if keep_columns and drop_columns:
            raise ValueError("You can't have both keep_columns and drop_columns")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.keep_columns:
            return X_copy[self.keep_columns]
        X_copy.drop(columns=self.drop_columns, inplace=True)
        return X_copy

    def inverse_transform(self, X):
        return X


class FillLandSurfaceForApartment(BaseEstimator, TransformerMixin):
    """
    Fill the Land Surface with 0 for the APARTMENT subtype.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy.loc[X_copy['Subtype'] == 'APARTMENT', 'Land Surface'] = 0
        return X_copy

    def inverse_transform(self, X):
        return X


class FacadeFixer(BaseEstimator, TransformerMixin):
    """
    Fix the Facade column by replacing the values with the most frequent value.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['Facades'] = X_copy['Facades'].apply(lambda x: 2 if x < 2 else x)
        X_copy['Facades'] = X_copy['Facades'].apply(lambda x: 4 if x > 4 else x)
        return X_copy

    def inverse_transform(self, X):
        return X


class Log10Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, add_constant=True):
        self.columns = columns
        self.add_constant = add_constant

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        if self.columns is not None:
            for col in self.columns:
                # Check if add_constant is True
                if self.add_constant:
                    # Adding a small constant to avoid log10(0)
                    X_transformed[col] = np.log10(X_transformed[col] + 1)
                else:
                    X_transformed[col] = np.log10(X_transformed[col])
        return X_transformed

    def inverse_transform(self, X):
        X_reversed = X.copy()
        if self.columns is not None:
            for col in self.columns:
                if self.add_constant:
                    # Inverse transformation considering the constant
                    X_reversed[col] = np.power(10, X_reversed[col]) - 1
                else:
                    X_reversed[col] = np.power(10, X_reversed[col])
        return X_reversed


class MyMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, feature_range=(0, 1)):
        self.column_scalers = {}
        self.column_types = {}
        for column in columns:
            self.column_scalers[column] = MinMaxScaler(feature_range=feature_range)

    def fit(self, X, y=None):
        for column, scaler in self.column_scalers.items():
            self.column_scalers[column].fit(X[[column]])
            self.column_types[column] = X[column].dtype
        self.column_names = X.columns
        return self

    def transform(self, X):
        X_copy = X.copy()
        for column, scaler in self.column_scalers.items():
            X_copy[column] = scaler.transform(X_copy[[column]])
        X_copy = pd.DataFrame(X_copy, columns=self.column_names)
        return X_copy

    def inverse_transform(self, X):
        X_copy = X.copy()
        for column, scaler in self.column_scalers.items():
            scaled_data = scaler.inverse_transform(X_copy[[column]])
            if self.column_types[column].kind in ['i', 'u']:  # Check if type is integer or unsigned integer
                scaled_data = np.round(scaled_data).astype(self.column_types[column])  # Round and convert to original type
            X_copy[column] = scaled_data
        X_copy = pd.DataFrame(X_copy, columns=self.column_names)
        return X_copy


class MyKNNImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, n_neighbors=5):
        self.columns_to_impute = columns
        self.n_neighbors = n_neighbors
        self.imputer = KNNImputer(n_neighbors=n_neighbors)

    def fit(self, X, y=None):
        self.imputer.fit(X[self.columns_to_impute])
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.columns_to_impute] = self.imputer.transform(X_copy[self.columns_to_impute])
        return X_copy

    def inverse_transform(self, X):
        return X


class MyOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column, map=None):
        self.map = map if map else {}
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.column] = X_copy[self.column].map(lambda x: self.map.get(x, -1))
        return X_copy

    def inverse_transform(self, X):
        return X


class BuildingStateOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.building_state_map = {
            "AS_NEW": 5,
            "JUST_RENOVATED": 4,
            "GOOD": 3,
            "TO_BE_DONE_UP": 2,
            "TO_RENOVATE": 1,
            "TO_RESTORE": 0,
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['State of Building'] = X_copy['State of Building'].map(lambda x: self.building_state_map.get(x, -1))
        return X_copy

    def inverse_transform(self, X):
        return X


class ConsumptionFixer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        epc_map = {
            "A": 50,
            "B": 150,
            "C": 250,
            "D": 350,
            "E": 450,
            "F": 550,
            "G": 650,
        }

        def replace_value(x):
            for k, v in epc_map.items():
                if str(k) in str(x):
                    return v
            return -1

        X_copy = X.copy()
        X_copy['Consumption'] = X_copy['Consumption'].apply(lambda x: 700 if x > 700 else x)
        X_copy['Consumption'] = X_copy['Consumption'].apply(lambda x: 0 if x < 0 else x)

        X_copy['EPC'] = X_copy['EPC'].apply(replace_value)
        X_copy['Consumption'] = X_copy.apply(
            lambda row: epc_map[row['EPC']] if pd.notna(row['EPC']) and row['EPC'] in epc_map and pd.isna(
                row['Consumption']) else row['Consumption'],
            axis=1
        )
        return X_copy

    def inverse_transform(self, X):
        return X


class InverseScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler):
        self.scaler = scaler

    def fit(self, X, y=None):
        return self  # Nothing to do here

    def transform(self, X):
        # Assuming X is scaled between 0 and 1, invert the scaling
        return self.scaler.inverse_transform(X)


class MyStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        print('Fitting Standard Scaler')
        self.scaler.fit(X)
        return self

    def transform(self, X):
        print('Transforming with Standard Scaler')
        X_copy = X.copy()
        X_copy = self.scaler.transform(X_copy)
        return X_copy

    def inverse_transform(self, X):
        X_copy = X.copy()
        X_copy = self.scaler.inverse_transform(X_copy)
        return X_copy





class CatBoostRegressorWrapper(BaseEstimator, RegressorMixin):
    """
    CatbootRegressor's fit function accepts parameters, but when you use CatBoostRegressor in a pipeline,
    you can't pass these parameters to the fit function. So when you create a CatBoostRegressorWrapper,
    you can pass the parameters to the fit_params parameter of the constructor, and they will be passed to
    the fit function.
    """
    def __init__(self, fit_params: dict = None, **kwargs):
        self.fit_params = fit_params if fit_params else {}
        self.model = CatBoostRegressor(**kwargs)

    def fit(self, X, y=None):
        self.model.fit(X, y, **self.fit_params)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def transform(self, X, y=None):
        return X

    def inverse_transform(self, X):
        return X
class NaNToCategoryTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that replaces NaN values in specified non-numerical columns
    (or all non-numerical columns if none are specified) with a specified value,
    defaulting to 'missing'.
    """
    def __init__(self, replacement='missing', columns=None):
        self.replacement = replacement
        self.columns_to_impute = columns

    def fit(self, X, y=None):
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        # If no columns are specified, use all non-numerical columns
        if not self.columns_to_impute:
            self.columns_to_impute = categorical_columns
            return self

        # Check if all specified columns are categorical
        for col in self.columns_to_impute:
            if col not in categorical_columns:
                raise ValueError(f"Column '{col}' is not a categorical column")

        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.columns_to_impute] = X_copy[self.columns_to_impute].fillna(self.replacement)
        return X_copy

    def inverse_transform(self, X):
        return X

