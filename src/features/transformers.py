import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler


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
        return X.reset_index(drop=self.drop)

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
        return X

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
    def __init__(self, columns, multipliers=None):
        multipliers = multipliers if multipliers else {}
        for key in multipliers.keys():
            if key not in columns:
                raise ValueError(f"The Multiplier {key} was not in columns")
        self.column_scalers = {}
        for column in columns:
            multiplier = multipliers[column] if column in multipliers else 1
            self.column_scalers[column] = MinMaxScaler(feature_range=(0, multiplier))

    def fit(self, X, y=None):
        for column, scaler in self.column_scalers.items():
            self.column_scalers[column].fit(X[[column]])
        self.column_names = X.columns
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        for column, scaler in self.column_scalers.items():
            X_copy[column] = scaler.transform(X_copy[[column]])
        return X_copy

    def inverse_transform(self, df_scaled):
        df_scaled = df_scaled.copy()
        for column, scaler in self.column_scalers.items():
            df_scaled[column] = scaler.inverse_transform(df_scaled[[column]])

        return df_scaled


class MyKNNImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, n_neighbors=5, **kwargs):
        self.columns = columns
        self.n_neighbors = n_neighbors
        self.imputer = KNNImputer(n_neighbors=n_neighbors, **kwargs)
        self.kwargs = kwargs

    def fit(self, X, y=None):
        # Ensure X is a DataFrame for column indexing
        X = pd.DataFrame(X)
        self.imputer.fit(X[self.columns])
        return self

    def transform(self, X):
        # Ensure X is a DataFrame for column indexing
        X = pd.DataFrame(X)
        # Impute the specified columns
        X[self.columns] = self.imputer.transform(X[self.columns])
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

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
