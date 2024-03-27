from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


class AddAvgRegionPrice(BaseEstimator, TransformerMixin):
    """
    Calculate the average price per square meter for each property in the dataset
    And add a column to each property with the average price per square meter of the region
    based on the closest neighbours defined in the neighbors attribute
    """

    def __init__(self, neighbors=25):
        """Store the number of neighbors to look at"""
        self.neighbors = neighbors
        self.values_in_column = None

    def fit(self, X, y=None, *args, **kwargs):
        """
        Calculate the average price per square meter for each property in the dataset
        and store it in a lookup table
        """
        if y is None:
            raise ValueError("y is required to fit the model")
        self.lookup = X.copy()
        self.lookup = self.lookup[['Longitude', 'Latitude', 'Habitable Surface']]
        self.lookup['Price'] = y.copy()
        self.lookup['PricePerSqm'] = self.lookup['Price'] / self.lookup['Habitable Surface']
        return self

    def get_closest_properties(self, longitude, latitude):
        """calculate the distance for each property to the input longitude and latitude and return the closest ones"""
        self.lookup['Distance'] = ((self.lookup['Longitude'] - longitude) ** 2 +
                                   (self.lookup['Latitude'] - latitude) ** 2)
        return self.lookup.sort_values('Distance').head(self.neighbors)

    def get_avg_price(self, longitude, latitude):
        """Get the average price per square meter of the closest properties to the input longitude and latitude"""
        closest_properties = self.get_closest_properties(longitude, latitude)
        return closest_properties['PricePerSqm'].mean()

    def transform(self, X):
        # print('column 5: ', X.columns[5])
        """Add the average price per square meter of the region to each property"""
        X_copy = X.copy()
        X_copy['RegionPricePerSqm'] = X_copy.apply(lambda x: self.get_avg_price(x['Longitude'], x['Latitude']), axis=1)
        return X_copy

    def inverse_transform(self, X):
        return X
