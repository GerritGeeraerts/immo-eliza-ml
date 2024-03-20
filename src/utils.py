import pandas as pd
from matplotlib import pyplot as plt
from shapely import Point

from config import joined_data_path
import geopandas as gpd


def load_data(path=None):
    path = path if path else joined_data_path
    df = pd.read_csv(path, low_memory=False)
    return df


def visualize_metrics(r_squared, y_test, y_pred, comments=""):
    # Convert R-squared value to percentage
    r_squared_percent = round(r_squared * 100, 2)

    # Print the metrics
    print("Evaluation Metrics:")
    print("R-squared value:", f"{r_squared_percent:.2f}%")

    # Plot the metrics
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Plot R-squared value
    ax[0].bar(["R-squared value"], [r_squared_percent], color='green')
    ax[0].set_title(f"R-squared value: {r_squared_percent:.2f}%")
    ax[0].set_ylim([0, 100])  # Set y-axis limits to 0 and 100

    # Plot predicted vs actual values
    ax[1].scatter(y_test, y_pred, color='blue', alpha=0.5)
    ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Plot diagonal line
    ax[1].set_xlabel('Actual')
    ax[1].set_ylabel('Predicted')
    ax[1].set_title('Predicted vs Actual')

    if comments:
        plt.text(0.5, -0.1, f"Comments: {comments}", horizontalalignment='center', verticalalignment='center',
                 transform=ax[1].transAxes, bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.show()


def join_data(raw_data):
    """
    Join the raw data with external datasets.

    Parameters:
    raw_data (DataFrame): Raw data.

    Returns:
    DataFrame: Joined data.
    """
    # Load external datasets
    geo_data = pd.DataFrame(raw_data)
    pop_density_data = pd.read_excel('../data/external_data/PopDensity.xlsx', dtype={'Refnis': int})
    house_income_data = pd.read_excel('../data/external_data/HouseholdIncome.xlsx', dtype={'Refnis': int})
    property_value_data = pd.read_excel('../data/external_data/PropertyValue.xlsx', dtype={'Refnis': int})

    # Define a function to create Point objects from latitude and longitude
    def create_point(row):
        try:
            latitude = float(row['Latitude'])
            longitude = float(row['Longitude'])
            return Point(longitude, latitude)
        except ValueError:
            return None

    # Create Point geometries from latitude and longitude coordinates in real estate data
    geo_data['geometry'] = geo_data.apply(create_point, axis=1)

    # Load the raw data into a GeoDataFrame
    geo_data = gpd.GeoDataFrame(raw_data, geometry=geo_data['geometry'], crs='EPSG:4326')

    # Read only the necessary column 'cd_munty_refnis' from the municipality GeoJSON file
    municipality_gdf = gpd.read_file('../data/external_data/REFNIS_CODES.geojson', driver='GeoJSON')[
        ['cd_munty_refnis', 'geometry']].to_crs(epsg=4326)

    # Perform spatial join with municipality data
    joined_data = gpd.sjoin(geo_data, municipality_gdf, how='left', predicate='within')

    # Convert 'cd_munty_refnis' column to int type
    joined_data['cd_munty_refnis'] = joined_data['cd_munty_refnis'].fillna(-1).astype(int)

    # Data Merge
    datasets = [pop_density_data, property_value_data, house_income_data]
    for dataset in datasets:
        joined_data = joined_data.merge(dataset, left_on='cd_munty_refnis', right_on='Refnis', how='left')
        joined_data.drop(columns=['Refnis'], inplace=True)

    joined_data.to_csv('../data/intermediate/joined_data.csv', index=False)

    # Return the resulting DataFrame
    return joined_data


if __name__ == '__main__':
    df = load_data()
    df = join_data(df)
    print(df.head())
    print(df.columns)
