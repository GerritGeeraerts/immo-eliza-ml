from sklearn.pipeline import Pipeline

from features.build_features import AddAvgRegionPrice
from features.transformers import RowFilter, DropNARows, ColumnSelector, ResetIndexTransformer, \
    FillLandSurfaceForApartment, FacadeFixer, ConsumptionFixer, MyOrdinalEncoder, MyMinMaxScaler, MyKNNImputer, \
    Log10Transformer

BASE_SUBTYPES_TO_KEEP = ['VILLA', 'HOUSE', 'APARTMENT', ]
BASE_COLUMNS_TO_KEEP = [
    'Bathroom Count', 'Bedroom Count', 'Habitable Surface', 'Land Surface', 'Consumption', 'Postal Code',
    'Facades', 'Subtype', 'Toilet Count', 'Kitchen Type', 'State of Building',  # 'Sea view', 'Swimming Pool',
    'Price', 'Longitude', 'Latitude', 'EPC',  # 'cd_munty_refnis', 'PopDensity', 'MedianPropertyValue',
    # 'NetIncomePerResident'
]
REQUIRED_COLUMNS = [
    'Habitable Surface', 'Land Surface', 'Price', 'Subtype', 'Bedroom Count', 'Postal Code', 'Bathroom Count',
]


def subtype_condition(df):
    """Filter the dataframe based on the subtype column"""
    return df['Subtype'].isin(BASE_SUBTYPES_TO_KEEP)


def sale_type_condition(df):
    """Filter the dataframe based on the sale type column"""
    return df['Sale Type'] == 'NORMAL_SALE'


kithcen_map = {
    "INSTALLED": 1,
    "HYPER_EQUIPPED": 3,
    "SEMI_EQUIPPED": 2,
    "USA_HYPER_EQUIPPED": 3,
    "NOT_INSTALLED": 0,
    "USA_INSTALLED": 1,
    "USA_SEMI_EQUIPPED": 2,
    "USA_UNINSTALLED": 0,
}
building_state_map = {
    "AS_NEW": 5,
    "JUST_RENOVATED": 4,
    "GOOD": 3,
    "TO_BE_DONE_UP": 2,
    "TO_RENOVATE": 1,
    "TO_RESTORE": 0,
}
subtype_map = {
    'VILLA': 3,
    'HOUSE': 2,
    'APARTMENT': 1,
}

pre_pipeline = Pipeline([
    ('Fix Facades < 1 and > 4', FacadeFixer()),
    ('row_filter', RowFilter(condition=subtype_condition)),
    ('row_filter', RowFilter(condition=sale_type_condition)),
    ('drop na rows for columns', DropNARows(columns=REQUIRED_COLUMNS)),
    ('Reset Index', ResetIndexTransformer()),
    ('ColumnSelector', ColumnSelector(keep_columns=BASE_COLUMNS_TO_KEEP)),
    ('Consumption fixer', ConsumptionFixer()),
    ('Drop EPC', ColumnSelector(drop_columns=['EPC'])),
    ('Fill Land Surface for APARTMENT', FillLandSurfaceForApartment()),
    ('Reset Index', ResetIndexTransformer()),
])

base_pipeline = Pipeline([
    ('Log Scale',
     Log10Transformer(columns=['Bathroom Count', 'Bedroom Count', 'Habitable Surface', 'Land Surface', 'Price'])),
])

base_after_split_pipeline_for_cat_boost = Pipeline([
    ('Min Max scaler', MyMinMaxScaler(columns=['Postal Code', 'Longitude', 'Latitude',])),
    ('KNN Lon, Lat', MyKNNImputer(columns=['Postal Code', 'Longitude', 'Latitude'])),
    ('Add RegionPricePerSqm', AddAvgRegionPrice(neighbors=25))
])

base_after_split_pipeline = Pipeline([
    ('Kitchen Type to Numeric', MyOrdinalEncoder('Kitchen Type', kithcen_map)),
    ('State of Building to Numeric', MyOrdinalEncoder('State of Building', building_state_map)),
    ('Subtype to numeric', MyOrdinalEncoder('Subtype', subtype_map)),
    ('Min Max scaler', MyMinMaxScaler(
        columns=['Land Surface', 'Habitable Surface', 'Bathroom Count', 'Toilet Count', 'Postal Code', 'Longitude',
                 'Latitude', 'Facades', 'Subtype', 'Consumption', 'State of Building', 'Kitchen Type', ],
        # 'cd_munty_refnis', 'PopDensity', 'MedianPropertyValue', 'NetIncomePerResident'],
        #multipliers={'Subtype': 100}  # make Subtype dominant for KNN
    )),
    ('KNN Toilets', MyKNNImputer(columns=['Habitable Surface', 'Bathroom Count', 'Toilet Count', 'Subtype'])),
    ('KNN Lon, Lat', MyKNNImputer(columns=['Postal Code', 'Longitude', 'Latitude'])),
    ('KNN Facade', MyKNNImputer(columns=['Facades', 'Land Surface', 'Habitable Surface', 'Subtype'])),
    ('KNN Consumption', MyKNNImputer(columns=['Consumption', 'State of Building', 'Kitchen Type', 'Subtype'])),
    # ('KNN REFNIS blanks', MyKNNImputer(
    #     columns=['Longitude', 'Latitude', 'cd_munty_refnis', 'PopDensity', 'MedianPropertyValue',
    #              'NetIncomePerResident']
    # )),
])
