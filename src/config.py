import os.path

# get the current directory

raw_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'data.csv'))
base_path_models = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
