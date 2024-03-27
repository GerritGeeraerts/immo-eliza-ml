import os.path
import pickle

from config import base_path_models


def save_model_as_pickle(model, python_script_file_name: str):
    """
    Saves the given model as a pickle file to the specified file path.

    Parameters:
    - model: The model to be saved.
    - file_path: The path where the model will be saved, including the file name.
    """
    file_name = python_script_file_name.replace('.py', '').replace('train_', '')
    file_path = os.path.join(base_path_models, file_name + '.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(model, file,)
