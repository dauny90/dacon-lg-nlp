
import yaml
import os
import random
import numpy as np
import pickle
import json

CONFIG_PATH = "../config/"

def load_config(config_name):
    '''
    Cargar config file.
    '''
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

def seed_everything(seed: int = 199030):
    '''
    Definir seed para replicar.
    '''
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def read_file(dir, file):
    '''
    lectura de input de los archivos entregados por DACON
    '''
    source = os.path.join(dir, file)
    with open(source) as f:
        source_data = json.loads(f.read())

    return source_data

def save_file(dir, filepath, file):
    '''
    save archivos intermedios
    '''
    source = os.path.join(dir, filepath)
    with open(source, 'wb') as handle:
        pickle.dump(file, handle)

def load_file(dir, file):
    '''
    lectura de objetos guardados previamente
    '''
    source = os.path.join(dir, file)
    with open(source, 'rb') as handle:
        load_object = pickle.load(handle)

    return load_object