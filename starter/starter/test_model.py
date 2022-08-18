import os
import logging
import pandas as pd
import train_model

logging.basicConfig(
    filename='logs/model_testing.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import:
    - check if file exists in directory
    - check if file is empty
    - check if dataframe has rows and columns
    '''
    try:
        import_data("../data/census_clean.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

if __name__ == "__main__":
    test_import(train_model.import_data)
