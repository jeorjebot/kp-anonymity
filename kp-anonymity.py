import os
import numpy as np
import pandas as pd
import sys
from loguru import logger
import random
from node import Node
from dataset_anonymized import DatasetAnonymized

max_level = 4 # TODO you can change


def clean_data(dataset_path_to_clean):
    """
        Print on file the dataset cleaned, in this case remove all columns normalized
    :param dataset_path_to_clean:
    :return:
    """
    time_series = pd.read_csv(dataset_path)
    time_series = time_series.loc[0:len(time_series), "Product_Code":"W51"]
    time_series.to_csv(dataset_path_to_clean.replace(".csv", "_Final.csv"), index=False)



def k_anonymity_top_down_approach(time_series=None, k_value=None, columns_list=None, maximum_value=None,
                                  time_series_k_anonymized=None):
    """
    k-anonymity based on work of Xu et al. 2006,
    Utility-Based Anonymization for Privacy Preservation with Less Information Loss
    :param time_series:
    :param k_value:
    :return:
    """
    """
        See Section 4.2 Utility-Based Anonymization for Privacy Preservation with Less Information Loss
    """
   
def compute_normalized_certainty_penalty_on_ai(table=None):
    """
    Compute NCP(T)
    :param table:
    :return:
    """

    """
        See Section 3.2.1 Utility-Based Anonymization for Privacy Preservation with Less Information Loss
    """
    


def main(k_value=None, p_value=None, paa_value=None, dataset_path=None):
    """

    :param k_value:
    :param p_value:
    :param dataset_path:
    :return:
    """
    if os.path.isfile(dataset_path):
        # read time_series_from_file
        time_series = pd.read_csv(dataset_path)
        
        time_series_dict = dict()
        # save dict file instead pandas
        for index, row in time_series.iterrows():
            time_series_dict[row["Product_Code"]] = list(row["W0":"W51"])

        time_series_k_anonymized = list()
        time_series_dict_copy = time_series_dict.copy()
        
        logger.info("Start k-anonymity top down approach")
        # TODO
        logger.info("End k-anonymity top down approach")

        logger.info("Start node splitting for each groups")
        # TODO
        logger.info("End node splitting for each groups")

        logger.info("Save dataset anonymized")
        # TODO
        
if __name__ == "__main__":

    if len(sys.argv) == 5:

        k_value = int(sys.argv[1])
        p_value = int(sys.argv[2])
        paa_value = int(sys.argv[3])
        dataset_path = sys.argv[4]
        if k_value > p_value:
            main(k_value=k_value, p_value=p_value, paa_value=paa_value, dataset_path=dataset_path)
        else:
            print("[*] Usage: python kp-anonymity.py k_value p_value paa_value dataset.csv")
            print("[*] k_value should be greater than p_value")
    else:
        print("[*] Usage: python kp-anonymity.py k_value p_value paa_value dataset.csv")