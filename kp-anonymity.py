import os
import numpy as np
import pandas as pd
import sys
import random
from loguru import logger
from node import Node
from pathlib import Path
from dataset_anonymized import DatasetAnonymized

max_level = 5 #caputo: 4
output_path = "Dataset/output.csv"


def clean_data(dataset_path_to_clean):
    """
    Print on file the dataset cleaned, in this case remove all columns normalized
    :param dataset_path_to_clean:
    :return:
    """
    time_series = pd.read_csv(dataset_path)
    time_series = time_series.loc[0:len(time_series), "Product_Code":"W51"]
    time_series.to_csv(dataset_path_to_clean.replace(".csv", "_Final.csv"), index=False)


# TODO check if |Ai| should be calculate on original table or not
def compute_normalized_certainty_penalty_on_ai(table=None, maximum_value=None, minimum_value=None):
    """
    Compute NCP(T)
    :param table:
    :return:
    """
    z_1 = list()
    y_1 = list()
    a = list()
    for index_attribute in range(0, len(table[0])): 
        temp_z1 = 0
        temp_y1 = float('inf') 
        for row in table: 
            if row[index_attribute] > temp_z1:
                temp_z1 = row[index_attribute]
            if row[index_attribute] < temp_y1:
                temp_y1 = row[index_attribute]
        z_1.append(temp_z1) 
        y_1.append(temp_y1) 
        a.append(abs(maximum_value[index_attribute] - minimum_value[index_attribute]))
    ncp_t = 0
    for index in range(0, len(z_1)):
        try:
            ncp_t += (z_1[index] - y_1[index]) / a[index]
        except ZeroDivisionError:
            ncp_t += 0
    ncp_T = len(table)*ncp_t 
    return ncp_T

def find_tuple_with_maximum_vl(fixed_tuple, time_series, key_fixed_tuple):
    """
    By scanning all tuples once, we can find tuple t1 that maximizes NCP(fixed_tuple, t1)
    :param fixed_tuple:
    :param time_series:
    :param key_fixed_tuple:
    :return:
    
    max_value = 0
    tuple_with_max_ncp = None
    for key, value in time_series.items():
        if key != key_fixed_tuple:
            ncp = compute_normalized_certainty_penalty_on_ai([fixed_tuple, time_series[key]], maximum_value, minimum_value)
            if ncp >= max_value:
                tuple_with_max_ncp = key
                max_value = ncp
    logger.info("Max ncp found: {} with tuple {} ".format(max_value, tuple_with_max_ncp))           
    return tuple_with_max_ncp"""
    pass


def find_tuple_with_maximum_ncp(fixed_tuple, time_series, key_fixed_tuple, maximum_value, minimum_value):
    """
    By scanning all tuples once, we can find tuple t1 that maximizes NCP(fixed_tuple, t1)
    :param fixed_tuple:
    :param time_series:
    :param key_fixed_tuple:
    :return:
    """
    max_value = 0
    tuple_with_max_ncp = None
    for key, value in time_series.items():
        if key != key_fixed_tuple:
            ncp = compute_normalized_certainty_penalty_on_ai([fixed_tuple, time_series[key]], maximum_value, minimum_value)
            if ncp >= max_value:
                tuple_with_max_ncp = key
                max_value = ncp
    logger.info("Max ncp found: {} with tuple {} ".format(max_value, tuple_with_max_ncp))           
    return tuple_with_max_ncp


def top_down_greedy_clustering(algorithm="naive", time_series=None, partition_size=None, maximum_value=None,
                                  minimum_value=None, time_series_clustered=None):
    """
    k-anonymity based on work of Xu et al. 2006,
    Utility-Based Anonymization for Privacy Preservation with Less Information Loss
    :param time_series:
    :param k_value:
    :return:
    """
    # len(time_series) < 2*k_value
    if len(time_series) <= partition_size:
        logger.info("End Recursion")
        time_series_clustered.append(time_series)
        return
    else:
        # TODO compute max and minumum_value for each recursive methods
        # partition time_series into two exclusive subsets time_series_1 and time_series_2
        # such that time_series_1 and time_series_2 are more local than time_series,
        # and either time_series_1 or time_series_2 have at least k tuples
        logger.info("Start Partition with size {}".format(len(time_series)))
        keys = list(time_series.keys())
        rounds = 3

        # pick random tuple
        random_tuple = keys[random.randint(0, len(keys) - 1)] 
        logger.info("Get random tuple (u1) {}".format(random_tuple))
        group_u = dict()
        group_v = dict()
        group_u[random_tuple] = time_series[random_tuple] 
        #del time_series[random_tuple]
        last_row = random_tuple
        for round in range(0, rounds*2 - 1): 
            if len(time_series) > 0:
                if round % 2 == 0:
                    if algorithm == "naive":
                        v = find_tuple_with_maximum_ncp(group_u[last_row], time_series, last_row, maximum_value, minimum_value)
                        logger.info("{} round: Find tuple (v) that has max ncp {}".format(round +1,v))
                    if algorithm == "kapra":
                        v = find_tuple_with_maximum_vl(group_u[last_row], time_series, last_row)
                        logger.info("{} round: Find tuple (v) that has max vl {}".format(round +1,v))

                    group_v.clear()
                    group_v[v] = time_series[v]
                    last_row = v
                    #del time_series[v]
                else:
                    if algorithm == "naive":
                        u = find_tuple_with_maximum_ncp(group_v[last_row], time_series, last_row, maximum_value, minimum_value)
                        logger.info("{} round: Find tuple (u) that has max ncp {}".format(round+1, u))
                    if algorithm == "kapra":
                        u = find_tuple_with_maximum_vl(group_v[last_row], time_series, last_row)
                        logger.info("{} round: Find tuple (u) that has max ncp {}".format(round+1, u))
                    
                    group_u.clear()
                    group_u[u] = time_series[u]
                    last_row = u
                    #del time_series[u]

        # Now Assigned to group with lower uncertain penality
        index_keys_time_series = [x for x in range(0, len(list(time_series.keys())))]
        random.shuffle(index_keys_time_series)
        # add random row to group with lower NCP
        keys = [list(time_series.keys())[x] for x in index_keys_time_series] 
        for key in keys:
            row_temp = time_series[key]
            group_u_values = list(group_u.values())
            group_v_values = list(group_v.values())
            group_u_values.append(row_temp)
            group_v_values.append(row_temp)

            if algorithm == "naive":
                ncp_u = compute_normalized_certainty_penalty_on_ai(group_u_values, maximum_value, minimum_value)
                ncp_v = compute_normalized_certainty_penalty_on_ai(group_v_values, maximum_value, minimum_value)

                if ncp_v < ncp_u:
                    group_v[key] = row_temp
                else:
                    group_u[key] = row_temp
                del time_series[key]

            if algorithm == "kapra":
                #devo fare lo stesso di sopra 
                pass

        logger.info("Group u: {}, Group v: {}".format(len(group_u), len(group_v)))
        if len(group_u) > partition_size:
            # recursive partition group_u
            # maximum_value, minimum_value = get_list_min_and_max_from_table(list(group_u.values()))
            top_down_greedy_clustering(time_series=group_u, partition_size=partition_size,
                                          maximum_value=maximum_value, minimum_value=minimum_value,
                                          time_series_clustered=time_series_clustered)
        else:
            time_series_clustered.append(group_u)

        if len(group_v) > partition_size:
            # recursive partition group_v

            # maximum_value, minimum_value = get_list_min_and_max_from_table(list(group_v.values()))
            top_down_greedy_clustering(time_series=group_v, partition_size=partition_size,
                                          maximum_value=maximum_value, minimum_value=minimum_value,
                                          time_series_clustered=time_series_clustered)
        else:
            time_series_clustered.append(group_v)


def get_list_min_and_max_from_table(dict_table): #input un dizionario, output la lista di max e min per ogni attributo (l'envelope)
    attributes_maximum_value = list()
    attributes_minimum_value = list()

    time_series = pd.DataFrame.from_dict(dict_table, orient="index")
    columns = list(time_series.columns)
    for column in columns:
        attributes_maximum_value.append(time_series[column].max())
        attributes_minimum_value.append(time_series[column].min())
    return attributes_minimum_value, attributes_maximum_value


def main_naive(k_value=None, p_value=None, paa_value=None, dataset_path=None):
    """
    k-P anonymity based on work of Shou et al. 2011,
    Supporting Pattern-Preserving Anonymization for Time-Series Data
    implementation of Naive approach
    :param k_value:
    :param p_value:
    :param dataset_path:
    :return:
    """

    if dataset_path.is_file():
        # read time_series_from_file
        time_series = pd.read_csv(dataset_path)

        # get columns name
        columns = list(time_series.columns)
        time_series_index = columns.pop(0)  # remove product code

        time_series_dict = dict()
        
        # save dict file instead pandas
        for index, row in time_series.iterrows():
            time_series_dict[row[time_series_index]] = list(row[columns])

        # save all maximum value for each attribute
        attributes_minimum_value, attributes_maximum_value = get_list_min_and_max_from_table(time_series_dict)


        # start k_anonymity_top_down
        time_series_k_anonymized = list()
        time_series_dict_copy = time_series_dict.copy()
        logger.info("Start k-anonymity top down approach")
        top_down_greedy_clustering(algorithm="naive", time_series=time_series_dict_copy, partition_size=k_value,
                                      maximum_value=attributes_maximum_value, minimum_value=attributes_minimum_value,
                                      time_series_clustered=time_series_k_anonymized)
        logger.info("End k-anonymity top down approach")

        # start kp anonymity
        dataset_anonymized = DatasetAnonymized()
        for group in time_series_k_anonymized:
            # append group to anonymized_data (after we will create a complete dataset anonymized)
            dataset_anonymized.anonymized_data.append(group) #TODO metto un k-group dentro a questa struttura
            # good leaf nodes
            good_leaf_nodes = list()
            bad_leaf_nodes = list()
            # creation root and start splitting node
            logger.info("Start Splitting node")
            node = Node(level=1, group=group, paa_value=paa_value)
            node.start_splitting(p_value, max_level, good_leaf_nodes, bad_leaf_nodes) # NOTE il nodo inizia la split node
            logger.info("Finish Splitting node")

            logger.info("Start postprocessing node merge all bad leaf node (if exists) in good " # NOTE : post processing dei bad leaf
                        "leaf node with most similar patter")
            for x in good_leaf_nodes:
                logger.info("Good leaf node {}, {}".format(x.size, x.pattern_representation))
            for x in bad_leaf_nodes:
                logger.info("Bad leaf node {}".format(x.size))
            if len(bad_leaf_nodes) > 0:
                logger.info("Add bad node {} to good node, start postprocessing".format(len(bad_leaf_nodes)))
                Node.postprocessing(good_leaf_nodes, bad_leaf_nodes)
                for x in good_leaf_nodes:
                    logger.info("Now Good leaf node {}, {}".format(x.size, x.pattern_representation))

            dataset_anonymized.pattern_anonymized_data.append(good_leaf_nodes) #TODO pattern_anonymized_data è una lista che contiene le liste dei good_leaf_nodes
        dataset_anonymized.compute_anonymized_data() # NOTE cosa fa? sembra mettere tutto insieme..
        dataset_anonymized.save_on_file(Path(output_path))

def main_kapra(k_value=None, p_value=None, paa_value=None, dataset_path=None):
    """
    k-P anonymity based on work of Shou et al. 2011,
    Supporting Pattern-Preserving Anonymization for Time-Series Data
    implementation of KAPRA approach
    :param k_value:
    :param p_value:
    :param dataset_path:
    :return:
    """

    if dataset_path.is_file():
        # read time_series_from_file
        time_series = pd.read_csv(dataset_path)

        # get columns name
        columns = list(time_series.columns)
        time_series_index = columns.pop(0)  # remove product code
        
        # save all maximum value for each attribute
        attributes_maximum_value = list()
        attributes_minimum_value = list()
        for column in columns:
            attributes_maximum_value.append(time_series[column].max())
            attributes_minimum_value.append(time_series[column].min())
        
        time_series_dict = dict()
        # save dict file instead pandas
        for index, row in time_series.iterrows():
            time_series_dict[row[time_series_index]] = list(row[columns])

        #NOTE fino a qua lo tengo uguale: ho i min e max per ogni attributo, e il time_series_dict

        # create-tree phase
        """
        In questa fase mettiamo nel nodo radice l'intero dataset T, e lo splittiamo.
        Viene eliminata la fase di postprocessing: tutte le good-leaf sono salvate in una leaf-list,
        mentre le bad-leaf passano alla recycle bad-leaves phase.
        """
        good_leaf_nodes = list()
        bad_leaf_nodes = list()

        # creation root and start splitting node
        logger.info("Start Splitting Dataset")
        node = Node(level=1, group=time_series_dict, paa_value=paa_value)
        node.start_splitting(p_value, max_level, good_leaf_nodes, bad_leaf_nodes) # NOTE il nodo inizia la split node
        logger.info("Finish Splitting Dataset")

        # recycle bad-leaves phase
        suppressed_nodes = list()
        if(len(bad_leaf_nodes) > 0):
            Node.recycle_bad_leaves(p_value, good_leaf_nodes, bad_leaf_nodes, suppressed_nodes, paa_value)


        # TODO group formation phase

        # preprocessing
        p_group_list = list() # è una lista di dizionari
        for node in good_leaf_nodes:
            p_group_list.append(node.group)

        p_group_list_copy = p_group_list.copy()

        for index, p_group in enumerate(p_group_list_copy):
            if len(p_group) >= 2*p_value:
                p_group_splitted = list()
                p_group_to_split = p_group_list.pop(index)
                #ma x e min da ricalcolare

                # start top down greedy clustering
                # non serve max e min perchè non ci servono parametri globali sugli attributi come in naive
                top_down_greedy_clustering(algorithm="kapra", time_series=p_group_to_split, partition_size=p_value, 
                                      time_series_clustered=p_group_splitted)


        # start k_anonymity_top_down
        time_series_k_anonymized = list()
        time_series_dict_copy = time_series_dict.copy()
        logger.info("Start k-anonymity top down approach")
        top_down_greedy_clustering(time_series=time_series_dict_copy, partition_size=k_value,
                                      maximum_value=attributes_maximum_value, minimum_value=attributes_minimum_value,
                                      time_series_clustered=time_series_k_anonymized)
        logger.info("End k-anonymity top down approach")

        # start kp anonymity
        # print(list(time_series_k_anonymized[0].values()))

        dataset_anonymized = DatasetAnonymized()
        for group in time_series_k_anonymized:
            # append group to anonymized_data (after we will create a complete dataset anonymized)
            dataset_anonymized.anonymized_data.append(group) #NOTE metto un k-group dentro a questa struttura
            # good leaf nodes
            good_leaf_nodes = list()
            bad_leaf_nodes = list()
            # creation root and start splitting node
            logger.info("Start Splitting node")
            node = Node(level=1, group=group, paa_value=paa_value)
            node.start_splitting(p_value, max_level, good_leaf_nodes, bad_leaf_nodes) # NOTE il nodo inizia la split node
            logger.info("Finish Splitting node")

            logger.info("Start postprocessing node merge all bad leaf node (if exists) in good " # NOTE : post processing dei bad leaf
                        "leaf node with most similar patter")
            for x in good_leaf_nodes:
                logger.info("Good leaf node {}, {}".format(x.size, x.pattern_representation))
            for x in bad_leaf_nodes:
                logger.info("Bad leaf node {}".format(x.size))
            if len(bad_leaf_nodes) > 0:
                logger.info("Add bad node {} to good node, start postprocessing".format(len(bad_leaf_nodes)))
                Node.postprocessing(good_leaf_nodes, bad_leaf_nodes)
                for x in good_leaf_nodes:
                    logger.info("Now Good leaf node {}, {}".format(x.size, x.pattern_representation))

            dataset_anonymized.pattern_anonymized_data.append(good_leaf_nodes)
        dataset_anonymized.compute_anonymized_data() # NOTE cosa fa? sembra mettere tutto insieme..
        dataset_anonymized.save_on_file(Path(output_path))


if __name__ == "__main__":

    if len(sys.argv) == 6:
        algorithm = sys.argv[1] #NOTE naive o kapra
        k_value = int(sys.argv[2])
        p_value = int(sys.argv[3])
        paa_value = int(sys.argv[4])
        dataset_path = sys.argv[5] #NOTE la gestione del path non va bene, deve essere assoluta rispetto a unix/win
        if k_value < p_value:
            print("[*] Usage: python kp-anonymity.py algorithm k_value p_value paa_value dataset.csv")
            print("[*] k_value should be greater than p_value")
        elif algorithm == "naive":
            main_naive(k_value=k_value, p_value=p_value, paa_value=paa_value, dataset_path=Path(dataset_path))
        elif algorithm == "kapra":    
            main_kapra(k_value=k_value, p_value=p_value, paa_value=paa_value, dataset_path=Path(dataset_path))
        else:
            print("[*] Usage: python kp-anonymity.py algorithm k_value p_value paa_value dataset.csv")
            print("[*] Algorithm supported: naive, kapra")
            
    else:
        print("[*] Usage: python kp-anonymity.py algorithm k_value p_value paa_value dataset.csv")