import os
import numpy as np
import pandas as pd
import sys
import random
from loguru import logger
from node import Node
from pathlib import Path
from dataset_anonymized import DatasetAnonymized

# caputo : 4
max_level = 5


def clean_data(dataset_path_to_clean):
    """
    Print on file the dataset cleaned, in this case remove all columns normalized
    :param dataset_path_to_clean:
    :return:
    """
    time_series = pd.read_csv(dataset_path)
    time_series = time_series.loc[0:len(time_series), "Product_Code":"W51"]
    time_series.to_csv(dataset_path_to_clean.replace(".csv", "_Final.csv"), index=False)


def compute_normalized_certainty_penalty_on_ai(table=None, maximum_value=None, minimum_value=None):
    """
    Compute NCP(T)
    :param table:
    :return:
    """
    z_1 = list()
    y_1 = list()
    a = list()
    n = len(table[0])

    for index_attribute in range(0, n): 
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
    for index in range(0, n):
        if a[index] == 0:
            ncp_t += 0
        else:
            ncp_t += (z_1[index] - y_1[index]) / a[index]
    ncp_T = len(table)*ncp_t 
    return ncp_T

def compute_instant_value_loss(table):
    """
    Compute VL(T)
    :param table:
    :return: [description]
    """ 
    r_plus = list()
    r_minus = list()
    n = len(table[0])

    for index_attribute in range(0, n): 
        temp_r_plus = 0
        temp_r_minus = float('inf')
        for row in table:
            if row[index_attribute] > temp_r_plus:
                temp_r_plus = row[index_attribute]
            if row[index_attribute] < temp_r_minus:
                temp_r_minus = row[index_attribute]
        r_plus.append(temp_r_plus) 
        r_minus.append(temp_r_minus)
    
    vl_t = 0
    for index in range(0, n):
        vl_t += pow((r_plus[index] - r_minus[index]), 2) / n
    vl_t = np.sqrt(vl_t)
    vl_T = len(table)*vl_t
    return vl_T

def find_tuple_with_maximum_vl(fixed_tuple, time_series, key_fixed_tuple):
    """
    By scanning all tuples once, we can find tuple t1 that maximizes VL(fixed_tuple, t1)
    :param fixed_tuple:
    :param time_series:
    :param key_fixed_tuple:
    :return:
    """
    max_value = 0
    tuple_with_max_vl = None
    for key, value in time_series.items():
        if key != key_fixed_tuple:
            vl = compute_instant_value_loss([fixed_tuple, time_series[key]])
            if vl >= max_value:
                tuple_with_max_vl = key
                max_value = vl
    #logger.info("Max vl found: {} with tuple {} ".format(max_value, tuple_with_max_vl))           
    return tuple_with_max_vl
    


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
    #logger.info("Max ncp found: {} with tuple {} ".format(max_value, tuple_with_max_ncp))           
    return tuple_with_max_ncp


def top_down_greedy_clustering(algorithm=None, time_series=None, partition_size=None, maximum_value=None,
                                  minimum_value=None, time_series_clustered=None, tree_structure=None, group_label="o"):
    """
    k-anonymity based on work of Xu et al. 2006,
    Utility-Based Anonymization for Privacy Preservation with Less Information Loss
    :param time_series:
    :param k_value:
    :return:
    """
    if len(time_series) < 2*partition_size:
        #logger.info("End Recursion")
        time_series_clustered.append(time_series)
        tree_structure.append(group_label)
        return
    else:
        #logger.info("Start Partition with size {}".format(len(time_series)))
        keys = list(time_series.keys())
        rounds = 3

        # pick random tuple
        random_tuple = keys[random.randint(0, len(keys) - 1)] 
        #logger.info("Get random tuple (u1) {}".format(random_tuple))
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
                        #logger.info("{} round: Find tuple (v) that has max ncp {}".format(round +1,v))
                    if algorithm == "kapra":
                        v = find_tuple_with_maximum_vl(group_u[last_row], time_series, last_row)
                        #logger.info("{} round: Find tuple (v) that has max vl {}".format(round +1,v))

                    group_v.clear()
                    group_v[v] = time_series[v]
                    last_row = v
                    #del time_series[v]
                else:
                    if algorithm == "naive":
                        u = find_tuple_with_maximum_ncp(group_v[last_row], time_series, last_row, maximum_value, minimum_value)
                        #logger.info("{} round: Find tuple (u) that has max ncp {}".format(round+1, u))
                    if algorithm == "kapra":
                        u = find_tuple_with_maximum_vl(group_v[last_row], time_series, last_row)
                        #logger.info("{} round: Find tuple (u) that has max vl {}".format(round+1, u))
                    
                    group_u.clear()
                    group_u[u] = time_series[u]
                    last_row = u
                    #del time_series[u]

        # Now Assigned to group with lower uncertain penality
        index_keys_time_series = [index for (index, key) in enumerate(time_series) if key not in [u, v]]
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
                vl_u = compute_instant_value_loss(group_u_values)
                vl_v = compute_instant_value_loss(group_v_values)

                if vl_v < vl_u:
                    group_v[key] = row_temp
                else:
                    group_u[key] = row_temp
                del time_series[key]


        #logger.info("Group u: {}, Group v: {}".format(len(group_u), len(group_v)))
        if len(group_u) > partition_size:
            top_down_greedy_clustering(algorithm=algorithm, time_series=group_u, partition_size=partition_size,
                                          maximum_value=maximum_value, minimum_value=minimum_value,
                                          time_series_clustered=time_series_clustered, tree_structure=tree_structure, 
                                          group_label=group_label+"a") # label : to track node position
        else:
            time_series_clustered.append(group_u)
            tree_structure.append(group_label)

        if len(group_v) > partition_size:
            top_down_greedy_clustering(algorithm=algorithm, time_series=group_v, partition_size=partition_size,
                                          maximum_value=maximum_value, minimum_value=minimum_value,
                                          time_series_clustered=time_series_clustered, tree_structure=tree_structure, 
                                          group_label=group_label+"b") # label : to track node position
        else:
            time_series_clustered.append(group_v)
            tree_structure.append(group_label)


def top_down_greedy_clustering_postprocessing(algorithm="naive", time_series_clustered=None, tree_structure=None, 
                                              partition_size=None, maximum_value=None, minimum_value=None,
                                              time_series_postprocessed=None):
    """
    Postprocessing to adjust group with size < partition_size

    :param algorithm: [description], defaults to "naive"
    :param time_series_clustered: [description], defaults to None
    :param tree_structure: [description], defaults to None
    :param partition_size: [description], defaults to None
    :param maximum_value: [description], defaults to None
    :param minimum_value: [description], defaults to None
    :param time_series_postprocessed: [description], defaults to None
    """
    
    index_change = list()
    group_change = list()
    tree_structure_change = list()

    for index_group_1, g_group_1 in enumerate(time_series_clustered):
        g_size = len(g_group_1)
        if g_size < partition_size:
            g_group_1_values = list(g_group_1.values())
            group_label = tree_structure[index_group_1]
            index_neighbour = -1
            measure_neighbour = float('inf') 
            for index_label, label in enumerate(tree_structure): 
                    if label[:-1] == group_label[:-1]: 
                        if index_label != index_group_1: 
                            if index_label not in index_change:
                                index_neighbour = index_label
            
            if index_neighbour > 0:
                table_1 = g_group_1_values + list(time_series_clustered[index_neighbour].values())
                
                if algorithm == "naive":
                    measure_neighbour = compute_normalized_certainty_penalty_on_ai(table=table_1, maximum_value=maximum_value, minimum_value=minimum_value)
                if algorithm == "kapra":
                    measure_neighbour = compute_instant_value_loss(table=table_1)

                group_merge_neighbour = dict()
                group_merge_neighbour.update(g_group_1)
                group_merge_neighbour.update(time_series_clustered[index_neighbour])

            measure_other_group = float('inf')   

            for index, other_group in enumerate(time_series_clustered): 
                if len(other_group) >= 2*partition_size - g_size: #2k - |G|   
                    if index not in index_change:    
                        g_group_2 = g_group_1.copy()
                        for round in range(partition_size - g_size): #k - |G|         
                            round_measure = float('inf')
                            g_group_2_values = list(g_group_2.values())
                            for key, time_series in other_group.items(): 
                                if key not in g_group_2.keys(): 
                                
                                    if algorithm == "naive":
                                        temp_measure = compute_normalized_certainty_penalty_on_ai(table=g_group_2_values + [time_series], 
                                                                                            maximum_value=maximum_value, 
                                                                                            minimum_value=minimum_value)
                                    if algorithm == "kapra":
                                        temp_measure = compute_instant_value_loss(table=g_group_2_values + [time_series])


                                    if temp_measure < round_measure:
                                        round_measure = temp_measure #set new min
                                        dict_to_add = { key : time_series }
                            
                            g_group_2.update(dict_to_add)

                        if round_measure < measure_other_group: # last ncp : ncp of the group
                            measure_other_group = round_measure 
                            group_merge_other_group = g_group_2
                            group_merge_remain = {key: value for (key, value) in other_group.items() if key not in g_group_2.keys()} 
                            index_other_group = index

            if measure_neighbour < measure_other_group: 
                index_change.append(index_neighbour)
                group_change.append(group_merge_neighbour)
                tree_structure_change.append(tree_structure[index_neighbour][:-1]) 

            else:
                index_change.append(index_other_group)
                group_change.append(group_merge_other_group)
                group_change.append(group_merge_remain)
                tree_structure_change.append("") # empty label


            index_change.append(index_group_1)
    
    time_series_clustered = [group for (index, group) in enumerate(time_series_clustered) if index not in index_change ]
    time_series_clustered += group_change 

    tree_structure = [label for (index, label) in enumerate(tree_structure) if index not in index_change]
    tree_structure += tree_structure_change

    bad_group_count = 0
    for index, group in enumerate(time_series_clustered):
        if len(group) < partition_size:
            bad_group_count +=1

    time_series_postprocessed += time_series_clustered
    
    if bad_group_count > 0:
        top_down_greedy_clustering_postprocessing(algorithm=algorithm, time_series_clustered=time_series_postprocessed, 
                                                  tree_structure=tree_structure, partition_size=partition_size, 
                                                  maximum_value=maximum_value, minimum_value=minimum_value)
    
        
def get_list_min_and_max_from_table(dict_table):
    attributes_maximum_value = list()
    attributes_minimum_value = list()

    time_series = pd.DataFrame.from_dict(dict_table, orient="index")
    columns = list(time_series.columns)
    for column in columns:
        attributes_maximum_value.append(time_series[column].max())
        attributes_minimum_value.append(time_series[column].min())
    return attributes_minimum_value, attributes_maximum_value

def find_group_with_min_value_loss(group_to_search=None, group_to_merge=dict(), index_ignored=list()):
    min_p_group = {"group" : dict(), "index" : None, "vl" : float("inf")} 
    for index, group in enumerate(group_to_search):
        if index not in index_ignored: 
            vl = compute_instant_value_loss(list(group.values()) + list(group_to_merge.values()))
            if vl < min_p_group["vl"]:
                min_p_group["vl"] = vl
                min_p_group["group"] = group
                min_p_group["index"] = index
    return min_p_group["group"], min_p_group["index"]


def main_naive(k_value=None, p_value=None, paa_value=None, dataset_path=None, output_path=None):
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
        tree_structure = list() # for postprocessing
        time_series_dict_copy = time_series_dict.copy()
        logger.info("Start k-anonymity top down approach")

        top_down_greedy_clustering(algorithm="naive", time_series=time_series_dict_copy, partition_size=k_value,
                                      maximum_value=attributes_maximum_value, minimum_value=attributes_minimum_value,
                                      time_series_clustered=time_series_k_anonymized, tree_structure=tree_structure)
        logger.info("End k-anonymity top down approach")

        logger.info("Start postprocessing k-anonymity top down approach")   
        time_series_postprocessed = list()
        top_down_greedy_clustering_postprocessing(algorithm="naive", time_series_clustered=time_series_k_anonymized, 
                                                  tree_structure=tree_structure, partition_size=k_value, 
                                                  maximum_value=attributes_maximum_value, minimum_value=attributes_minimum_value,
                                                  time_series_postprocessed=time_series_postprocessed)
        logger.info("End postprocessing k-anonymity top down approach")

        time_series_k_anonymized = time_series_postprocessed

        # start kp anonymity
        pattern_representation_dict = dict()
        k_group_list = list()

        for group in time_series_k_anonymized:
            # append group to anonymized_data (after we will create a complete dataset anonymized)
            k_group_list.append(group) 
            # good leaf nodes
            good_leaf_nodes = list()
            bad_leaf_nodes = list()
            # creation root and start splitting node
            logger.info("Create-tree phase: initialization and start node splitting")
            node = Node(level=1, group=group, paa_value=paa_value)
            node.start_splitting(p_value, max_level, good_leaf_nodes, bad_leaf_nodes) 
            logger.info("Create-tree phase: finish node splitting")

            logger.info("Create-tree phase: start postprocessing")
            #for x in good_leaf_nodes:
            #   logger.info("Good leaf node {}, {}".format(x.size, x.pattern_representation))
            #for x in bad_leaf_nodes:
            #   logger.info("Bad leaf node {}".format(x.size))
            if len(bad_leaf_nodes) > 0:
            #    logger.info("Add bad node {} to good node, start postprocessing".format(len(bad_leaf_nodes)))
                Node.postprocessing(good_leaf_nodes, bad_leaf_nodes)
            #    for x in good_leaf_nodes:
            #        logger.info("Now Good leaf node {}, {}".format(x.size, x.pattern_representation))
            logger.info("Create-tree phase: finish postprocessing")
            for node in good_leaf_nodes:
                pr = node.pattern_representation
                for key in node.group:
                    pattern_representation_dict[key] = pr

        dataset_anonymized = DatasetAnonymized(pattern_anonymized_data=pattern_representation_dict,
                                               anonymized_data=k_group_list)
        dataset_anonymized.compute_anonymized_data()
        dataset_anonymized.save_on_file(output_path)


def main_kapra(k_value=None, p_value=None, paa_value=None, dataset_path=None, output_path=None):
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

        time_series_dict = dict()
        
        # save dict file instead pandas
        for index, row in time_series.iterrows():
            time_series_dict[row[time_series_index]] = list(row[columns])

        # create-tree phase
        good_leaf_nodes = list()
        bad_leaf_nodes = list()

        # creation root and start splitting node
        logger.info("Create-tree phase: initialization and start node splitting with entire dataset")
        node = Node(level=1, group=time_series_dict, paa_value=paa_value)
        node.start_splitting(p_value, max_level, good_leaf_nodes, bad_leaf_nodes)
        logger.info("Create-tree phase: finish node splitting")

        # recycle bad-leaves phase
        logger.info("Start recycle bad-leaves phase")
        suppressed_nodes = list()
        if(len(bad_leaf_nodes) > 0):
            Node.recycle_bad_leaves(p_value, good_leaf_nodes, bad_leaf_nodes, suppressed_nodes, paa_value)
        logger.info("Finish recycle bad-leaves phase")
        suppressed_nodes_list = list()
        for node in suppressed_nodes:
            suppressed_nodes_list.append(node.group) # suppressed nodes!!!
        
        # group formation phase
        # preprocessing
        logger.info("Start group formation phase")
        pattern_representation_dict = dict() 
        p_group_list = list() 
        for node in good_leaf_nodes: 
            p_group_list.append(node.group)
            pr = node.pattern_representation
            for key in node.group:
                pattern_representation_dict[key] = pr

        p_group_to_add = list()
        index_to_remove = list()

        for index, p_group in enumerate(p_group_list): 
            if len(p_group) >= 2*p_value:
                
                tree_structure = list()
                p_group_splitted = list()
                p_group_to_split = p_group 

                # start top down greedy clustering
                top_down_greedy_clustering(algorithm="kapra", time_series=p_group_to_split, partition_size=p_value, 
                                      time_series_clustered=p_group_splitted, tree_structure=tree_structure)

                #logger.info("Start postprocessing k-anonymity top down approach")
                time_series_postprocessed = list()
                top_down_greedy_clustering_postprocessing(algorithm="kapra", time_series_clustered=p_group_splitted, 
                                                          tree_structure=tree_structure, partition_size=p_value,
                                                          time_series_postprocessed=time_series_postprocessed)
                                                  
                #logger.info("End postprocessing k-anonymity top down approach")
                
                p_group_to_add += time_series_postprocessed
                index_to_remove.append(index)
        
        
        p_group_list = [group for (index, group) in enumerate(p_group_list) if index not in index_to_remove ]
        p_group_list += p_group_to_add
        
        
        k_group_list = list()
        index_to_remove = list() 
        
        # step 1
        for index, group in enumerate(p_group_list):
            if len(group) >= k_value:
                index_to_remove.append(index)
                k_group_list.append(group)
        
        p_group_list = [group for (index, group) in enumerate(p_group_list) if index not in index_to_remove ]

        # step 2 - 3 - 4
        index_to_remove = list()
        p_group_list_size = sum([len(group) for group in p_group_list])
        
        while p_group_list_size >= k_value:
            k_group, index_min = find_group_with_min_value_loss(group_to_search=p_group_list, 
                                                                index_ignored=index_to_remove)
            index_to_remove.append(index_min)
            p_group_list_size -= len(k_group)

            while len(k_group) < k_value:
                group_to_add, index_group_to_add = find_group_with_min_value_loss(group_to_search=p_group_list,
                                                                                  group_to_merge=k_group, 
                                                                                  index_ignored=index_to_remove)
                index_to_remove.append(index_group_to_add)
                k_group.update(group_to_add) 
                p_group_list_size -= len(group_to_add)
            k_group_list.append(k_group)   
        
        # step 5
        p_group_remaining = [group for (index, group) in enumerate(p_group_list) if index not in index_to_remove ]
        
        for p_group in p_group_remaining:
            k_group, index_k_group = find_group_with_min_value_loss(group_to_search=k_group_list,
                                                                    group_to_merge=p_group)
            k_group_list.pop(index_k_group)
            k_group.update(p_group)
            k_group_list.append(k_group)

        logger.info("Finish group formation phase")
        dataset_anonymized = DatasetAnonymized(pattern_anonymized_data=pattern_representation_dict,
                                               anonymized_data=k_group_list,
                                               suppressed_data=suppressed_nodes_list)
        dataset_anonymized.compute_anonymized_data()
        dataset_anonymized.save_on_file(output_path)


if __name__ == "__main__":

    if len(sys.argv) == 7:
        algorithm = sys.argv[1]
        k_value = int(sys.argv[2])
        p_value = int(sys.argv[3])
        paa_value = int(sys.argv[4])
        dataset_path = sys.argv[5]
        output_path = sys.argv[6]
        if k_value < p_value:
            print("[*] Usage: python kp-anonymity.py algorithm k_value p_value paa_value dataset_path dataset_output_path")
            print("[*] k_value should be greater than p_value")
        elif algorithm == "naive":
            main_naive(k_value=k_value, p_value=p_value, paa_value=paa_value, 
                       dataset_path=Path(dataset_path), output_path=Path(output_path))
        elif algorithm == "kapra":    
            main_kapra(k_value=k_value, p_value=p_value, paa_value=paa_value, 
                       dataset_path=Path(dataset_path), output_path=Path(output_path))
        else:
            print("[*] Usage: python kp-anonymity.py algorithm k_value p_value paa_value dataset_path dataset_output_path")
            print("[*] Algorithm supported: naive, kapra")
            
    else:
        print("[*] Usage: python kp-anonymity.py algorithm k_value p_value paa_value dataset_path dataset_output_path")