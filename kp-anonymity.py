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
        try:
            ncp_t += (z_1[index] - y_1[index]) / a[index]
        except ZeroDivisionError:
            ncp_t += 0
    ncp_T = len(table)*ncp_t 
    return ncp_T

def compute_instant_value_loss(table): #table è una lista di liste, con dentro le time series
    r_plus = list()
    r_minus = list()
    n = len(table[0])

    for index_attribute in range(0, n): # NOTE da 0 a 51, per ogni attributo
        # NOTE vengono reinizializzati ogni iterazione
        temp_r_plus = 0
        temp_r_minus = float('inf') # NOTE : infinito
        for row in table:
            if row[index_attribute] > temp_r_plus:
                temp_r_plus = row[index_attribute]
            if row[index_attribute] < temp_r_minus:
                temp_r_minus = row[index_attribute]
        r_plus.append(temp_r_plus) # NOTE appendo il maggior z1 tra i due
        r_minus.append(temp_r_minus) # NOTE appendo il minor y1 tra i due
    
    vl_t = 0
    for index in range(0, n):
        vl_t += pow((r_plus[index] - r_minus[index]), 2) / n
    vl_t = np.sqrt(vl_t)
    vl_T = len(table)*vl_t
    return vl_T

def find_tuple_with_maximum_vl(fixed_tuple, time_series, key_fixed_tuple):
    """
    By scanning all tuples once, we can find tuple t1 that maximizes NCP(fixed_tuple, t1)
    :param fixed_tuple:
    :param time_series:
    :param key_fixed_tuple:
    :return:
    """
    max_value = 0
    tuple_with_max_vl = None
    for key, value in time_series.items():
        if key != key_fixed_tuple:
            vl = compute_instant_value_loss([fixed_tuple, time_series[key]]) #NOTE sono sempre due le tuple da confrontare.. sono quelle che danno origine alla partizione
            if vl >= max_value:
                tuple_with_max_vl = key
                max_value = vl
    logger.info("Max vl found: {} with tuple {} ".format(max_value, tuple_with_max_vl))           
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
    logger.info("Max ncp found: {} with tuple {} ".format(max_value, tuple_with_max_ncp))           
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
    # len(time_series) < 2*k_value
    if len(time_series) < 2*partition_size:
        logger.info("End Recursion")
        time_series_clustered.append(time_series)
        tree_structure.append(group_label)
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
        group_u[random_tuple] = time_series[random_tuple] # NOTE assegna al dizionario u, la key e relativi valori: "P13" : [...]
        #del time_series[random_tuple]
        last_row = random_tuple
        for round in range(0, rounds*2 - 1): 
            if len(time_series) > 0:#NOTE fanno una volta per uno
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
                        logger.info("{} round: Find tuple (u) that has max vl {}".format(round+1, u))
                    
                    group_u.clear()
                    group_u[u] = time_series[u]
                    last_row = u
                    #del time_series[u]

        # Now Assigned to group with lower uncertain penality
        #index_keys_time_series = [x for x in range(0, len(list(time_series.keys())))] #NOTE forse dovrei togliere group_u e group_v
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


        logger.info("Group u: {}, Group v: {}".format(len(group_u), len(group_v)))
        if len(group_u) > partition_size:
            # recursive partition group_u
            # maximum_value, minimum_value = get_list_min_and_max_from_table(list(group_u.values()))
            top_down_greedy_clustering(algorithm=algorithm, time_series=group_u, partition_size=partition_size,
                                          maximum_value=maximum_value, minimum_value=minimum_value,
                                          time_series_clustered=time_series_clustered, tree_structure=tree_structure, 
                                          group_label=group_label+"a") #NOTE tengo traccia della posizione del node in base alla codifica
        else:
            time_series_clustered.append(group_u)
            tree_structure.append(group_label)

        if len(group_v) > partition_size:
            # recursive partition group_v

            # maximum_value, minimum_value = get_list_min_and_max_from_table(list(group_v.values()))
            top_down_greedy_clustering(algorithm=algorithm, time_series=group_v, partition_size=partition_size,
                                          maximum_value=maximum_value, minimum_value=minimum_value,
                                          time_series_clustered=time_series_clustered, tree_structure=tree_structure, 
                                          group_label=group_label+"b")
        else:
            time_series_clustered.append(group_v)
            tree_structure.append(group_label)

        
def top_down_greedy_clustering_postprocessing(algorithm="naive", time_series_clustered=None, tree_structure=None, 
                                              partition_size=None, maximum_value=None, minimum_value=None):
    
    index_change = list()
    group_change = list()
    #time_series_clustered è una lista di dizionari
    
    for index_group_1, g_group_1 in enumerate(time_series_clustered):
        g_size = len(g_group_1)
        if g_size < partition_size: #NOTE allora è da processare, ha meno di k elementi

            g_group_1_values = list(g_group_1.values())

            group_label = tree_structure[index_group_1]
            index_neighbour = -1
            measure_neighbour = float('inf') 
            for index_label, label in enumerate(tree_structure): 
                    if label[:-1] == group_label[:-1]: # stesso pattern a parte la lettera finale
                        if index_label != index_group_1: # diverso da se stesso
                         
                            if index_label not in index_change: # se non è stato cambiato
                                index_neighbour = index_label
            
            if index_neighbour > 0:
                table_1 = g_group_1_values + list(time_series_clustered[index_neighbour].values())
                
                if algorithm == "naive":
                    measure_neighbour = compute_normalized_certainty_penalty_on_ai(table=table_1, maximum_value=maximum_value, minimum_value=minimum_value)
                if algorithm == "kapra":
                    measure_neighbour = compute_instant_value_loss(table=table_1)

                group_merge_neighbour = dict()
                group_merge_neighbour.update(g_group_1)
                group_merge_neighbour.update(time_series_clustered[index_neighbour]) #questo è il gruppo, se npc_neighbour fosse > dell'altro


            measure_other_group = float('inf')   

            for index, other_group in enumerate(time_series_clustered): #per ogni gruppo
                if len(other_group) >= 2*partition_size - g_size: #2k - |G|   
                    if index not in index_change:    
                        g_group_2 = g_group_1.copy()
                        for round in range(partition_size - g_size):#k-|G| volte
                            
                            round_measure = float('inf')
                            g_group_2_values = list(g_group_2.values())

                            for key, time_series in other_group.items(): #aggiunge una ts alla volta
                                
                                if key not in g_group_2.keys(): # non permette di aggiungere ts già aggiunte
                                    
                                    #NOTE ncp
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

                        if round_measure < measure_other_group: # è l'ultimo, quindi è ncp dell'intero gruppo "mergiato"
                            measure_other_group = round_measure #aggiorno ncp other group
                            group_merge_other_group = g_group_2
                            group_merge_remain = {key: value for (key, value) in other_group.items() if key not in g_group_2.keys()} # dict complementare a quello sopra, aggiunge le cose rimaste
                            index_other_group = index

            if measure_neighbour < measure_other_group: # aggiungi neighbour
                index_change.append(index_neighbour)
                group_change.append(group_merge_neighbour)
            else:
                index_change.append(index_other_group)
                group_change.append(group_merge_other_group)
                group_change.append(group_merge_remain)

            index_change.append(index_group_1) #anche questo è da rimuovere!!!

            # TODO ricorda di segnare quale "struttura" hai modificato, 
            # segnalare i nodi modificati in index_change, e i nodi da aggiungere in group_change
    
    time_series_clustered = [group for (index, group) in enumerate(time_series_clustered) if index not in index_change ]
    time_series_clustered += group_change #sopra tolgo i modificati, sotto aggiungo i modificati

    #TODO modifica del gruppo: tolgo i gruppi con indice index_change e aggungo al gruppo i nodi in group_change       

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
        tree_structure = list() #NOTE serve per il postprocessing
        time_series_dict_copy = time_series_dict.copy()
        logger.info("Start k-anonymity top down approach")

        top_down_greedy_clustering(algorithm="naive", time_series=time_series_dict_copy, partition_size=k_value,
                                      maximum_value=attributes_maximum_value, minimum_value=attributes_minimum_value,
                                      time_series_clustered=time_series_k_anonymized, tree_structure=tree_structure)
        logger.info("End k-anonymity top down approach")

        logger.info("Start postprocessing k-anonymity top down approach")
        top_down_greedy_clustering_postprocessing(algorithm="naive", time_series_clustered=time_series_k_anonymized, 
                                                  tree_structure=tree_structure, partition_size=k_value, 
                                                  maximum_value=attributes_maximum_value, minimum_value=attributes_minimum_value)
        logger.info("End postprocessing k-anonymity top down approach")

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

        time_series_dict = dict()
        
        # save dict file instead pandas
        for index, row in time_series.iterrows():
            time_series_dict[row[time_series_index]] = list(row[columns])

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

        p_group_to_add = list()
        index_to_remove = list()

        #NOTE qui ho 810 ts invece che 811 perchè una è stata soppressa!!!!!
        
        for index, p_group in enumerate(p_group_list):
            if len(p_group) >= 2*p_value: #quindi devo splittarlo
                
                tree_structure = list() #NOTE serve per il postprocessing
                p_group_splitted = list()
                p_group_to_split = p_group # è un dizionario NOTE fa casino con gli indici questo pop

                # start top down greedy clustering
                top_down_greedy_clustering(algorithm="kapra", time_series=p_group_to_split, partition_size=p_value, 
                                      time_series_clustered=p_group_splitted, tree_structure=tree_structure)

                logger.info("Start postprocessing k-anonymity top down approach") #tree structure??
                top_down_greedy_clustering_postprocessing(algorithm="kapra", time_series_clustered=p_group_splitted, 
                                                  tree_structure=tree_structure, partition_size=p_value)
                logger.info("End postprocessing k-anonymity top down approach")
                
                p_group_to_add += p_group_splitted # aggiungo il nuovo gruppone a quelli da aggiungere
                index_to_remove.append(index) # mi segno l'indice da togliere

        p_group_list = [group for (index, group) in enumerate(p_group_list) if index not in index_to_remove ]
        p_group_list += p_group_to_add #sopra tolgo i modificati, sotto aggiungo i modificati


        # start k_anonymity_top_down
        time_series_k_anonymized = list()
        """
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
        dataset_anonymized.save_on_file(Path(output_path))"""


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