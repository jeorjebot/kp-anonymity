import numpy as np
from loguru import logger
from pathlib import Path


class DatasetAnonymized:
    def __init__(self, anonymized_data: list = list(), pattern_anonymized_data: list = list()):
        self.anonymized_data = anonymized_data #NOTE contiene i k-group
        self.pattern_anonymized_data = pattern_anonymized_data #NOTE contiene le liste dei relativi good-leaf-nodes corrispondenti ai k-group
        self.final_data_anonymized = dict()


    def compute_anonymized_data(self):
        """
        Create dataset ready to be anonymized
        :return:
        """
        logger.info("Start creation dataset anonymized")
        for index in range(0, len(self.anonymized_data)): #NOTE gli anonymized_data sono i k-groups
            logger.info("Start creation Group {}".format(index))

            group = self.anonymized_data[index] #NOTE prende un k-group
            list_good_leaf_node = self.pattern_anonymized_data[index] #NOTE prende la lista delle good-leaf relativa al k-group
            max_value = np.amax(np.array(list(group.values())), 0)
            min_value = np.amin(np.array(list(group.values())), 0)
            for key in group.keys(): #NOTE per ogni chiave nel gruppo di chiavi del k-group
                # key = row product
                self.final_data_anonymized[key] = list()
                value_row = list()
                for column_index in range(0, len(max_value)):
                    value_row.append("[{}-{}]".format(min_value[column_index], max_value[column_index])) # NOTE anonimizza con il min e max del gruppo, ovvero l'envelope
                for node in list_good_leaf_node: #NOTE itera tutti i nodi good leaf relativi al k-group, cercando il nodo (P-group) che contenga la time series, in modo da aggiungere il PR
                        value_row.append(node.pattern_representation) #NOTE aggiunge la PR relativa alla riga
                value_row.append("Group: {}".format(index))
                self.final_data_anonymized[key] = value_row #NOTE la riga comprensiva di tutti i dati
                logger.info(key)
                logger.info(value_row)
            logger.info("Finish creation Group {}".format(index))

    def save_on_file(self, name_file):
        with open(name_file, "w") as file_to_write:
            value_to_print_on_file = ""
            for key, value in self.final_data_anonymized.items():
                value_to_print_on_file = key
                value_to_print_on_file = "{},{}".format(value_to_print_on_file, ",".join(value))
                file_to_write.write(value_to_print_on_file+"\n")