import numpy as np
from loguru import logger
from pathlib import Path


class DatasetAnonymized:
    def __init__(self, anonymized_data: list = list(), pattern_anonymized_data: dict = dict(), suppressed_data: list = list()):
        self.anonymized_data = anonymized_data #NOTE contiene i k-group
        self.pattern_anonymized_data = pattern_anonymized_data #dizionario con key, pattern per ogni tupla
        self.suppressed_data = suppressed_data
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
            
            #list_good_leaf_node = self.pattern_anonymized_data[index] #NOTE prende la lista delle good-leaf relativa al k-group
            
            max_value = np.amax(np.array(list(group.values())), 0)
            min_value = np.amin(np.array(list(group.values())), 0)
            for key in group.keys(): #NOTE per ogni chiave nel gruppo di chiavi del k-group
                # key = row product
                self.final_data_anonymized[key] = list()
                value_row = list()
                for column_index in range(0, len(max_value)):
                    value_row.append("[{}-{}]".format(min_value[column_index], max_value[column_index])) # NOTE anonimizza con il min e max del gruppo, ovvero l'envelope
                
                value_row.append(self.pattern_anonymized_data[key]) #FIXME aggiunge il pr 
                value_row.append("Group: {}".format(index))

                self.final_data_anonymized[key] = value_row #NOTE la riga comprensiva di tutti i dati
                logger.info(key)
                logger.info(value_row)
            logger.info("Finish creation Group {}".format(index))
        
        logger.info("Add suppressed data")
        for index in range(0, len(self.suppressed_data)):
            group = self.suppressed_data[index]
            for key in group.keys():
                value_row = [" - "]*len(group[key])
                value_row.append(" - ") # pattern rapresentation
                value_row.append(" - ") # group
                self.final_data_anonymized[key] = value_row



    def save_on_file(self, output_path):
        with open(output_path, "w") as file_to_write:
            value_to_print_on_file = ""
            for key, value in self.final_data_anonymized.items():
                value_to_print_on_file = key
                value_to_print_on_file = "{},{}".format(value_to_print_on_file, ",".join(value))
                file_to_write.write(value_to_print_on_file+"\n")