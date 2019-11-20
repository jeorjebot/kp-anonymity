import numpy as np
from loguru import logger


class DatasetAnonymized:
    def __init__(self, anonymized_data: list = list(), pattern_anonymized_data: list = list()):
        self.anonymized_data = anonymized_data
        self.pattern_anonymized_data = pattern_anonymized_data
        self.final_data_anonymized = dict()


    def compute_anonymized_data(self):
        """
        create dataset ready to be anonymized
        :return:
        """
        logger.info("Start creation dataset anonymized")
        for index in range(0, len(self.anonymized_data)):
            logger.info("Start creation Group {}".format(index))

            group = self.anonymized_data[index]
            list_good_leaf_node = self.pattern_anonymized_data[index]
            max_value = np.amax(np.array(list(group.values())), 0)
            min_value = np.amin(np.array(list(group.values())), 0)
            for key in group.keys():
                # key = row product
                self.final_data_anonymized[key] = list()
                value_row = list()
                for column_index in range(0, len(max_value)):
                    value_row.append("[{}-{}]".format(min_value[column_index], max_value[column_index]))
                for node in list_good_leaf_node:
                    if key in node.group.keys():
                        value_row.append(node.pattern_representation)
                value_row.append("Group: {}".format(index))
                self.final_data_anonymized[key] = value_row
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