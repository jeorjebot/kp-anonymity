import numpy as np
from saxpy.znorm import znorm
from saxpy.sax import ts_to_string
from saxpy.alphabet import cuts_for_asize
from loguru import logger
from saxpy.paa import paa


class Node:

    def __init__(self, level: int = 1, pattern_representation: str = "", label: str = "intermediate",
                 group: dict = None, parent=None, paa_value: int = 3):
        self.level = level
        self.paa_value = paa_value
        if pattern_representation == "":
            pr = "a"*self.paa_value  # using SAX
            self.pattern_representation = pr
        else:
            self.pattern_representation = pattern_representation
        self.members = list(group.keys())  # members time series contained in N
        self.size = len(group)  # numbers of time series contained
        self.label = label  # each node has tree possible labels: bad-leaf, good-leaf or intermediate
        self.group = group  # group obtained from k-anonymity top-down
        self.child_node = list()  # all childs node
        self.parent = parent  # parent

    def start_splitting(self, p_value: int, max_level: int, good_leaf_nodes: list(), bad_leaf_nodes: list()):
        """
        Splitting Node Naive algorithm (k, P) Anonymity
        :param p_value:
        :param max_level:
        :param paa_value
        :return:
        """
        if self.size < p_value:
            #logger.info("size:{}, p_value:{} == bad-leaf".format(self.size, p_value))
            self.label = "bad-leaf"
            bad_leaf_nodes.append(self)
            return

        if self.level == max_level:
            #logger.info("size:{}, p_value:{} == good-leaf".format(self.size, p_value))
            self.label = "good-leaf"
            good_leaf_nodes.append(self)
            return

        if p_value <= self.size < 2*p_value:
            #logger.info("Maximize-level, size:{}, p_value:{} == good-leaf".format(self.size, p_value))
            self.maximize_level_node(max_level)
            self.label = "good-leaf"
            good_leaf_nodes.append(self)
            return
        """
        Otherwise, we need to check if node N has to be split. The checking relies on a tentative split performed on N. 
        Suppose that, by increasing the level of N, N is tentatively split into a number of child nodes. 
        If all these child nodes contain fewer than P time series, no real split is performed and the original node N is
        labeled as good-leaf and the recursion terminates on N. Otherwise, there must exist tentative child node(s) 
        whose size >= P, also called TG-node(s) (Tentative Good Nodes). 
        The rest children whose size < P are called TB-nodes (Tentative Bad Nodes), if any. 
        If the total number of records in all TB-nodes under N is no less than P, we merge them into a single tentative
        node, denoted by childmerge, at the level of N.level. If the above tentative process produces nc tentative 
        child nodes (including TB and TG) and nc >= 2, N will really be split into nc children and then the node 
        splitting procedure will be recursively invoked on each of them 
        """
        tentative_child_node = dict() 
        temp_level = self.level + 1
        for key, value in self.group.items():
            # to reduce dimensionality
            data = np.array(value)
            data_znorm = znorm(data)
            data_paa = paa(data_znorm, self.paa_value)
            pr = ts_to_string(data_paa, cuts_for_asize(temp_level))

            if pr in tentative_child_node.keys():
                tentative_child_node[pr].append(key)
            else:
                tentative_child_node[pr] = [key]

        length_all_tentative_child = [len(x) for x in list(tentative_child_node.values())] 
        good_leaf = np.all(np.array(length_all_tentative_child) < p_value)
       
        if good_leaf:
            #logger.info("Good-leaf, all_tentative_child are < {}".format(p_value))
            self.label = "good-leaf"
            good_leaf_nodes.append(self)
            return
        else:
            #logger.info("N can be split")
            #logger.info("Compute tentative good nodes and tentative bad nodes")
            pr_keys = list(tentative_child_node.keys()) 
            # get index tentative good node
            pattern_representation_tg = list()

            tg_nodes_index = list(np.where(np.array(length_all_tentative_child) >= p_value)[0])
            
            # logger.info(pr_keys)
            tg_nodes = list()
            for index in tg_nodes_index:
                keys_elements = tentative_child_node[pr_keys[index]]
                dict_temp = dict()
                for key in keys_elements:
                    dict_temp[key] = self.group[key]
                tg_nodes.append(dict_temp)
                pattern_representation_tg.append(pr_keys[index])

            # tentative bad nodes
            tb_nodes_index = list(np.where(np.array(length_all_tentative_child) < p_value)[0])
            tb_nodes = list()
            pattern_representation_tb = list()

            for index in tb_nodes_index:
                keys_elements = tentative_child_node[pr_keys[index]]
                dict_temp = dict()
                for key in keys_elements:
                    dict_temp[key] = self.group[key]
                tb_nodes.append(dict_temp)
                pattern_representation_tb.append(pr_keys[index])

            total_size_tb_nodes = 0
            for tb_node in tb_nodes:
                total_size_tb_nodes += len(tb_node)

            if total_size_tb_nodes >= p_value:
                #logger.info("Merge all bad nodes in a single node, and label it as good-leaf")
                child_merge_node_group = dict()
                for tb_node in tb_nodes:
                    for key, value in tb_node.items():
                        child_merge_node_group[key] = value
                node_merge = Node(level=self.level, pattern_representation=self.pattern_representation,
                                  label="good-leaf", group=child_merge_node_group, parent=self, paa_value=self.paa_value)
                self.child_node.append(node_merge)
                good_leaf_nodes.append(node_merge)

                nc = len(tg_nodes) + len(tb_nodes) 
                #logger.info("Split only tg_nodes {0}".format(len(tg_nodes)))
                if nc >= 2:
                    for index in range(0, len(tg_nodes)):
                        node = Node(level=self.level, pattern_representation=pattern_representation_tg[index],
                                    label="intermediate", group=tg_nodes[index], parent=self, paa_value=self.paa_value)
                        self.child_node.append(node)
                        node.start_splitting(p_value, max_level, good_leaf_nodes, bad_leaf_nodes)
                else:
                    for index in range(0, len(tg_nodes)):
                        node = Node(level=self.level, pattern_representation=pattern_representation_tg[index],
                                    label="good-leaf", group=tg_nodes[index], parent=self, paa_value=self.paa_value)
                        self.child_node.append(node)
                        good_leaf_nodes.append(node)

            else: 
                nc = len(tg_nodes) + len(tb_nodes) 
                #logger.info("Label all tb_node {0} as bad-leaf and split only tg_nodes {1}".format(len(tb_nodes),len(tg_nodes)))
                for index in range(0, len(tb_nodes)):
                    node = Node(level=self.level, pattern_representation=pattern_representation_tb[index], label="bad-leaf",
                                group=tb_nodes[index], parent=self, paa_value=self.paa_value)
                    self.child_node.append(node)
                    bad_leaf_nodes.append(node)
                if nc >= 2:
                    for index in range(0, len(tg_nodes)):
                        node = Node(level=self.level, pattern_representation=pattern_representation_tg[index],
                                    label="intermediate", group=tg_nodes[index], parent=self, paa_value=self.paa_value)
                        self.child_node.append(node)
                        node.start_splitting(p_value, max_level, good_leaf_nodes, bad_leaf_nodes)
                else:
                    for index in range(0, len(tg_nodes)):
                        node = Node(level=self.level, pattern_representation=pattern_representation_tg[index],
                                    label="good-leaf", group=tg_nodes[index], parent=self, paa_value=self.paa_value)
                        self.child_node.append(node)
                        good_leaf_nodes.append(node)

    @staticmethod
    def postprocessing(good_leaf_nodes, bad_leaf_nodes):
        difference = float('inf')
        for bad_leaf_node in bad_leaf_nodes: 
            pattern_representation_bad_node = bad_leaf_node.pattern_representation
            choose_node = None
            for index in range(0, len(good_leaf_nodes)):
                pattern_representation_good_node = good_leaf_nodes[index].pattern_representation
                difference_good_bad = sum(1 for a, b in zip(pattern_representation_good_node,
                                                            pattern_representation_bad_node) if a != b)
                                         
                if difference_good_bad < difference:
                    choose_node = index
            Node.add_row_to_node(good_leaf_nodes[choose_node], bad_leaf_node)
        bad_leaf_nodes = list()

    @staticmethod
    def add_row_to_node(node_original, node_to_add):
        """
        add node_to_add content to node_original
        :param node_original:
        :param node_to_add:
        :return:
        """
        for key, value in node_to_add.group.items():
            node_original.group[key] = value
        node_original.members = list(node_original.group.keys())
        node_original.size = len(node_original.group)

    def maximize_level_node(self, max_level):
        """
        Try to maximaxe the level value
        :param p_value:
        :return:
        """
        values_group = list(self.group.values())
        original_level = self.level
        equal = True
        while equal and self.level < max_level:
            temp_level = self.level + 1
            data = np.array(values_group[0])
            data_znorm = znorm(data)
            data_paa = paa(data_znorm, self.paa_value)
            pr = ts_to_string(data_paa, cuts_for_asize(temp_level))
            for index in range(1, len(values_group)):
                data = np.array(values_group[index])
                data_znorm = znorm(data)
                data_paa = paa(data_znorm, self.paa_value)
                pr_2 = ts_to_string(data_paa, cuts_for_asize(temp_level))
                if pr_2 != pr:
                    equal = False
            if equal:
                self.level = temp_level 
        if original_level != self.level: 
            #logger.info("New level for node: {}".format(self.level))
            data = np.array(values_group[0])
            data_znorm = znorm(data)
            data_paa = paa(data_znorm, self.paa_value)
            self.pattern_representation = ts_to_string(data_paa, cuts_for_asize(self.level))
        else:
            #logger.info("Can't split again, max level already reached") # max level reached
            pass

    @staticmethod
    def recycle_bad_leaves(p_value, good_leaf_nodes, bad_leaf_nodes, suppressed_nodes, paa_value):
        """
        Recycle bad-leaves phase
        :param bad_leaf_nodes: [description]
        """

        bad_leaf_nodes_dict = dict()
        for node in bad_leaf_nodes:
            if node.level in bad_leaf_nodes_dict.keys():
                bad_leaf_nodes_dict[node.level].append(node)
            else:
                bad_leaf_nodes_dict[node.level] = [node]

        bad_leaf_nodes_size = sum([node.size for node in bad_leaf_nodes])
        
        if bad_leaf_nodes_size >= p_value:
    
            # max bad level
            current_level = max(bad_leaf_nodes_dict.keys())

            while bad_leaf_nodes_size >= p_value:
                
                if current_level in bad_leaf_nodes_dict.keys():
                    merge_dict = dict()
                    keys_to_be_removed = list()
                    merge = False
                    for current_level_node in bad_leaf_nodes_dict[current_level]:
                        pr_node = current_level_node.pattern_representation
                        if pr_node in merge_dict.keys():
                            merge = True
                            merge_dict[pr_node].append(current_level_node)
                            if pr_node in keys_to_be_removed:
                                keys_to_be_removed.remove(pr_node)
                        else:
                            merge_dict[pr_node] = [current_level_node]
                            keys_to_be_removed.append(pr_node)
                    
                    if merge:
                        for k in keys_to_be_removed:
                            del merge_dict[k]

                        for pr, node_list in merge_dict.items():
                            group = dict()
                            for node in node_list:
                                bad_leaf_nodes_dict[current_level].remove(node)
                                group.update(node.group)
                            if current_level > 1:
                                level = current_level
                            else:
                                level = 1
                            leaf_merge = Node(level=level, pattern_representation=pr,
                                group=group, paa_value=paa_value)

                            if leaf_merge.size >= p_value:
                                leaf_merge.label = "good-leaf"
                                good_leaf_nodes.append(leaf_merge)
                                bad_leaf_nodes_size -= leaf_merge.size
                            else: 
                                leaf_merge.label = "bad-leaf"
                                bad_leaf_nodes_dict[current_level].append(leaf_merge)

                temp_level = current_level-1
                for node in bad_leaf_nodes_dict[current_level]:
                    if temp_level > 1:
                        values_group = list(node.group.values())
                        data = np.array(values_group[0])
                        data_znorm = znorm(data)
                        data_paa = paa(data_znorm, paa_value)
                        pr = ts_to_string(data_paa, cuts_for_asize(temp_level))
                    else:
                        pr = "a"*paa_value
                    node.level = temp_level
                    node.pattern_representation = pr

                if current_level > 0:
                    if temp_level not in bad_leaf_nodes_dict.keys():
                        bad_leaf_nodes_dict[temp_level] = bad_leaf_nodes_dict.pop(current_level)
                    else:
                        bad_leaf_nodes_dict[temp_level] = bad_leaf_nodes_dict[temp_level] + bad_leaf_nodes_dict.pop(current_level) 
                    current_level -= 1
                else:
                    break 

        #print("sopprimo le serie rimanenti")
        remaining_bad_leaf_nodes = list(bad_leaf_nodes_dict.values())[0]
        for node in remaining_bad_leaf_nodes:
            suppressed_nodes.append(node)
