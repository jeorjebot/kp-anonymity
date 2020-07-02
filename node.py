import numpy as np
from saxpy.znorm import znorm
from saxpy.sax import ts_to_string
from saxpy.alphabet import cuts_for_asize
from loguru import logger
from saxpy.paa import paa


class Node:

    def __init__(self, level: int = 1, pattern_representation: str = "", label: str = "intermediate",
                 group: dict = None, parent=None, paa_value: int = 3):
        self.level = level  # number of different char for rappresentation
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
        # self.left = None  # left child
        # self.right = None  # right child
        self.parent = parent  # parent

    def start_splitting(self, p_value: int, max_level: int, good_leaf_nodes: list(), bad_leaf_nodes: list()):
        """
        Splitting Node Naive algorithm (k, P) Anonymity
        :param p_value:
        :param max_level:
        :param paa_value
        :return:
        """
        # logger.info("good_leaf_nodes: {}, bad_leaf_nodes: {}".format(len(good_leaf_nodes), len(bad_leaf_nodes)))
        if self.size < p_value:
            logger.info("size:{}, p_value:{} == bad-leaf".format(self.size, p_value))
            self.label = "bad-leaf"
            bad_leaf_nodes.append(self)
            return

        if self.level == max_level:
            logger.info("size:{}, p_value:{} == good-leaf".format(self.size, p_value))
            self.label = "good-leaf"
            good_leaf_nodes.append(self)
            return

        if p_value <= self.size < 2*p_value:
            logger.info("Maximize-level, size:{}, p_value:{} == good-leaf".format(self.size, p_value))
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
        tentative_child_node = dict() # NOTE è un dizionario di array, ogni array contiene le time series che hanno stesso pr (pattern repres.)
        temp_level = self.level + 1
        for key, value in self.group.items():
            # to reduce dimensionality
            data = np.array(value)
            data_znorm = znorm(data)
            data_paa = paa(data_znorm, self.paa_value)
            pr = ts_to_string(data_paa, cuts_for_asize(temp_level)) # NOTE crea il pattern sax della serie

            # NOTE se il pattern è uguale a uno presente, accodalo nell'array dentro al dizionario, altimenti crea un nuovo array con quel pr
            if pr in tentative_child_node.keys():
                tentative_child_node[pr].append(key)
            else:
                tentative_child_node[pr] = [key]

        length_all_tentative_child = [len(x) for x in list(tentative_child_node.values())] # NOTE crea un array di size degli array del dizionario [5,6,4,5] -> sono le size dei vari tentativi di child
        good_leaf = np.all(np.array(length_all_tentative_child) < p_value) # NOTE np.all() ritorna true se tutti gli elementi sono true.. 
       
        # NOTE quindi se tutti sono bad tentative allora lo segno come good_leaf e non faccio lo split
        if good_leaf:
            logger.info("Good-leaf, all_tentative_child are < {}".format(p_value))
            self.label = "good-leaf"
            good_leaf_nodes.append(self)
            return
        else: # NOTE se ho avuto almeno un tentativo > p_value, faccio lo split
            logger.info("N can be split")
            logger.info("Compute tentative good nodes and tentative bad nodes")
            
            # tentative good nodes
            # index of nodes in tentative_child_node with more p_value
            pr_keys = list(tentative_child_node.keys()) #NOTE sono le chiavi dei child node.. ma sono dei pattern!! "aaaabb" etc etc
            # get index tentative good node
            pattern_representation_tg = list()

            tg_nodes_index = list(np.where(np.array(length_all_tentative_child) >= p_value)[0]) #NOTE mi ritorna gli indici in length_all_tentat con size > p
            # NOTE sintassi strana: questo accrocchio ritorna una lista di array con le posizioni degli elementi che soddisfano la condizione. 
            # siccome length_all_tentative_child è monodimensionale, la lista conterrà un solo array
            # se fosse bidimensionale, mi ritornerebbe 2 array, con gli indici di x,y degli elementi
            # ecco spiegato perchè metto [0], mi interessa il primo array siccome so che length_all ha solo una dimensione -> mi interessa solo il primo array 
            
            # logger.info(pr_keys)
            tg_nodes = list()
            for index in tg_nodes_index:
                keys_elements = tentative_child_node[pr_keys[index]] #NOTE ritrova le chiavi del good leaf: pr[index] trova il pattern corrispondente, e in tentative_child il pattern è associato alle keys delle time series
                dict_temp = dict()
                for key in keys_elements:
                    dict_temp[key] = self.group[key] # NOTE aggiunge le time series corrispondenti alle chiavi
                tg_nodes.append(dict_temp)
                pattern_representation_tg.append(pr_keys[index])

            # tentative bad nodes
            tb_nodes_index = list(np.where(np.array(length_all_tentative_child) < p_value)[0]) #NOTE come sopra ma < p_value
            tb_nodes = list()
            pattern_representation_tb = list()

            for index in tb_nodes_index: #NOTE come sopra, crea la lista dei tb_nodes e appende la loro rappresentazione sax
                keys_elements = tentative_child_node[pr_keys[index]]
                dict_temp = dict()
                for key in keys_elements:
                    dict_temp[key] = self.group[key]
                tb_nodes.append(dict_temp)
                pattern_representation_tb.append(pr_keys[index])

            total_size_tb_nodes = 0 #NOTE per vedere se raggiungono p e quindi fare il childmerge
            for tb_node in tb_nodes:
                total_size_tb_nodes += len(tb_node)

            if total_size_tb_nodes >= p_value: #NOTE childmerge
                logger.info("Merge all bad nodes in a single node, and label it as good-leaf")
                child_merge_node_group = dict()
                for tb_node in tb_nodes:
                    for key, value in tb_node.items():
                        child_merge_node_group[key] = value #NOTE sposta tutto in childmerge
                node_merge = Node(level=self.level, pattern_representation=self.pattern_representation,
                                  label="good-leaf", group=child_merge_node_group, parent=self, paa_value=self.paa_value) #NOTE lo inizializzano con tt i parametri
                self.child_node.append(node_merge) #NOTE viene aggiunto alla struttura
                good_leaf_nodes.append(node_merge)

                nc = len(tg_nodes) + len(tb_nodes) 
                logger.info("Split only tg_nodes {0}".format(len(tg_nodes)))
                if nc >= 2:
                    for index in range(0, len(tg_nodes)): #NOTE aggiunge i tg node ai child e continua lo splitting
                        node = Node(level=self.level, pattern_representation=pattern_representation_tg[index],
                                    label="intermediate", group=tg_nodes[index], parent=self, paa_value=self.paa_value)
                        self.child_node.append(node)
                        node.start_splitting(p_value, max_level, good_leaf_nodes, bad_leaf_nodes)
                else:
                    for index in range(0, len(tg_nodes)): #NOTE aggiunge i tg ma non fa splitting
                        node = Node(level=self.level, pattern_representation=pattern_representation_tg[index],
                                    label="good-leaf", group=tg_nodes[index], parent=self, paa_value=self.paa_value)
                        self.child_node.append(node)
                        good_leaf_nodes.append(node)

            else: # NOTE se non c'è childmerge
                nc = len(tg_nodes) + len(tb_nodes) #NOTE come sopra ma senza childmerge, vedi il logger.info qua sotto
                logger.info("Label all tb_node {0} as bad-leaf and split only tg_nodes {1}".format(len(tb_nodes),len(tg_nodes)))
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
        # count = sum(1 for a, b in zip(seq1, seq2) if a != b)
        difference = float('inf')
        for bad_leaf_node in bad_leaf_nodes: #TODO dovrebbero venire ordinate in ordine ascendente, anche se non impatta, e dovebbe venire joinato con il nodo con minore size, a parità di PR
            pattern_representation_bad_node = bad_leaf_node.pattern_representation
            choose_node = None
            for index in range(0, len(good_leaf_nodes)):
                pattern_representation_good_node = good_leaf_nodes[index].pattern_representation
                difference_good_bad = sum(1 for a, b in zip(pattern_representation_good_node,
                                                            pattern_representation_bad_node) if a != b)
                #NOTE difference_good_bad conta le "lettere" di differenza tra le due pattern repres. "aaab" e "abbb" : 2 diff
                                         
                if difference_good_bad < difference:
                    choose_node = index

            # choose_node contain good node with minimum difference between pattern representation
            Node.add_row_to_node(good_leaf_nodes[choose_node], bad_leaf_node)
        # delete all bad_leaf nodes
        bad_leaf_nodes = list()

    @staticmethod
    def add_row_to_node(node_original, node_to_add):
        """
        add node_to_add content to node_original
        :param node_original:
        :param node_to_add:
        :return:
        """
        for key, value in node_to_add.group.items():  #NOTE aggiunge tutti gli elementi del bad leaf al good leaf
            node_original.group[key] = value
        node_original.members = list(node_original.group.keys()) #NOTE aggiorna size e members
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
            pr = ts_to_string(data_paa, cuts_for_asize(temp_level)) #NOTE prende il primo elemento e fa il pr con level+1, sarà il campione con cui confrontare gli altri
            for index in range(1, len(values_group)):
                data = np.array(values_group[index])
                data_znorm = znorm(data)
                data_paa = paa(data_znorm, self.paa_value)
                pr_2 = ts_to_string(data_paa, cuts_for_asize(temp_level))
                if pr_2 != pr:
                    equal = False #NOTE fa il check per tutti gli altri elementi se hanno stesso pr
            if equal:
                self.level = temp_level #NOTE tutti hanno stesso pr, quindi il livello si può incrementare
        if original_level != self.level: #NOTE nuovo livello, si aggiornano un po' di attributi
            logger.info("New level for node: {}".format(self.level))
            data = np.array(values_group[0])
            data_znorm = znorm(data)
            data_paa = paa(data_znorm, self.paa_value)
            self.pattern_representation = ts_to_string(data_paa, cuts_for_asize(self.level))
        else:
            logger.info("Can't split again, max level already reached") #NOTE: max level reached

    @staticmethod
    def recycle_bad_leaves(p_value, good_leaf_nodes, bad_leaf_nodes, suppressed_nodes, paa_value):
        """
        Recycle bad-leaves phase
        :param bad_leaf_nodes: [description]
        """
        
        """
        # da cancellare
        basic_node_1 = Node(label="bad-leaf", group=good_leaf_nodes[1].group, paa_value=paa_value)
        basic_node_1.level +=2
        basic_node_2 = Node(label="bad-leaf", group=good_leaf_nodes[0].group, paa_value=paa_value)
        basic_node_2.level +=2
        basic_node_3 = Node(label="bad-leaf", group=good_leaf_nodes[2].group, paa_value=paa_value, pattern_representation="aaaab")
        basic_node_3.level = 1
        bad_leaf_nodes.append(basic_node_1)
        bad_leaf_nodes.append(basic_node_2)
        bad_leaf_nodes.append(basic_node_3)
        """
        
        bad_leaf_nodes_dict = dict()
        for node in bad_leaf_nodes:
            if node.level in bad_leaf_nodes_dict.keys():
                bad_leaf_nodes_dict[node.level].append(node)
            else:
                bad_leaf_nodes_dict[node.level] = [node]

        bad_leaf_nodes_size = sum([node.size for node in bad_leaf_nodes])
        
        if bad_leaf_nodes_size >= p_value: #NOTE fai la recycle solo se ci sono più di p elementi, altrimenti sopprimi
        

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
                                keys_to_be_removed.remove(pr_node) #tolgo questo pr perchè un altro nodo con stesso pr è stato aggiunto, e quindi è un pr da tenere per il merge
                        else:
                            merge_dict[pr_node] = [current_level_node]
                            keys_to_be_removed.append(pr_node) #se non interviene un altro nodo sopra a toglierlo, alla fine verrà rimosso
                    
                    if merge:
                        for k in keys_to_be_removed: #pulizia delle keys che non devono essere mergiate
                            del merge_dict[k]

                        for pr, node_list in merge_dict.items():
                            group = dict()
                            for node in node_list:
                                bad_leaf_nodes_dict[current_level].remove(node) #elimino questi nodi dal dict
                                group.update(node.group) #concateno il gruppo
                            if current_level > 1:
                                level = current_level
                            else:
                                level = 1
                            leaf_merge = Node(level=level, pattern_representation=pr,
                                group=group, paa_value=paa_value)

                            # qua la metto bad o good. se good la sposto nelle good, se bad, la sposto nel bad-lead-dict
                            if leaf_merge.size >= p_value:
                                leaf_merge.label = "good-leaf"
                                good_leaf_nodes.append(leaf_merge)
                                bad_leaf_nodes_size -= leaf_merge.size #NOTE devo aggiornarla!!!
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
                        pr = ts_to_string(data_paa, cuts_for_asize(temp_level)) #NOTE prende il primo elemento e fa il pr con level+1, sarà il campione con cui confrontare gli altri
                    else:
                        pr = "a"*paa_value
                    node.level = temp_level
                    node.pattern_representation = pr

                if current_level > 0:
                    if temp_level not in bad_leaf_nodes_dict.keys(): # se la lista di nodi esiste già, concatenas
                        bad_leaf_nodes_dict[temp_level] = bad_leaf_nodes_dict.pop(current_level) #aggiorna il dizionario
                    else:
                        bad_leaf_nodes_dict[temp_level] = bad_leaf_nodes_dict[temp_level] + bad_leaf_nodes_dict.pop(current_level)
                    #del bad_leaf_nodes_dict[current_level] 
 
                    current_level -= 1
                else:
                    break 

        # TODO sopprimere le altre
        print("sopprimo le serie rimanenti")
        remaining_bad_leaf_nodes = list(bad_leaf_nodes_dict.values())[0]
        for node in remaining_bad_leaf_nodes:
            suppressed_nodes.append(node)
