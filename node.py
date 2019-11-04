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
        self.members = list(group.keys())  # members   time series contained in N
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
        
        """
        - If N.size < P, then the node is labeled as bad-leaf and the recursion terminates.
        
        - Otherwise if N.level == max-level, then the node is labeled as good leaf and the recursion terminates
        
        - Else if P <= N.size < 2 <= P, we try to maximize the level of N as long as all records of N have the identical PR. 
            The node is labeled as good-leaf and the recursion terminates. 
            We try to avoid node splitting in such case for two reasons: 
            First, this node is a good candidate for the resulting P-subgroups; 
            second, a split on this node will for sure generate at least one bad-leaf, which will subsequently increase the burden of the next step.
        
        - Otherwise, we need to check if node N has to be split. The checking relies on a tentative split performed on N. 
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
        