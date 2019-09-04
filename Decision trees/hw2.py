import numpy as np
np.random.seed(42)

chi_table = {0.01  : 6.635,
             0.005 : 7.879,
             0.001 : 10.828,
             0.0005 : 12.116,
             0.0001 : 15.140,
             0.00001: 19.511}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the gini impurity of the dataset.    
    """
    gini = 0.0
    if (len(data) < 2):
        return 0
    instance_num, nonzero_num, zero_num  = numInstances(data)
    gini = 1 - (zero_num / instance_num)**2 - (nonzero_num / instance_num)**2
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the entropy of the dataset.    
    """
    entropy = 0.0
    if (len(data) < 2):
        return 0
    instance_num, nonzero_num, zero_num  = numInstances(data)
    s1 = zero_num / instance_num
    s2 = nonzero_num / instance_num
    if (s1 == 0 or s2 == 0):
        return 0
    entropy = - ((s1)*np.log2(s1)) - ((s2)*np.log2(s2))
    return entropy

class DecisionNode:

    # This class will hold everything you require to construct a decision tree.
    # The structure of this class is up to you. However, you need to support basic 
    # functionality as described in the notebook. It is highly recommended that you 
    # first read and understand the entire exercise before diving into this class.
    
    def __init__(self, feature, value):
        self.feature = feature # column index of criteria being tested
        self.value = value # value necessary to get a true result
        self.children = []
        self.isLeaf = False
        self.split = {0 : 0, 1 : 0, "total" : 0}
        self.parent = None
        
    def add_child(self, node):
        self.children.append(node)


def build_tree(data, impurity, chi_value=1):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure. 

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output: the root node of the tree.
    """
    
    thresh, feature, left_Child, right_Child = bestFeature(data,impurity)
    root = DecisionNode(feature, thresh)
    instance_num, nonzero_num, zero_num = numInstances(data)
    root.split[0] = zero_num
    root.split[1] = nonzero_num
    root.split["total"] = instance_num
    
    build_rec(root, data, impurity, chi_value)
    return root


def build_rec(father, father_data, impurity, chi_value):
    #stopping criteria
    if(impurity(father_data) == 0):
        father.isLeaf = True
        return
    #else, keep building the tree
    thresh, feature, left_data, right_data = bestFeature(father_data, impurity)
    
    #left_child
    left_thresh, left_feature, left_left_child, left_right_child = bestFeature(left_data,impurity)
    left_node = DecisionNode(left_feature, left_thresh)
    instance_num, nonzero_num, zero_num = numInstances(left_data)
    left_node.split[0] = zero_num
    left_node.split[1] = nonzero_num
    left_node.split["total"] = instance_num
    
    
    #right_child
    right_thresh, right_feature, right_left_Child, right_right_Child = bestFeature(right_data,impurity)
    right_node = DecisionNode(right_feature, right_thresh)
    instance_numr, nonzero_numr, zero_numr = numInstances(right_data)
    right_node.split[0] = zero_numr
    right_node.split[1] = nonzero_numr
    right_node.split["total"] = instance_numr
    
    #chi step
    if chi_value in chi_table:
        chiVal = chi_calc(father, left_node, right_node)
        if (chiVal < chi_table[chi_value]):
            father.isLeaf = True
            return
    
    left_node.parent = father
    right_node.parent = father
    father.add_child(left_node)
    father.add_child(right_node)
    
    build_rec(left_node, left_data, impurity, chi_value)
    build_rec(right_node, right_data, impurity, chi_value)
    
def chi_calc(father, left_child, right_child):
   # if (father.split["total"] == 0) :
   #     return 0
    E0 = (father.split[0]/(father.split["total"]))*(left_child.split["total"])
    p0 = left_child.split[0]
    n0 = left_child.split[1]
    
    E1 = (father.split[1]/(father.split["total"]))*(right_child.split["total"])
    p1 = right_child.split[0]
    n1 = right_child.split[1]
    
    #if (E0 == 0 or E1 == 0):
    #    return 0
    
    chi0 = ((p0 - E0)**2 / E0) + ((n0 - E1)**2 / E1)
    chi1 = ((p1 - E0)**2 / E0) + ((n1 - E1)**2 / E1)
    
    return chi0 + chi1

def bestThreshold (data,feature,impurity):
    currenThresh = 0
    maxThresh = 0
    maxGoodness = 0
    featureValue = 0
    maxSmaller = []
    maxBigger = []
    
    sort = np.sort(data[:,feature])
    #create the list of the thresholds
    thresholds = [(sort[i]+sort[i+1])/2 for i in range(0, len(sort)-1)]
    #avoid duplicate
    thresholds = set(thresholds)
    
    for i in range(len(thresholds)):
        currenThresh = thresholds.pop()
        big , small = splitData(data,feature,currenThresh)
        
        sizeBig = big.shape[0]
        sizeSmall = small.shape[0]
        sizeData = data.shape[0]
        featureValue = impurity(data) - ((sizeBig/sizeData)*impurity(big)) - ((sizeSmall/sizeData)*impurity(small))
        if (featureValue > maxGoodness):
            maxGoodness = featureValue
            maxThresh = currenThresh
            maxSmaller = small
            maxBigger = big
    
    return maxThresh, maxGoodness , maxSmaller , maxBigger

def splitData(data, col, threshold):
    above = []
    below = []
    for i in range(data.shape[0]):
        if (data[i,col] < threshold) :
            below.append(data[i])
        else:
            above.append(data[i])
    return np.array(above), np.array(below)


def bestFeature (data,impurity):
    maxFeature = 0
    maxValue = 0
    maxThresh = 0
    leftChild = []
    rightChild = []
    for i in range(data.shape[1]-1):
        thresh, value, small, big = bestThreshold (data,i,impurity);
        if (value > maxValue):
            maxFeature = i
            maxValue = value
            maxThresh = thresh
            leftChild = small
            rightChild = big
    
    return maxThresh, maxFeature, leftChild, rightChild

def numInstances(data):
    labels_col = data[:,-1]
    instance_num = labels_col.shape[0]
    nonzero_num = np.nonzero(labels_col)[0].shape[0]
    zero_num = instance_num - nonzero_num
    return instance_num, nonzero_num, zero_num

def predict(node, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    pred = None
    current_node = node
    while(current_node.isLeaf == False):
        if(instance[current_node.feature] >= current_node.value):
            current_node = current_node.children[1]
        else:
             current_node = current_node.children[0]
    pred = max(current_node.split.keys(), key=lambda i: current_node.split[i])
    return pred

def calc_accuracy(node, dataset):
    """
    calculate the accuracy starting from some node of the decision tree using
    the given dataset.

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0.0
    succesNum = 0
    for instance in dataset:
        if (predict(node, instance) == instance[-1]):
            succesNum = succesNum + 1
    accuracy = (succesNum / dataset.shape[0]) * 100
    return accuracy


def post_pruning(root,dataset):
    accuracy_full_tree = calc_accuracy(root, dataset) 
    acc_list_by_num_nodes = [accuracy_full_tree]
    current_nodes_num = numOfNodes(root)
    nodes_num = [current_nodes_num]
    best_accuracy = accuracy_full_tree
    best_parent = root
    
    leaves = leafList(root,[])
    
    while (len(root.children) != 0):
        best_accuracy = -1
        for leaf in leaves:
            temp = leaf.parent.children.copy()
            leaf.parent.children = []
            leaf.parent.isLeaf = True
            acc = calc_accuracy(root,dataset)
            if (acc>best_accuracy):
                best_accuracy = acc
                best_parent = leaf.parent
            leaf.parent.children = temp.copy()
            leaf.parent.isLeaf = False
        best_parent.children = []
        best_parent.isLeaf = True
        acc_list_by_num_nodes.append(best_accuracy) #insert for each cheack the best acc
        leaves = leafList(root,[])
        print()
        current_nodes_num = numOfNodes(root)
        print(numOfNodes)
        nodes_num.append(current_nodes_num)
     
    nodes_num.append(current_nodes_num)
    return acc_list_by_num_nodes, nodes_num      
        


def leafList (node,leaves):
    current_node = node
    if (current_node.isLeaf == True):
        leaves.append(current_node)
    else:
        leafList(current_node.children[0],leaves)
        leafList(current_node.children[1],leaves)
    return leaves

def numOfNodes(node):
    if (node.isLeaf):
        return 0
    return 1 + (numOfNodes(node.children[0]) + numOfNodes(node.children[1]))
    
        

def print_tree(node):
    '''
    prints the tree according to the example in the notebook

	Input:
	- node: a node in the decision tree

	This function has no return value
	'''

    print_tree_level(node,0)
    
    
def print_tree_level(node,level):
    print_node(node,level)
    if(len(node.children)==0):
        return
    print_tree_level(node.children[0],level+1)
    print_tree_level(node.children[1],level+1)
    
def print_node(node,level):
    if(len(node.children)>0):
        print("\t"*level,"[X",node.feature, " <= ", node.value, "]",sep='')
    else:
        if(max(node.split.keys(), key=lambda i: node.split[i]) == 1):
            a= {1.0: node.split[1]}
        else:
            a= {0.0: node.split[0]}
        print("\t"*level,"leaf: [",a ,"]",sep='' )
