# This program uses Machine Learning concepts to predict with certain probability whether visiting customers will purchase vehicle during their visit to a 2- wheeler dealership.

# This program uses decision tree and perceptron to learn the data

# Karthik Suresh and Achal Puri are the authors 

#Import modified_binary_set_0_1.txt dataset
import numpy as np
import random as rm

#Importing training values
map_file = file("BINARY_SET_TRAIN_0_1_DECISION_TREE.txt", 'r')
lines = [l.rstrip().lower() for l in map_file.readlines()]
map_file.close()
_ROWS = len(lines)
a = list(lines[0])

#Extract data and make X,y and W arrays from it.
master_array_of_X = []
for each_row_of_lines in lines:
    array_of_X = []
    for len_each_row in xrange(len(each_row_of_lines)):
        if(each_row_of_lines[len_each_row] == '1'):
            array_of_X.append(1)
        elif(each_row_of_lines[len_each_row] == '0'):
            array_of_X.append(0)
    master_array_of_X.append(array_of_X)

master_array_of_labels = []
for len_of_X in xrange(len(master_array_of_X)):
    master_array_of_labels.append(master_array_of_X[len_of_X][-1])
y = master_array_of_labels
MAOX_ROWS = len(master_array_of_X)
MAOX_COLS = len(master_array_of_X[0])
arraytype_X = np.array(master_array_of_X)
arraytype_X_wo_label = np.zeros((MAOX_ROWS, MAOX_COLS-1), np.float)
for length_no_label_rows in xrange(MAOX_ROWS):
    for height_no_label_cols in xrange(MAOX_COLS - 1):
        arraytype_X_wo_label[length_no_label_rows][height_no_label_cols] = master_array_of_X[length_no_label_rows][height_no_label_cols]
array_of_X_values_of_ones = np.ones((MAOX_ROWS, 1), np.float)
array_of_X_values = np.concatenate((array_of_X_values_of_ones, arraytype_X_wo_label), axis = 1)
AOXV_ROWS = len(array_of_X_values)
AOXV_COLS = len(array_of_X_values[0])
x_new = arraytype_X
x_new_row = len(x_new)
x_new_col = len(x_new[0])


def entropy(pplus, pminus):
    if(pplus == 0 or pminus == 0):
        return(0)
    return((-pplus*np.log2(pplus))-(pminus*np.log2(pminus)))
    

#Calculating Expected Entropy
def Expected_Entropy(x_new, Attribute_location):
    x_new_ROWS = len(x_new)
    p_1 = 0
    p_2 = 0
    n_1 = 0
    n_2 = 0
    for row_no_x in xrange(len(x_new)):
        if(int(x_new[row_no_x][Attribute_location]) == 1):
            if(int(x_new[row_no_x][-1]) == 1):
                p_1 += 1
            elif(int(x_new[row_no_x][-1]) == 0):
                n_1 += 1
        elif(int(x_new[row_no_x][Attribute_location]) == 0):
            if(int(x_new[row_no_x][-1]) == 1):
                p_2 += 1
            elif(int(x_new[row_no_x][-1]) == 0):
                n_2 += 1
    total_1 = p_1 + n_1
    total_2 = p_2 + n_2
    if(total_1 == 0 or total_2 == 0):
        return(0)
    entropy_1 = entropy(float(p_1)/float(total_1), float(n_1)/float(total_1))
    entropy_2 = entropy(float(p_2)/float(total_2), float(n_2)/float(total_2))
    Expected_Entropy_2 = ((float(total_1)*float(entropy_1)/float(x_new_ROWS)) + (float(total_2)*float(entropy_2)/float(x_new_ROWS)))
    return(Expected_Entropy_2)

   

class Tree:
    def __init__(self, root):
        #root is the column number of Attribute 
        self.key = root
        #Creating a list of objects as self.children
        self.children = []          
        self.parent = None
        self.edge = []
        self.label = []
    
    def insert_children(self, child_node):
        self.children.append(child_node)     
    
    def get_children(self):
        return self.children
    
    def set_parent(self, parent):
        self.parent = parent
    
    def set_root_val(self, obj):
        self.key = obj
        
    def get_root_val(self):
        return self.key
    
    def insert_edge(self, value):
        self.edge.append(value)
        
    
def ID3(S,A):
    S_ROWS = len(S)
    if(S == []):
        S_COLS = 0
    elif(S!= []):
        S_COLS = len(S[0])
    ones_count = 0
    negative_ones_count = 0
    Tree_Object = Tree(None)
    for row_length_S in xrange(S_ROWS):
        if(int(S[row_length_S][-1]) == 1):
            ones_count  += 1
        elif(int(S[row_length_S][-1]) == 0):
            negative_ones_count += 1
    
    # If there are no edges and it is a label, then make key = None
    
    if(ones_count == S_ROWS):
        Tree_Object_1 = Tree(None)
        Tree_Object.insert_children(Tree_Object_1)
        Tree_Object.label.append(1)
        Tree_Object.label.append(1)
        Tree_Object_2 = Tree(None)
        Tree_Object.insert_children(Tree_Object_2)
        Tree_Object.set_root_val(None)
        return Tree_Object
    if(negative_ones_count == S_ROWS):
        Tree_Object_1 = Tree(None)
        Tree_Object.insert_children(Tree_Object_1)
        Tree_Object.label.append(0)
        Tree_Object.label.append(0)
        Tree_Object_2 = Tree(None)
        Tree_Object.insert_children(Tree_Object_2)
        Tree_Object.set_root_val(None)
        return Tree_Object
    
    #Insert edges = 0,1 and key
    
    if(A != [] and S_COLS != 0):
        ones_count = 0
        negative_ones_count = 0
        positive_labels = 0
        negative_labels = 0
        for rows in xrange(S_ROWS):
            if(int(S[rows][-1]) == 1):
                positive_labels += 1
            elif(int(S[rows][-1]) == 0):
                negative_labels += 1
    
        Label_Entropy = entropy(float(positive_labels)/float(S_ROWS), float(negative_labels)/float(S_ROWS))
        Array_of_Information_Gain = []
        for Attribute_location in A:
            Expected_Entropy_1 = Expected_Entropy(S, Attribute_location)
            Information_Gain = Label_Entropy - Expected_Entropy_1
            Array_of_Information_Gain.append(Information_Gain)
        max_Information_Gain = max(Array_of_Information_Gain)
        index_max_Information_Gain = Array_of_Information_Gain.index(max_Information_Gain)
        Best_Attribute = A[index_max_Information_Gain]
        Tree_Object.set_root_val(Best_Attribute)

    
        #You should change the below code for another dataset
        for all_values_of_Attribute in xrange(2):
            Tree_Object.insert_edge(all_values_of_Attribute)
        
        Rest_Attributes = []
        for remove_an_attribute in xrange(len(A)):
            if(A[remove_an_attribute] != Best_Attribute):
                Rest_Attributes.append(A[remove_an_attribute])
        master_array_of_index_of_S_with_value_i_of_Best_Attribute = []
        
        #inserting children
        for all_values_in_Best_Attribute in Tree_Object.edge:
            array_of_index_of_S_with_value_i_of_Best_Attribute = []
            for make_S_v in xrange(S_ROWS):
                if(int(S[make_S_v][Best_Attribute]) == all_values_in_Best_Attribute):
                    array_of_index_of_S_with_value_i_of_Best_Attribute.append(make_S_v)
            master_array_of_index_of_S_with_value_i_of_Best_Attribute.append(array_of_index_of_S_with_value_i_of_Best_Attribute)
    
    #Insert label, key = None 
    elif(A == []):
        if(ones_count >= negative_ones_count):
            Tree_Object.label.append(1)
            Tree_Object_1 = Tree(None)
            Tree_Object.insert_children(Tree_Object_1)
            Tree_Object.label.append(1)
            Tree_Object_2 = Tree(None)
            Tree_Object.insert_children(Tree_Object_2)
            Tree_Object.set_root_val(None)
            return Tree_Object
        else:
            Tree_Object.label.append(0)
            Tree_Object_1 = Tree(None)
            Tree_Object.insert_children(Tree_Object_1)
            Tree_Object.label.append(1)
            Tree_Object_2 = Tree(None)
            Tree_Object.insert_children(Tree_Object_2)
            Tree_Object.set_root_val(None)
            return Tree_Object
    
    #Create S_v(subset)
    total_of_S_v_equal_to_S = []
    for number in xrange(len(Tree_Object.edge)):
        S_v = np.zeros((len(master_array_of_index_of_S_with_value_i_of_Best_Attribute[number]), S_COLS), dtype = int)
        (S_v_rows,S_v_cols) = np.shape(S_v)
        row_put = 0
        for row_index_value in master_array_of_index_of_S_with_value_i_of_Best_Attribute[number]:
            S_v[row_put] = S[row_index_value]
            row_put += 1    
        total_of_S_v_equal_to_S.append(S_v)         #total_of_S_v_equal_to_S = [[S_v1], [S_v2], ...]
        
    #I have S_v and Rest_Attributes
    #Insert children = No of S_v
    for no_of_children in xrange(len(Tree_Object.edge)):
        ones_count = 0
        negative_ones_count = 0
        #print("total_of_S_v_equal_to_S[no_of_children]",total_of_S_v_equal_to_S[no_of_children])
        checking_value_present = 0
        for checking_value in total_of_S_v_equal_to_S[no_of_children]:
            checking_value_present = 1
        return_object = 0        
        if(checking_value_present == 0):
        #Return majority of label in S
            return_object = 1
            for row_length_S in xrange(S_ROWS):
                if(S[row_length_S][-1] == 1):
                    ones_count  += 1
                else:
                    negative_ones_count += 1
            if(ones_count >= negative_ones_count):
                Tree_Object_1 = Tree(None)
                Tree_Object.insert_children(Tree_Object_1)
                Tree_Object.label.append(1)
                
            elif(ones_count < negative_ones_count):
                Tree_Object_1 = Tree(None)
                Tree_Object.insert_children(Tree_Object_1)
                Tree_Object.label.append(0)


        if(return_object == 0):
            Tree_Object.insert_children(ID3(total_of_S_v_equal_to_S[no_of_children],Rest_Attributes))
            if(Tree_Object.get_children()[-1].key == None):
                Tree_Object.label.append(Tree_Object.get_children()[-1].label[0])
            elif(Tree_Object.get_children()[-1].key != None):
                Tree_Object.label.append(None)
    return(Tree_Object)
                
def display(objectoftree):
    if(objectoftree.key != None):
        print("key is", objectoftree.key)
        print("objectoftree.label",objectoftree.label)
        for i in xrange(2):
            print("%r Child is" %i)
            print(objectoftree.children[i].key)
    else:
        print("No key value")
    
def predict(objectoftree, dataset):
    if(objectoftree.key != None):
        value_in_edge = dataset[objectoftree.key]
        objectoftree.children[value_in_edge].set_parent(objectoftree)
        final_label = predict(objectoftree.children[value_in_edge], dataset)
        return(final_label)
    elif(objectoftree.key == None):
        for i in xrange(2):
            if(objectoftree.parent.label[i] != None):
                return(objectoftree.parent.label[i])   
    
Attributes_are_numbers = range(0,x_new_col-1)
id3_object = ID3(x_new, Attributes_are_numbers)

#Import test values
map_file = file("BINARY_SET_TEST_0_1_DECISION_TREE.txt", 'r')
lines = [l.rstrip().lower() for l in map_file.readlines()]
map_file.close()
_ROWS = len(lines)
a = list(lines[0])

#Extract data and make X,y and W arrays from it.
master_array_of_X = []
for each_row_of_lines in lines:
    array_of_X = []
    for len_each_row in xrange(len(each_row_of_lines)):
        if(each_row_of_lines[len_each_row] == '1'):
            array_of_X.append(1)
        elif(each_row_of_lines[len_each_row] == '0'):
            array_of_X.append(0)

    master_array_of_X.append(array_of_X)

master_array_of_labels = []
for len_of_X in xrange(len(master_array_of_X)):
    master_array_of_labels.append(master_array_of_X[len_of_X][-1])
y = master_array_of_labels
MAOX_ROWS = len(master_array_of_X)
MAOX_COLS = len(master_array_of_X[0])
arraytype_X = np.array(master_array_of_X)
arraytype_X_wo_label = np.zeros((MAOX_ROWS, MAOX_COLS-1), np.float)
for length_no_label_rows in xrange(MAOX_ROWS):
    for height_no_label_cols in xrange(MAOX_COLS - 1):
        arraytype_X_wo_label[length_no_label_rows][height_no_label_cols] = master_array_of_X[length_no_label_rows][height_no_label_cols]
array_of_X_values_of_ones = np.ones((MAOX_ROWS, 1), np.float)
array_of_X_values = np.concatenate((array_of_X_values_of_ones, arraytype_X_wo_label), axis = 1)
AOXV_ROWS = len(array_of_X_values)
AOXV_COLS = len(array_of_X_values[0])
x_new = arraytype_X
x_new_row = len(x_new)
x_new_col = len(x_new[0])



#Prediction
#When it comes to prediction, the decision tree goes from the root to the children until it finds a label. The root is given the highest priority since the output depends on the root more than it does on the root's children.
prediction = []
for testing_each_row in xrange(x_new_row):
    prediction.append(predict(id3_object, x_new[testing_each_row]))
correct_prediction = 0
list_of_true_labels = []
for length in xrange(len(prediction)):
    list_of_true_labels.append(x_new[length][-1])
    if(x_new[length][-1] == prediction[length]):
        correct_prediction += 1
print("prediction",prediction)
print("actuallabels",list_of_true_labels)
accuracy = (float(correct_prediction)/float(x_new_row))*100
print("accuracy of test data is",accuracy)

#Below is the code to display the root and the immediate children. I have the decision tree displayed till level 6.
display(id3_object)
display(id3_object.children[0])
display(id3_object.children[1])
display(id3_object.children[0].children[0])
display(id3_object.children[0].children[1])
display(id3_object.children[1].children[0])
display(id3_object.children[1].children[1])
display(id3_object.children[0].children[0].children[0])
display(id3_object.children[0].children[0].children[1])
display(id3_object.children[1].children[1].children[0])
display(id3_object.children[1].children[1].children[1])
display(id3_object.children[0].children[0].children[0].children[0])
display(id3_object.children[0].children[0].children[1].children[0])
display(id3_object.children[0].children[0].children[1].children[1])
display(id3_object.children[1].children[1].children[0].children[0])
display(id3_object.children[0].children[0].children[0].children[0].children[0])
display(id3_object.children[0].children[0].children[1].children[0].children[0])
display(id3_object.children[0].children[0].children[1].children[1].children[0])
display(id3_object.children[0].children[0].children[1].children[1].children[1])
display(id3_object.children[1].children[1].children[0].children[0].children[0])
