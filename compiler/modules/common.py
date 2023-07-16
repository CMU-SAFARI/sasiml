""" $lic$
  Copyright (C) 2022 by Safari Research Group at ETH Zurich

  This file is part of SASiML.

  SASiML is free software; you can redistribute it and/or modify it under the
  terms of the MIT License as published by the Open Source Initiative.

  If you use this software in your research, we request that you reference
  the SASiML paper ("EcoFlow: Efficient Convolutional Dataflows for Low-Power
  Neural Network Accelerators", Orosa et al., arXiv, February 2022) as the
  source of the simulator in any publications that use this software, and that
  you send us a citation of your work.

  SASiML is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE. See the MIT License for more
  details.

  You should have received a copy of the MIT License along with
  this program. If not, see <https://opensource.org/licenses/MIT/>.
"""
#from numpy import nan
import modules.constants as c

def next_num(current, max):
    '''
    Next element
    '''
    next = None
    if current == max - 1:
        next = 0
    else:
        next = current + 1
    return next

def prev_num(current, max):
    '''
    Next element
    '''
    prev = None
    if current == 0:
        prev = max - 1
    else:
        prev = current - 1
    return prev

def gen_pos(matrix):
    matrix_w = len(matrix[0])
    matrix_h = len(matrix)
    res = [[None for w in range(matrix_w)] for h in range(matrix_h)]
    for h in range(matrix_h):
        for w in range(matrix_w):
            res[h][w] = [h,w]
    return res


def del_nan(array):
    '''
    Delete nan from an array
    '''
    res = []
    for i in range(len(array)):
        if array[i] is not c.NAN:
            res.append(array[i])
    return res

def index_nonan(array):
    '''
    Return the index of the first element that is not nan
    '''
    for i in range(len(array)):
        if array[i] is not c.NAN:
            return i

def index_nonan2(array):
    '''
    Return the index of the first element that is not nan
    '''
    for i in range(len(array)):
        if array[i] is not [c.NAN]:
            return i

def val_nonan(array,idx):
    '''
    Return the val of the idx element that is nan
    '''
    count = 0
    for i in range(len(array)):
        if array[i] is not c.NAN:
            if count == idx:
                return array[i]
            count += 1
    return -1

def val_nonan2(array,idx):
    '''
    Return the val of the idx element that is nan
    '''
    count = 0
    for i in range(len(array)):
        if type(array[i]) == type([]):
            if array[i] != [c.NAN]:
                for a in range(len(array[i])):
                    if count == idx:
                        return array[i][a]
                    count += 1
        else:
            if array[i] is not c.NAN:
                if count == idx:
                    return array[i]
                count += 1
    print("array: "+str(array))
    print("idx= "+str(idx))
    raise
    return -1

def first_index_noval(array, val):
    '''
    Return the index of the first element that is not nan
    '''
    for i in range(len(array)):
        if array[i] != val:
            return i
    return len(array)

def last_index_nonan(array):
    '''
    Return the index of the last element that is not nan
    '''
    idx = 0
    for i in range(len(array)):
        if (array[i] is not c.NAN) and (array[i] is not [c.NAN]):
            idx = i
    return idx


def get_idx_withnan(array,index):
    '''
    Skip the nan

    '''
    res_idx = 0
    for i in range(len(array)):
        if array[i] is not c.NAN:
            if res_idx == index:
                return i
            res_idx += 1

def merge(list1, val1, list2, val2, outval):
    '''
    Merge two lists
    '''
    mergedlist = []
    min_len = min(len(list1), len(list2))
    max_list = []
    max_val = None

    if len(list1) > len(list2):
        max_list = list1
        max_val  = val1
    else:
        max_list = list2
        max_val  = val2

    for i in range(min_len):
        if list1[i] is not val1 or list2[i] is not val2:
            mergedlist.append(1)
        else:
            mergedlist.append(outval)

    for i in range(min_len, len(max_list), 1):
        if max_list[i] is not max_val:
            mergedlist.append(1)
        else:
            mergedlist.append(outval)

    return mergedlist


def same_size(list1, val1, list2, val2):
    '''
    Make two list the same size
    '''
    if len(list1) > len(list2):
        for i in range(len(list2),len(list1),1):
                list2.append(val2)
    else:
        for i in range(len(list1),len(list2),1):
                list1.append(val1)




def all_same_size(array_h, array_w, *arg):
    '''
    Make all the input vectors the same size
    arg[i] is vector, arg[i+1] is the value to fill the vector (0, nan or '')
    Return the max size
    '''
    max_size = 0
    # Fist, we calculate the max of all of them
    for i in range(0, len(arg), 2):
        # For each argument
        for x in range(array_h):
            for y in range(array_w):
                max_size = max(max_size, len(arg[i][x][y]))
    for i in range(0, len(arg), 2):
        # For each argument
        for x in range(array_h):
            for y in range(array_w):
                for a in range(len(arg[i][x][y]), max_size, +1):
                    arg[i][x][y].append(arg[i+1])

    # Check that all have the same size
    for i in range(0, len(arg), 2):
        # For each argument
        for x in range(array_h):
            for y in range(array_w):
                if max_size != len(arg[i][x][y]):
                    raise

    return max_size


def fill_1Darray(max_size, val, array):
    '''
    Fill the array at the end with value, up to max size
    '''
    for i in range(len(array), max_size, 1):
        array.append(val)

