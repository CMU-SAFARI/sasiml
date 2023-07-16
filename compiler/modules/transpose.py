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
from modules.common import next_num
from modules.common import prev_num

def matrix_trans(matrix):
    '''
    Regular transposition
    '''
    # Declare transposed matrix
    matrix_t = [[None for x in range(len(matrix))] for y in range(len(matrix[0]))]

    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
            matrix_t[y][x] = matrix[x][y]
    return matrix_t

def matrix_rot(matrix):
    '''
    Regular rotation
    '''
    # Declare rotated matrix
    matrix_r = [[None for x in range(len(matrix))] for y in range(len(matrix[0]))]

    matrix_h = len(matrix)
    matrix_w = len(matrix[0])

    for h in range(matrix_h):
        for w in range(matrix_w):
            rot_h = matrix_h - h -1
            rot_w = matrix_w - w -1
            matrix_r[rot_h][rot_w] = matrix[h][w]
    return matrix_r

def matrix_trans_conv(error, stride, fil, gradient, pad=0):
    '''
    Prepare the matrix for doing a transposed convolution
    '''
    _x = int(len(gradient[0]) - 1) + len(fil[0])
    _y = int(len(gradient) - 1) + len(fil)
    error_t_conv =  [[ pad for x in range(_x)] for y in range(_y)]
    if stride == 1:
        # Padding the borders
        offset_x = len(fil[0]) -1
        offset_y = len(fil) -1
        for x in range(len(error[0])):
            for y in range(len(error)):
                error_t_conv[offset_x+x][offset_y+y] = error[x][y]

    else:
        separation = stride    - 1
        offset_x = len(fil[0]) -1
        offset_y = len(fil) -1
        for x in range(len(error[0])):
            for y in range(len(error)):
                error_t_conv[offset_x+x*(1+separation)][offset_y+y*(1+separation)] = error[x][y]

    return error_t_conv


def gen_rot_from_trans(matrix_t):
    '''
    Generate the rotate matrix from the transposed matrix
    '''
    # The transpose of the transpose is the original matrix
    matrix = matrix_trans(matrix_t)
    matrix_r = matrix_rot(matrix)
    return matrix_r

def inner_padding(matrix, inner_pad):
    '''
    Insert inner padding
    '''
    h_m = len(matrix)
    w_m = len(matrix[0])
    res = [[0 for h in range(h_m+(h_m-1)*inner_pad)] for w in range(w_m+(w_m-1)*inner_pad)]

    for h in range(len(matrix)):
        for w in range(len(matrix[0])):
            res[h*(inner_pad+1)][w*(inner_pad+1)] = matrix[h][w]

    return res
