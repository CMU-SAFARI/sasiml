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

def lowering(ifm, fil, str):
    '''
    lower the convolution into a matrix multiplication
    '''
    # Declare transposed matrix
    mat1_x = int((((len(ifm[0])-len(fil[0]))/str)+1)*(((len(ifm) - len(fil))/str) +1 ))
    mat1_y = len(fil)*len(fil[0])
    mat2_x = mat1_y
    mat2_y = 1

    mat1 = [[None for x in range(mat1_x)] for y in range(mat1_y)]
    mat2 = [[None for x in range(mat2_x)] for y in range(mat2_y)]

    w = 0
    h = 0
    for x in range(0,(len(ifm[0])-len(fil[0])+1), str):
        for y in range(0,(len(ifm)-len(fil)+1), str):
            # This is the initial position of a particular convolution
            h = 0
            for fx in range(len(fil[0])):
                for fy in range(len(fil)):
                    mat1[h][w] = ifm[y+fy][x+fx]
                    h +=1
            w +=1

    idx = 0
    for x in range(len(fil[0])):
        for y in range(len(fil)):
            mat2[0][idx] = fil[y][x]
            idx += 1

    return mat1, mat2

def lowering_fgrad(ifm, fil, stride):
    '''
    lower the convolution into a matrix multiplication, fgrad
    '''
    # Declare transposed matrix
    mat1_x = int((((len(ifm[0])-(len(fil[0])*stride-1))+1)))
    mat1_x *= mat1_x
    mat1_y = len(fil)*len(fil[0])
    mat2_x = mat1_y
    mat2_y = 1

    print("mat1_x: "+str(mat1_x)+" mat1_y: "+str(mat1_y))

    #assert 0

    mat1 = [[None for x in range(mat1_x)] for y in range(mat1_y)]
    mat2 = [[None for x in range(mat2_x)] for y in range(mat2_y)]

    w = 0
    h = 0
    for x in range(0,int((((len(ifm[0])-(len(fil[0])*stride-1))+1))), stride):
        for y in range(0,int((((len(ifm[0])-(len(fil[0])*stride-1))+1))), stride):
            # This is the initial position of a particular convolution
            h = 0
            for fx in range(len(fil[0])):
                for fy in range(len(fil)):
                    mat1[h][w] = ifm[y+fy][x+fx]
                    h +=1
            w +=1

    idx = 0
    for x in range(len(fil[0])):
        for y in range(len(fil)):
            mat2[0][idx] = fil[y][x]
            idx += 1

    return mat1, mat2

def zeros_systolic(mat1,mat2):
    '''
    Adapt the matrix to the systolic array dataflow
    '''
    mat1_zeros = [[None for x in range(len(mat1[0]))] for y in range(len(mat1)+len(mat1[0])-1)]
    mat2_zeros = [[None for x in range(len(mat1_zeros))] for y in range(len(mat2))]

    # Matrix 1
    for x in range(len(mat1_zeros[0])):
        for y in range(len(mat1_zeros)):
            if y < x:
                mat1_zeros[y][x] = None
            elif y > (x +len(mat1)-1):
                mat1_zeros[y][x] = None
            else:
                mat1_zeros[y][x] = mat1[y-x][x]

    # Matrix 2
    for x in range(len(mat2_zeros[0])):
        if x < len(mat2[0]):
            mat2_zeros[0][x] = mat2[0][x]
        else:
            mat2_zeros[0][x] = None
    return mat1_zeros, mat2_zeros

