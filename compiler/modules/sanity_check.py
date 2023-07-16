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
"""
Helper functions to validate the validity
of the ifm, filter and output dimensions
"""

def check_validity(f):
    '''
    Check the valifity of the input arguments (i.e., the dimensions of the matrix)
    '''
    check_dimension(f.filter)
    check_dimension(f.ifm)
    check_dimension(f.ofm)
    check_dimension(f.error)
    check_dimension(f.gradient)
    check_conv(f.ifm, f.filter, f.ofm, f.stride)

    # IFM and gradient have to be the same dimensions
    if len(f.ifm) != len(f.gradient):
        raise
    if len(f.ifm[0]) != len(f.gradient[0]):
        raise

    # OFM and Error have to be the same dimension
    if len(f.ofm) != len(f.error):
        raise
    if len(f.ofm[0]) != len(f.error[0]):
        raise

def check_fgrad(f):
    '''
    Check the validity of the dimensions to calculate the Filter Gradients
    '''
    # H
    p1 = (len(f.ifm[0]) - (len(f.error) + (len(f.error) - 1)*(f.stride - 1)))
    dim_h = p1 + 1
    if dim_h != len(f.filter):
        print("[ERROR] Wrong input dimensions, they do not match")
        raise

    # W
    p1 = (len(f.ifm) - (len(f.error[0]) + (len(f.error) - 1)*(f.stride - 1)))
    dim_w = p1 + 1
    if dim_w != len(f.filter[0]):
        print("[ERROR] Wrong input dimensions, they do not match")
        raise

def check_dimension(array):
    '''
    Check that all the columns of the array have the same dimensions
    Check that all the rows of the array have the  same dimensions
    '''
    for i in range(1, len(array), 1):
        if len(array[i-1]) is not len(array[i]):
            raise
    return 0

def check_conv(ifm, fil, ofm, stride):
    '''
    The input should be already padded, and all the dimensions should match
    '''
    # Check the w axis
    if (len(ifm[0]) - len(fil[0]))%stride != 0:
        print("[ERROR] Wrong input dimensions, they do not match")
        print("ifm="+str(len(ifm[0]))+" , fil="+str(len(fil[0]))+" , stride="+str(stride))
        raise
    e_w_ofm = (len(ifm[0]) - len(fil[0]))/stride + 1
    if e_w_ofm != len(ofm[0]):
        print("[ERROR] ofm w="+str(float(len(ofm[0])))+", expected w="+str(e_w_ofm))
        print("ifm="+str(len(ifm))+" , fil="+str(len(fil))+" , stride="+str(stride)+" , ofm="+str(len(ofm)))
        raise
    # Check the h axis
    if (len(ifm) - len(fil))%stride != 0:
        print("[ERROR] Wrong input dimensions, they do not match")
        print("ifm="+str(len(ifm))+" , fil="+str(len(fil))+" , stride="+str(stride))
        raise
    e_h_ofm = (len(ifm) - len(fil))/stride + 1
    if e_h_ofm != len(ofm):
        print("[ERROR] ofm h="+str(len(ofm))+", expected h="+str(e_h_ofm))
        raise
    return 0
