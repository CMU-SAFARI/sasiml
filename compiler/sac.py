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

'''
SASiML Compiler
'''

import argparse
import importlib.util
import types
from argparse import RawTextHelpFormatter
from modules.conv import conv
from modules.multiply import multiply
from modules.gflow_igrad import gflow_igrad
from modules.gflow_fgrad import gflow_fgrad
from modules.dump import dump_to_file
import modules.constants as s
import os.path

from modules.transpose import matrix_trans
from modules.transpose import matrix_trans_conv
from modules.transpose import matrix_rot
from modules.lowering import lowering
from modules.lowering import lowering_fgrad
from modules.lowering import zeros_systolic
from modules.sanity_check import check_conv
from modules.sanity_check import check_validity
from modules.sanity_check import check_fgrad
from modules.transpose import inner_padding

print("Compiler...")
import os
sasimpath = os.environ['SASIMPATH']


import sys
sys.path.insert(0,sasimpath)
import hw.constants as c

# Parse arguments
parser = argparse.ArgumentParser(description='Compile signal for a spatial architecture', formatter_class=RawTextHelpFormatter)
parser.add_argument('-i', help="file where the IFM, the FILTER and the OFM are described", required=True)
parser.add_argument('-o', help="output file", default="out.py")
parser.add_argument('-p', choices=['forward','igrad','fgrad'], default='forward', help="forward = forward pass\nigrad = input gradients\nfgrad = filter gradients")
parser.add_argument('-t', choices=['conv','gflow', 'lowering'], default='conv', help="conv = [conv] convolution (regular for forward, transposed for igrad and fgrad), row stationary dataflow\ngflow = [gflow] New proposed dataflow\nlowering: Lower the convolution into a matrix multiplication (for systolic arrays)")
parser.add_argument('-g', default="1", help="For our gflow dataflow only: it groups vertical PEs (p PEs are merged into just one PE)")

args = parser.parse_args()

# Import the inputs file
spec = importlib.util.spec_from_file_location('fin',args.i)
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


# Template
template = sasimpath+"/compiler/modules/template.py"
print("template: "+str(template))
fout = args.o
if not os.path.isfile(template):
    print("[ERROR] The template does not exist. It is not posible to generate the output.")
    raise
if os.path.isfile(fout):
    print("[WARNING] The output file already exists, overwriting "+fout)

# Original Eyeriss Architecture
# Or new architecture with the mux in a different position
if args.t == "conv":
    _pe_type = c.EYERISS_PE
elif args.t == "gflow":
    _pe_type = c.EYERISS_PE
elif args.t == "lowering" :
    _pe_type = c.SYSTOLIC
else:
    print("[ERROR] Architecture not supported. ")
    sys.exit(-1)

#######################################
# Checking validity of the dimentsions
#######################################
check_validity(f)

info = {
        c.DATAFLOW: args.t,
        c.PASS_T: args.p,
        c.OFM_H: len(f.ofm),
        c.OFM_W: len(f.ofm[0]),
        c.IFM_PAD: f.ifm_pad,
        c.IFM_H: len(f.ifm),
        c.IFM_W: len(f.ifm[0]),
        c.FIL_H: len(f.filter),
        c.FIL_W: len(f.filter[0]),
        c.ERROR_H: len(f.error),
        c.ERROR_W: len(f.error[0]),
        c.GRADIENT_H: len(f.gradient),
        c.GRADIENT_W: len(f.gradient[0]),
        c.STRIDE: f.stride,
        c.BATCH: f.batch,
        # Dataflow mapping parameters
        c.DF_M: f.df_m,
        c.DF_N: f.df_n,
        c.DF_E: f.df_e,
        c.DF_P: f.df_p,
        c.DF_Q: f.df_q,
        c.DF_R: f.df_r,
        c.DF_T: f.df_t
}

if ((args.p == 'forward') and (args.t == 'conv')):
    # Parameters for the forward pass
    _ifm = f.ifm
    _fil = f.filter
    _ofm = f.ofm
    _str = f.stride
    _hw = f.hw
    # Check validity of the input
    print("Forward - conv")
    compiler =  conv(_ifm, _fil, _ofm, _str, _hw, _pe_type, f.num_channels, f.num_filters, f.batch, info)
    signals = compiler.gen_signals()
    print("template: "+str(template))
    dump_to_file(template,fout,signals, info)

elif ((args.p == 'forward') and (args.t == 'lowering')):
    # This is for the Systolic Array
    # Parameters for the forward pass
    _ifm = f.ifm
    _fil = f.filter
    _ofm = f.ofm
    _str = f.stride
    _hw = f.hw

    mat1, mat2 = lowering(_ifm, _fil,_str)
    mat2_t     = matrix_trans(mat2)
    _filter_size = len(_fil)*len(_fil[0])
    print("mat1["+str(len(mat1))+"]["+str(len(mat1[0]))+"]")
    print("mat2["+str(len(mat2))+"]["+str(len(mat2[0]))+"]")

    compiler =  multiply(mat1, mat2_t, _ofm, _filter_size, _hw, _pe_type, f.num_channels, f.num_filters, f.batch)
    signals = compiler.gen_signals()
    dump_to_file(template,fout,signals, info)

elif ((args.p == 'forward') and (args.t == 'gflow')):
    _error    = f.ifm #f.error
    _fil      = f.filter
    _gradient = f.ofm #f.gradient
    _str      = f.stride
    _hw       = f.hw

    print("TODO")
    raise

    # Grouping
    _grouping = int(args.g)
    if _grouping >= 1:
        if _grouping > len(_error):
            print("[ERROR] Grouping has to be less or equal to "+str(len(_error)))
            sys.exit(-1)
        # The filter is transposed and rotated in gflow
        compiler =  gflow_igrad(_error, _fil, _gradient, _str, _hw, _pe_type, _grouping, f.num_channels, f.num_filters, f.batch)
        signals = compiler.gen_signals()
        dump_to_file(template,fout,signals, info)
    else:
        print("[ERROR] Grouping has to be 1 or bigger")
elif ((args.p == 'igrad') and (args.t == 'conv')):
    # Prepare the error matrix
    _error    = f.error
    _fil      = f.filter
    _gradient = f.gradient
    _str      = f.stride
    _hw       = f.hw

    # Rotated Filter
    _fil_t         = matrix_rot(_fil)
    _error_t_conv  = matrix_trans_conv(_error, _str, _fil, _gradient)

    new_str  = 1 # Str > 1 is already taked into account in the data transformations
    check_conv(_error_t_conv, _fil, _gradient, new_str)

    compiler =  conv(_error_t_conv, _fil_t, _gradient, new_str, _hw, _pe_type, f.num_channels, f.num_filters, f.batch, info)
    signals = compiler.gen_signals()
    dump_to_file(template,fout,signals, info)

elif ((args.p == 'igrad') and (args.t == 'gflow')):
    _error    = f.error
    _fil      = f.filter
    _gradient = f.gradient
    _str      = f.stride
    _hw       = f.hw

    # Grouping
    _grouping = int(args.g)

    # Just to check that dimensions are right
    _error_t_conv  = matrix_trans_conv(_error, _str, _fil, _gradient)
    check_conv(_error_t_conv, _fil, _gradient, 1)

    if _grouping >= 1:
        if _grouping > len(_error):
            print("[ERROR] Grouping has to be less or equal to "+str(len(_error)))
            sys.exit(-1)
        # The filter is transposed and rotated in gflow
        compiler =  gflow_igrad(_error, _fil, _gradient, _str, _hw, _pe_type, _grouping, f.num_channels, f.num_filters, f.batch)
        signals = compiler.gen_signals()
        dump_to_file(template,fout,signals, info)
    else:
        print("[ERROR] Grouping has to be 1 or bigger")

elif ((args.p == 'igrad') and (args.t == 'lowering')):
    # Prepare the error matrix
    _error    = f.error
    _fil      = f.filter
    _gradient = f.gradient
    _str      = f.stride
    _hw       = f.hw

    # Rotated Filter
    _fil_t         = matrix_rot(_fil)
    _error_t_conv  = matrix_trans_conv(_error, _str, _fil, _gradient)

    new_str  = 1 # Str > 1 is already taked into account in the data transformations
    check_conv(_error_t_conv, _fil, _gradient, new_str)

    mat1, mat2 = lowering(_error_t_conv, _fil_t, new_str)
    mat2_t     = matrix_trans(mat2)
    _filter_size = len(_fil_t)*len(_fil_t[0])

    compiler =  multiply(mat1, mat2_t, _gradient, _filter_size, _hw, _pe_type, f.num_channels, f.num_filters, f.batch)
    signals = compiler.gen_signals()
    dump_to_file(template,fout,signals, info)

elif ((args.p == 'fgrad') and (args.t == 'conv')):
    check_fgrad(f)
    _inner_pad_errors = inner_padding(f.error, (f.stride-1))
    new_str  = 1 # Str > 1 is already taked into account in the data transformations
    compiler =  conv(f.ifm, _inner_pad_errors, f.filter, new_str, f.hw, _pe_type, f.num_channels, f.num_filters, f.batch, info)
    signals = compiler.gen_signals()
    dump_to_file(template,fout,signals,info)

elif ((args.p == 'fgrad') and (args.t == 'gflow')):
    check_fgrad(f)
    _grouping = int(args.g)
    if _grouping < 0:
        if abs(_grouping) > (len(f.error)*len(f.error[0])):
            print("[ERROR] Negative Grouping has to be bigger or equal to -"+str(len(f.error)*len(f.error[0])))
            sys.exit(-1)
    if _grouping > len(f.filter):
        print("[ERROR] Grouping has to be less or equal to "+str(len(f.filter)))
        sys.exit(-1)
    compiler =  gflow_fgrad(f.ifm, f.error, f.filter, f.stride, f.hw, _pe_type, _grouping, f.num_channels, f.num_filters, f.batch)
    signals = compiler.gen_signals()
    dump_to_file(template,fout,signals,info)

elif ((args.p == 'fgrad') and (args.t == 'lowering')):
    check_fgrad(f)

    mat1, mat2 = lowering_fgrad(f.ifm, f.error, f.stride)
    _filter_size = len(f.error)*len(f.error[0])

    mat2_t = matrix_trans(mat2)

    print("mat1["+str(len(mat1))+"]["+str(len(mat1[0]))+"]")
    print(str(mat1))
    print("mat2["+str(len(mat2))+"]["+str(len(mat2[0]))+"]")
    print(str(mat2))
    compiler =  multiply(mat1, mat2_t, f.filter, _filter_size, f.hw, _pe_type, f.num_channels, f.num_filters, f.batch)
    signals = compiler.gen_signals()
    dump_to_file(template,fout,signals,info)

else:
    print("eval_p: "+str((args.p == 'backward')))
    print("eval_t: "+str((args.t == 'conv')))
    print("[ERROR] Option not valid at the moment: -p "+str(args.p)+" -t "+str(args.t))
    sys.exit(-1)



