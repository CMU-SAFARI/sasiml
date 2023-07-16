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
Constant definition
'''
from numpy import nan
NAN = -1 #nan


# different constants are used to different memories or hardware parameters
ZERO_CLOCKGATE = 100
IFM_MEM    = 0
FILTER_MEM = 1
PSUM_MEM   = 2
MUL        = 3
SUM        = 4
MUX_MUL    = 5
IN_IFM     = 6
IN_FILTER  = 7
IN_PSUM    = 8
OUT_PSUM   = 9
GBUFFER    = 10
IFM_BW     = 100
FIL_BW     = 101
OFM_BW     = 102
LINK       = 103

# Global Buffer
GB_BANK_SIZE  = 300
GB_IFM_PSUM            = 301
GB_IFM            = 302
GB_PSUM            = 303
GB_FIL            = 304
# Register File
RF_IFM  = 305
RF_FIL  = 306
RF_PSUM = 307

GB_RD_IFM        = 308
GB_RD_FIL        = 309
GB_RD_PSUM       = 310
GB_WR_PSUM       = 311
DRAM_RD_IFM      = 312
DRAM_RD_FIL      = 313
DRAM_WR_OFM      = 314


# Physical dimensions
ARRAY_H    = 315
ARRAY_W    = 316
QUANTIZATION = 317

# Core clock
CLOCK       = 318

# BATCH SIZE
BATCH       = 319

# For Systolic Array Only
OUT_IFM    = 20
OUT_FILTER = 21

# Dataflow info passed
DATAFLOW   = 104
PASS_T     = 105
OFM_H      = 106
OFM_W      = 107
IFM_H      = 108
IFM_W      = 109
FIL_H      = 110
FIL_W      = 111
ERROR_H    = 112
ERROR_W    = 113
GRADIENT_H = 114
GRADIENT_W = 115
NUM_CHANNELS = 116
NUM_FILTERS = 117
STRIDE = 118

CONV_LAYER = 300
POOLING_LAYER = 301
FC_LAYER = 302
LAYER_TYPE = 303

#
IFM_PAD = 130

# Dataflow info
DF_M = 119
DF_N = 120
DF_E = 121
DF_P = 122
DF_Q = 123
DF_R = 124
DF_T = 125
DF_E = 126

GB_IFM_CAP = 127
GB_PSUM_CAP = 128

# To get the energies
E_QUEUE = 129
E_SPAD  = 130
E_GB    = 131
E_DRAM  = 132
E_LINK  = 133

# General to refer to all types of memory
MEMORY     = 11

# For configuring the PE like the original Eyeriss paper
# or like our new proposal
EYERISS_PE = 201
NEW_PE     = 202
SYSTOLIC   = 203
