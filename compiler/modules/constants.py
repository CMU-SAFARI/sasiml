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
Constants
'''
# Important constants
from numpy import nan
NAN = -1#nan

# Signals
MEM_IFM_WR    = 0
MEM_IFM_RD    = 1
MEM_FILTER_WR = 2
MEM_FILTER_RD = 3
MEM_PSUM_WR   = 4
MEM_PSUM_RD   = 5
MUX_SEQ       = 6
OUT_PSUM      = 7
OFM_SEQ       = 8
#
IFM_BW        = 9
FIL_BW        = 10
OFM_BW        = 100
#
HW            = 11
PE_TYPE       = 12

#
NUM_CHANNELS  = 101
NUM_FILTERS   = 102
BATCH         = 103


# Dataflow info
DF_M = 119
DF_N = 120
DF_E = 121
DF_P = 122
DF_Q = 123
DF_R = 124
DF_T = 125

# Other definitions
EYERISS       = 201
NEWARCH       = 202
# New!!!
SYSTOLIC      = 203

ARRAY_W       = 15
ARRAY_H       = 16

IFM           = 17
FILTER        = 18

IFM_SEQ_MULTICAST = 19
FILTER_SEQ_MULTICAST = 20

MULTICAST_FILTER = 21
MULTICAST_IFM = 22

MEM_IFM_INIT = 23
MEM_FILTER_INIT = 24

