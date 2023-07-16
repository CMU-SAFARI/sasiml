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
Definition of input, weights and output
'''
import sys
sys.path.insert(0,'..')
import hw.constants as c

hw = {
    c.ZERO_CLOCKGATE: True, # clock gating when multiply by zero
    c.GBUFFER:        156000,
    c.IFM_MEM:        52012,   # capacity
    c.FILTER_MEM:     52224,
    c.PSUM_MEM:       52024, #24,
    c.IN_IFM:         64,
    c.IN_FILTER:      64,
    c.IN_PSUM:        64,
    c.OUT_PSUM:       64,
    c.OUT_IFM:        64, # For systolic array only
    c.OUT_FILTER:     64, # For systolic array only
    c.MUL:            2,   # pipeline stages
    c.SUM:            1,   # pipeline stage
    c.IFM_BW:         1,
    c.FIL_BW:         1,
    c.OFM_BW:         1
}

num_channels = 256
num_filters  = 384

# Forward pass
ifm_h = 13
ifm_w = 13
ifm = [['' for h in range(ifm_h)] for w in range(ifm_w)]
for h in range(ifm_h):
    for w in range(ifm_w):
        ifm[h][w] = "i"+str(h)+"-"+str(w)

ofm_h = 6
ofm_w = 6
ofm = [['' for h in range(ofm_h)] for w in range(ofm_w)]
for h in range(ofm_h):
    for w in range(ofm_w):
        ofm[h][w] = "o"+str(h)+"-"+str(w)


# Common to forward and backward pass
# Stride
stride =2

#  Filter
filter_h = 3
filter_w = 3
filter = [[None for h in range(filter_h)] for w in range(filter_w)]
for h in range(filter_h):
    for w in range(filter_w):
        filter[h][w] = "w"+str(h)+"-"+str(w)


# Backward pass
# Errors
error = [[None for h in range(ofm_h)] for w in range(ofm_w)]
for h in range(ofm_h):
    for w in range(ofm_w):
        error[h][w] = "e"+str(h)+"-"+str(w)

# Gradients
gradient = [[None for h in range(ifm_h)] for w in range(ifm_w)]
for h in range(ifm_h):
    for w in range(ifm_w):
        gradient[h][w] = "g"+str(h)+"-"+str(w)


