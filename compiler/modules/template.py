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
from hw.array import array
from hw.gbuffer import gbuffer
from hw.pe import pe
from numpy import nan
import hw.constants as c
import numpy as np
import pickle

# PRINT cycle-by-cycle state
debug = {
    c.IFM_MEM:    False,
    c.FILTER_MEM: False,
    c.PSUM_MEM:   True,
    c.MUL:        True,
    c.SUM:        True,
    c.MUX_MUL:    True,
    c.IN_IFM:     False,
    c.IN_FILTER:  False,
    c.IN_PSUM:    True,
    c.OUT_PSUM:   True,
    c.GBUFFER:    False,
    c.OUT_IFM:    False,
    c.OUT_FILTER: False
}




# Enable or disable debug prints for each PE
$DEBUG_PE

# size of each memory, see hw.constants for explanation
$SIZE

$NUM_CHANNELS
$NUM_FILTERS
$BATCH

# PE type
# EYERISS_PE: (I-MUX1: IPSUM, I-MUX0: MUL)
# NEW PE:     (I-MUX1: IPSUM, I-MUX0: MEM_PSUM)
# the new PE (different accumulation datapath) is not used so far
$PE_TYPE

# Quantization info
quantization = 8 # 8 bit quantization

# PE array
dataflow_info = {
    c.DATAFLOW: $DATAFLOW,
    c.PASS_T:   $PASS_T,
    # Data sizes
    c.OFM_H:    $OFM_H,
    c.OFM_W:    $OFM_W,
    c.IFM_PAD:  $IFM_PAD,
    c.IFM_H:    $IFM_H,
    c.IFM_W:    $IFM_W,
    c.FIL_H:    $FIL_H,
    c.FIL_W:    $FIL_W,
    c.ERROR_H:  $ERROR_H,
    c.ERROR_W:  $ERROR_W,
    c.GRADIENT_H: $GRADIENT_H,
    c.GRADIENT_W: $GRADIENT_W,
    c.NUM_CHANNELS: num_channels,
    c.NUM_FILTERS: num_filters,
    c.BATCH: batch,
    c.STRIDE: $STRIDE,
    # From eyeriss paper
    c.DF_M: $DF_M, # NUmber of ofmap channels stored in the global buffer
    c.DF_N: $DF_N, # Number of ifmaps used in a processing pass
    c.DF_E: $DF_E, # Width of the PE Set (strip-mined if necessary)
    c.DF_P: $DF_P, # number of filters processed by a PE set
    c.DF_Q: $DF_Q, # number of channels processed by a PE set
    c.DF_R: $DF_R, # number of PE sets that process different channels in the PE array
    c.DF_T: $DF_T  # number of PE sets that process different filters in the PE array
}
pearray = array('Eyeriss', $ARRAY_H, $ARRAY_W, size, dataflow_info, quantization, pe_type)

#pearray.add_num_channels(num_channels)
#pearray.add_num_filters(num_filters)


# MICROPROGRAMMING THE PEs (cycle by cycle)
# the NaN for memory indices means there is no memory access in that cycle
# note that mem bw is controlled in essence in the compiler:
# a list inside the list indicates multiple accesses (to simulate larger bandwidth) that cycle
# the hw simulator checks how many accesses there are
$MEM_IFM_INIT
$MEM_IFM_WR
$MEM_IFM_RD
$MEM_FILTER_INIT
$MEM_FILTER_WR
$MEM_FILTER_RD
$MEM_PSUM_WR
$MEM_PSUM_RD
$MUX_SEQ
$OUT_PSUM

$OFM_SEQ

# DATA definitions
$IFM_STREAM
$FILTER_STREAM

## Global buffer
gb = gbuffer(ifm, filter, size[c.GBUFFER], debug[c.GBUFFER])


# Configuration of the PE array: NoC
# Multicast groups filter (we assing labels to each PE)
$MULTICAST_FILTER

$MULTICAST_IFM

pearray.conf_noc_filter(multicast_filter)
pearray.conf_noc_ifmap(multicast_ifmap)

# DATA distribution into the PEs (multicast groups). Cycle by cycle
$IFM_SEQ_MULTICAST

$FILTER_SEQ_MULTICAST

pearray.conf_datamov(ifm_seq_multicast, filter_seq_multicast)

pearray.conf_gbuffer(gb)

# Instanciating the PEs
$PES
$PE_INIT_IFM
$PE_INIT_FILTER

# Adding the PEs to the array
$PEARRAY


