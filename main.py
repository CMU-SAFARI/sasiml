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

from confs import import_conf
from hw.array import array
import hw.constants as c
import sys
import os
import numpy as np
import argparse
from argparse import RawTextHelpFormatter

# Parse arguments
parser = argparse.ArgumentParser(description='Run a CNN layer in the spatial architecture', formatter_class=RawTextHelpFormatter)
parser.add_argument('-i', help="Input", required=True)
parser.add_argument('-y', default = 12, help="Hight of the array")
parser.add_argument('-x', default = 14, help="Weight of the array")
parser.add_argument('-q', default='16', help="Quantization (default = 16bits)")

# Type of layer
parser.add_argument('-l', help="Layer type", choices=["conv","pooling","fc"], default="conv")


# Bandwidth of the network
parser.add_argument('-bw_i', default='-1', help="Bandwidth ifm (default = -1 = infinite)")
parser.add_argument('-bw_o', default='-1', help="Bandwidth ofm (default = -1 = infinite)")
parser.add_argument('-bw_f', default='-1', help="Bandwidth filter (default = -1 = infinite)")
# Register File Size (Eyeriss training)
parser.add_argument('-rf_ifm', default='75', help="Size of the IFM RF per PE (in number of elements)") # This has to be larger to support fgrad
parser.add_argument('-rf_fil', default='224', help="Size of the FIL RF per PE (in number of elements)")
parser.add_argument('-rf_psum', default='24', help="Size of the PSUM RF per PE (in number of elements)")

# Global buffer size
parser.add_argument('-gb_bank_size', default='4096', help="Size of a global buffer bank (bytes)")
# GB for IFM/PSUM should be total 27
parser.add_argument('-gb_ifm_psum', default='25', help="Size of the IFM/PSUM Global buffer (in banks)")
# GB filter is only for prefetching, 2 banks, fixed (In Eyeriss)
parser.add_argument('-gb_fil', default='2', help="Size of the FILTER Global buffer (in banks)")

# Core frequency
parser.add_argument('-clock', default='200', help="Core clock (in MHz)") # 200 MHz is Eyeriss clock
parser.add_argument('-t', help="Name of the file to save the memory traces")
args = parser.parse_args()

sasimpath = os.environ['SASIMPATH']

conf_file = args.i
pearray = import_conf(conf_file)

phy_array_w = int(args.x)
phy_array_h = int(args.y)
quantization = int(args.q) # Overwrite the compiler parameter


layer_type = c.CONV_LAYER
if args.l == "pooling":
    layer_type = c.POOLING_LAYER
elif args.l == "fc":
    layer_type = c.FC_LAYER


accelerator_info = {
    # Layer type
    c.LAYER_TYPE: layer_type,
    # Physical dimensions
    c.ARRAY_H: phy_array_h,
    c.ARRAY_W: phy_array_w,
    # Quantization
    c.QUANTIZATION: quantization,
    # Global buffer
    c.GB_BANK_SIZE: int(args.gb_bank_size),
    c.GB_IFM_PSUM: int(args.gb_ifm_psum),
    c.GB_FIL: int(args.gb_fil),
    # Register File
    c.RF_IFM: int(args.rf_ifm),
    c.RF_FIL: int(args.rf_fil),
    c.RF_PSUM: int(args.rf_psum),
    # BW network
    c.IFM_BW: int(args.bw_i),
    c.FIL_BW: int(args.bw_f),
    c.OFM_BW: int(args.bw_o),
    # Clock
    c.CLOCK: int(args.clock),
}

# Memory Traces
if args.t is not None:
    print("Recording memory traces")
    pearray.enable_traces(args.t)

# Results directory
results_dir = sasimpath+"/results"
if not os.path.isdir(results_dir):
    # If it does not exist, create directory
    os.mkdir(results_dir)

# We name the output stats the same as the input, but with another sxtension
fname_stats = conf_file+"-w-"+str(phy_array_w)+"-h-"+str(phy_array_h)+"-bw_i-"+str(accelerator_info[c.IFM_BW])+"-bw_o-"+str(accelerator_info[c.OFM_BW])+"-bw_f-"+str(accelerator_info[c.FIL_BW])
pearray.set_name_stats(fname_stats, results_dir)
pearray.set_hw(accelerator_info)

max_cycles = 90000000000
ccycle = 0 # Current cycle
# Main loop
while ccycle < max_cycles:
    print("cycle: "+str(ccycle))
    if pearray.advance(ccycle) == -1 :
        break
    ccycle+=1
pearray.print_result()
pearray.print_stats()
