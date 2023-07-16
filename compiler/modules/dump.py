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
Dump variables to file
'''
import modules.constants as s
import numpy as np
import os
import sys
import pickle
import sys
sys.path.insert(0,'..')
import hw.constants as c

# Write the signals into a file
folder = "" # Path of the signals' files

# Create a folder to save the files with different signals
def create_folder(path):
    global folder
    folder = path+".d"
    if not os.path.exists(path+".d"):
        print("Path doesn't exist. Trying to make it.")
        os.makedirs(path+".d")

# Save the array into a file
def array_to_file(folder, name, array):
    output = open(folder+".d/"+name, 'wb')
    pickle.dump(np.array(array), output, protocol=pickle.HIGHEST_PROTOCOL)

# Load the signal's files into variables
def file_to_vars(signal):
    result  = "file = open(\""+folder+"/"+signal+"\",'rb')\n"
    result +=  signal+" = pickle.load(file, encoding='latin1')\n"
    result +=  "file.close\n"
    return result

def dump_to_file(template, out, signals, info):
    print("dump_to_file: "+out)
    create_folder(out) # Create forlder for
    with open(template, "rt") as fin:
        with open(out, "w") as fout:
            for line in fin:
                # Write the size of the memories and alu
                oline = line
                oline = oline.replace("$DATAFLOW",repr(info[c.DATAFLOW]))
                oline = oline.replace("$PASS_T",repr(info[c.PASS_T]))
                oline = oline.replace("$OFM_H",repr(info[c.OFM_H]))
                oline = oline.replace("$OFM_W",repr(info[c.OFM_W]))
                oline = oline.replace("$IFM_PAD",repr(info[c.IFM_PAD]))
                oline = oline.replace("$IFM_H",repr(info[c.IFM_H]))
                oline = oline.replace("$IFM_W",repr(info[c.IFM_W]))
                oline = oline.replace("$FIL_H",repr(info[c.FIL_H]))
                oline = oline.replace("$FIL_W",repr(info[c.FIL_W]))
                oline = oline.replace("$ERROR_H",repr(info[c.ERROR_H]))
                oline = oline.replace("$ERROR_W",repr(info[c.ERROR_W]))
                oline = oline.replace("$GRADIENT_H",repr(info[c.GRADIENT_H]))
                oline = oline.replace("$GRADIENT_W",repr(info[c.GRADIENT_W]))
                oline = oline.replace("$STRIDE",repr(info[c.STRIDE]))

                #Template
                oline = oline.replace("$DF_M",repr(info[c.DF_M]))
                oline = oline.replace("$DF_N",repr(info[c.DF_N]))
                oline = oline.replace("$DF_E",repr(info[c.DF_E]))
                oline = oline.replace("$DF_P",repr(info[c.DF_P]))
                oline = oline.replace("$DF_Q",repr(info[c.DF_Q]))
                oline = oline.replace("$DF_R",repr(info[c.DF_R]))
                oline = oline.replace("$DF_T",repr(info[c.DF_T]))

                oline = oline.replace("$SIZE","size = "+repr(signals[s.HW]))
                oline = oline.replace("$PE_TYPE","pe_type = "+repr(signals[s.PE_TYPE]))
                oline = oline.replace("$ARRAY_W",repr(signals[s.ARRAY_W]))
                oline = oline.replace("$ARRAY_H",repr(signals[s.ARRAY_H]))

                # insert MEM_IFM_WR
                if "$MEM_IFM_WR" in oline:
                    if info[c.DATAFLOW] == "lowering":
                        str_mem_ifm_wr = ""
                    else:
                        str_mem_ifm_wr = ""
                        for h in range(signals[s.ARRAY_H]):
                            for w in range(signals[s.ARRAY_W]):
                                array_to_file(out, "pe_"+str(h)+"_"+str(w)+"_mem_ifm_wr", signals[s.MEM_IFM_WR][h][w])
                                str_mem_ifm_wr += file_to_vars("pe_"+str(h)+"_"+str(w)+"_mem_ifm_wr")
                    try:
                        oline = oline.replace("$MEM_IFM_WR", str_mem_ifm_wr)
                    except CaughtException as e:
                        print("Exception: "+str(e))
                        raise

                # insert MEM_IFM_INIT
                if "$MEM_IFM_INIT" in oline:
                    str_mem_ifm_init = ""
                    for h in range(signals[s.ARRAY_H]):
                        for w in range(signals[s.ARRAY_W]):
                            array_to_file(out, "pe_"+str(h)+"_"+str(w)+"_mem_ifm_init", signals[s.MEM_IFM_INIT][h][w])
                            str_mem_ifm_init += file_to_vars("pe_"+str(h)+"_"+str(w)+"_mem_ifm_init")
                    try:
                        oline = oline.replace("$MEM_IFM_INIT", str_mem_ifm_init)
                    except CaughtException as e:
                        print("Exception: "+str(e))
                        raise

                # insert MEM_IFM_RD
                if "$MEM_IFM_RD" in oline:
                    str_mem_ifm_rd = ""
                    for h in range(signals[s.ARRAY_H]):
                        for w in range(signals[s.ARRAY_W]):
                            array_to_file(out, "pe_"+str(h)+"_"+str(w)+"_mem_ifm_rd", signals[s.MEM_IFM_RD][h][w])
                            str_mem_ifm_rd += file_to_vars("pe_"+str(h)+"_"+str(w)+"_mem_ifm_rd")
                    oline = oline.replace("$MEM_IFM_RD", str_mem_ifm_rd)

                # insert MEM_FILTER_WR
                if "$MEM_FILTER_WR" in oline:
                    if info[c.DATAFLOW] == "lowering":
                        str_mem_filter_wr = ""
                    else:
                        str_mem_filter_wr = ""
                        for h in range(signals[s.ARRAY_H]):
                            for w in range(signals[s.ARRAY_W]):
                                array_to_file(out, "pe_"+str(h)+"_"+str(w)+"_mem_filter_wr", signals[s.MEM_FILTER_WR][h][w])
                                str_mem_filter_wr += file_to_vars("pe_"+str(h)+"_"+str(w)+"_mem_filter_wr")
                    oline = oline.replace("$MEM_FILTER_WR", str_mem_filter_wr)

                # insert MEM_FILTER_INIT
                if "$MEM_FILTER_INIT" in oline:
                    str_mem_filter_init = ""
                    for h in range(signals[s.ARRAY_H]):
                        for w in range(signals[s.ARRAY_W]):
                            array_to_file(out, "pe_"+str(h)+"_"+str(w)+"_mem_filter_init", signals[s.MEM_FILTER_INIT][h][w])
                            str_mem_filter_init += file_to_vars("pe_"+str(h)+"_"+str(w)+"_mem_filter_init")
                    try:
                        oline = oline.replace("$MEM_FILTER_INIT", str_mem_filter_init)
                    except CaughtException as e:
                        print("Exception: "+str(e))
                        raise

                # insert MEM_FILTER_RD
                if "$MEM_FILTER_RD" in oline:
                    if info[c.DATAFLOW] == "lowering":
                        str_mem_filter_rd = ""
                    else:
                        str_mem_filter_rd = ""
                        for h in range(signals[s.ARRAY_H]):
                            for w in range(signals[s.ARRAY_W]):
                                array_to_file(out, "pe_"+str(h)+"_"+str(w)+"_mem_filter_rd", signals[s.MEM_FILTER_RD][h][w])
                                str_mem_filter_rd += file_to_vars("pe_"+str(h)+"_"+str(w)+"_mem_filter_rd")
                    oline = oline.replace("$MEM_FILTER_RD", str_mem_filter_rd)

                # insert MEM_PSUM_WR
                if "$MEM_PSUM_WR" in oline:
                    str_mem_psum_wr = ""
                    for h in range(signals[s.ARRAY_H]):
                        for w in range(signals[s.ARRAY_W]):
                            array_to_file(out, "pe_"+str(h)+"_"+str(w)+"_mem_psum_wr", signals[s.MEM_PSUM_WR][h][w])
                            str_mem_psum_wr += file_to_vars("pe_"+str(h)+"_"+str(w)+"_mem_psum_wr")
                    oline = oline.replace("$MEM_PSUM_WR", str_mem_psum_wr)

                # insert MEM_PSUM_RD
                if "$MEM_PSUM_RD" in oline:
                    if info[c.DATAFLOW] == "lowering":
                        str_mem_psum_rd = ""
                    else:
                        str_mem_psum_rd = ""
                        for h in range(signals[s.ARRAY_H]):
                            for w in range(signals[s.ARRAY_W]):
                                array_to_file(out, "pe_"+str(h)+"_"+str(w)+"_mem_psum_rd", signals[s.MEM_PSUM_RD][h][w])
                                str_mem_psum_rd += file_to_vars("pe_"+str(h)+"_"+str(w)+"_mem_psum_rd")
                    oline = oline.replace("$MEM_PSUM_RD", str_mem_psum_rd)

                # insert MUX_SEQ
                if "$MUX_SEQ" in oline:
                    if info[c.DATAFLOW] == "lowering":
                        str_mux_seq = ""
                    else:
                        str_mux_seq = ""
                        for h in range(signals[s.ARRAY_H]):
                            for w in range(signals[s.ARRAY_W]):
                                array_to_file(out, "pe_"+str(h)+"_"+str(w)+"_mux_seq", signals[s.MUX_SEQ][h][w])
                                str_mux_seq += file_to_vars("pe_"+str(h)+"_"+str(w)+"_mux_seq")
                    oline = oline.replace("$MUX_SEQ", str_mux_seq)

                # insert OUT_PSUM
                if "$OUT_PSUM" in oline:
                    if info[c.DATAFLOW] == "lowering":
                        str_out_psum = ""
                    else:
                        str_out_psum = ""
                        for h in range(signals[s.ARRAY_H]):
                            for w in range(signals[s.ARRAY_W]):
                                array_to_file(out, "pe_"+str(h)+"_"+str(w)+"_out_psum", signals[s.OUT_PSUM][h][w])
                                str_out_psum += file_to_vars("pe_"+str(h)+"_"+str(w)+"_out_psum")
                    oline = oline.replace("$OUT_PSUM", str_out_psum)

                # insert OFM_SEQ
                if "$OFM_SEQ" in oline:
                    str_ofm_seq = ""
                    for h in range(signals[s.ARRAY_H]):
                        for w in range(signals[s.ARRAY_W]):
                            array_to_file(out, "pe_"+str(h)+"_"+str(w)+"_ofm", signals[s.OFM_SEQ][h][w])
                            str_ofm_seq += file_to_vars("pe_"+str(h)+"_"+str(w)+"_ofm")
                    oline = oline.replace("$OFM_SEQ", str_ofm_seq)

                # insert IFM
                oline = oline.replace("$IFM_STREAM","ifm = "+repr(signals[s.IFM]))
                # insert FILTER
                oline = oline.replace("$FILTER_STREAM","filter = "+repr(signals[s.FILTER]))

                # insert NUM_CHANNELS
                oline = oline.replace("$NUM_CHANNELS","num_channels = "+repr(signals[s.NUM_CHANNELS]))
                # insert NUM_FILTERS
                oline = oline.replace("$NUM_FILTERS","num_filters = "+repr(signals[s.NUM_FILTERS]))
                # insert BATCH
                oline = oline.replace("$BATCH","batch = "+repr(signals[s.BATCH]))

                # MULTICAST IFM
                oline = oline.replace("$MULTICAST_IFM","multicast_ifmap = "+repr(signals[s.MULTICAST_IFM]))
                # MULTICAST FILTER
                oline = oline.replace("$MULTICAST_FILTER","multicast_filter = "+repr(signals[s.MULTICAST_FILTER]))


                # insert IFM_SEQ_MULTICAST
                oline = oline.replace("$IFM_SEQ_MULTICAST","ifm_seq_multicast = "+repr(signals[s.IFM_SEQ_MULTICAST]))

                # insert FILTER_SEQ_MULTICAST
                oline = oline.replace("$FILTER_SEQ_MULTICAST","filter_seq_multicast = "+repr(signals[s.FILTER_SEQ_MULTICAST]))

                # insert PE instantiation
                if "$PES" in oline:
                    str_pes = ""
                    for h in range(signals[s.ARRAY_H]):
                        for w in range(signals[s.ARRAY_W]):
                            pe_id = "pe_"+str(h)+"_"+str(w)

                            if signals[s.PE_TYPE] == c.SYSTOLIC:
                                if info[c.PASS_T] == "fgrad":
                                    str_pes += pe_id+" = pe([], "+pe_id+"_mem_ifm_rd, [], [], "+pe_id+"_mem_psum_wr, [], [], "+pe_id+"_ofm, "+pe_id+"_ofm, size, \"PE"+str(h)+str(w)+"\", pe_type, debug, debug_pe["+str(h)+"]["+str(w)+"],"+str(h)+","+str(w)+","+str(len(signals[s.FILTER])+10)+","+str((info[c.OFM_H] + (info[c.OFM_H]-1)*(info[c.STRIDE]-1))*(info[c.OFM_W] + (info[c.OFM_W]-1)*(info[c.STRIDE]-1)))+")\n"
                                else:
                                    str_pes += pe_id+" = pe([], "+pe_id+"_mem_ifm_rd, [], [], "+pe_id+"_mem_psum_wr, [], [], "+pe_id+"_ofm, "+pe_id+"_ofm, size, \"PE"+str(h)+str(w)+"\", pe_type, debug, debug_pe["+str(h)+"]["+str(w)+"],"+str(h)+","+str(w)+","+str(len(signals[s.FILTER])+10)+","+str(info[c.FIL_H]*info[c.FIL_W])+")\n"
                            else:
                                str_pes += pe_id+" = pe("+pe_id+"_mem_ifm_wr, "+pe_id+"_mem_ifm_rd, "+pe_id+"_mem_filter_wr, "+pe_id+"_mem_filter_rd, "+pe_id+"_mem_psum_wr, "+pe_id+"_mem_psum_rd, "+pe_id+"_mux_seq, "+pe_id+"_ofm, "+pe_id+"_out_psum, size, \"PE"+str(h)+str(w)+"\", pe_type, debug, debug_pe["+str(h)+"]["+str(w)+"],"+str(h)+","+str(w)+")\n"
                    oline = oline.replace("$PES", str_pes)

                if "$PE_INIT_IFM" in oline:
                    str_init_ifm = ""
                    for h in range(signals[s.ARRAY_H]):
                        for w in range(signals[s.ARRAY_W]):
                            pe_id = "pe_"+str(h)+"_"+str(w)
                            str_init_ifm += pe_id+".load_reg(c.IFM_MEM,"+pe_id+"_mem_ifm_init)\n"
                    oline = oline.replace("$PE_INIT_IFM", str_init_ifm)
                if "$PE_INIT_FILTER" in oline:
                    str_init_filter = ""
                    for h in range(signals[s.ARRAY_H]):
                        for w in range(signals[s.ARRAY_W]):
                            pe_id = "pe_"+str(h)+"_"+str(w)
                            str_init_filter += pe_id+".load_reg(c.FILTER_MEM,"+pe_id+"_mem_filter_init)\n"
                    oline = oline.replace("$PE_INIT_FILTER", str_init_filter)

                # insert PE in the array
                if "$PEARRAY" in oline:
                    str_pearray = ""
                    for h in range(signals[s.ARRAY_H]):
                        for w in range(signals[s.ARRAY_W]):
                            str_pearray += "pearray.add_pe("+str(h)+","+str(w)+", pe_"+str(h)+"_"+str(w)+")\n"
                    oline = oline.replace("$PEARRAY", str_pearray)


                # Debug PE
                if "$DEBUG_PE" in oline:
                    str_debug_pe = "debug_pe = ["
                    for h in range(signals[s.ARRAY_H]):
                        str_debug_pe += "["
                        for w in range(signals[s.ARRAY_W]):
                            if w == 1:
                                str_debug_pe += "False,"
                            else:
                                str_debug_pe += "False,"
                        str_debug_pe += "],\n"
                    str_debug_pe += "]"
                    oline = oline.replace("$DEBUG_PE", str_debug_pe)

                fout.write(oline)
