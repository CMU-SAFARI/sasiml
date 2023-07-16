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

from numpy import nan
from hw.energy_model import active_energy
from hw.energy_model import idle_energy
import hw.constants as c
import math
import logging, sys

import numpy as np
import queue as Q


STRIP_MINED_INT = False

class opt(object):
    def __init__(self, priority, param):
        self.priority = priority
        self.param = param
        return
    def __lt__(self, other):
        return self.priority < other.priority
    def __eq__(self, other):
        return (self.priority == other.priority)

class memory_model:
    '''
    Calculates:
        1) Size of the local registers required
        2) Bandwidth required for each data type
        3) DRAM/HMC bandwidth required. Estimates also the number of memory channels
    '''

    def __init__(self, dinfo, ainfo, pe_sets_w, pe_sets_h, pe_set_w, pe_set_h ):
       logging.basicConfig(stream=sys.stderr, level=logging.CRITICAL) # Managing the prints
       self.calculate(dinfo, ainfo, pe_sets_w, pe_sets_h, pe_set_w, pe_set_h)

    def calculate_accesses(self, dinfo, ainfo, n, p, t, r, q):
        '''
        Calculate the data accesses to GBUF and DRAM
        '''
        if dinfo[c.DATAFLOW] == "conv" or dinfo[c.DATAFLOW] == "gflow" or dinfo[c.DATAFLOW] == "lowering":
            # row stationary dataflow
            # Data Read from the Global Buffer (taking into account the data reuse)
            # Overall, the computation of this layer uses eight processing passes. Each group of ifmaps is read from DRAM once, stored in the GLB, and reused in two consecutive passes with total eight filters to generate eight ofmap channels. However, this also requires the GLB to store psums from two consecutive passes so they do not go to DRAM. In this case, the GLB needs to store m = 8 ofmap channels. Each filter weight is read from DRAM into the PE array once for every four passes.
            ########################################
            # Global Buffer
            ########################################

            gb_rd_ifm = (self.ifm_h * self.ifm_w) * self.num_channels * self.batch # total IFM maps from memory
            # Additionally, if we can fit more than one PE set per array, we can reuse across PE sets
            gb_rd_ifm *= self.num_filters / (p * t) # additional reads to the same IFMmaps between processing passes
            gb_rd_ifm *= ainfo[c.QUANTIZATION] / 8  # additional reads to the same IFMmaps between processing passes
            gb_rd_fil  = 0
            # The PSUMS generated for each 2D conv is: (OFM_H * OFM_W) * p
            gb_rd_psum = self.ofm_h * self.ofm_w * p * n * t # OFMs generated in each pass
            gb_rd_psum *= self.num_filters / (p * t)  # additional reads to the same IFMmaps between processing passes
            gb_rd_psum *= self.num_channels / (r * q)  # additional reads to the same IFMmaps between processing passes
            gb_rd_psum *= self.batch / n # additional reads to the same IFMmaps between processing passes
            gb_rd_psum *= ainfo[c.QUANTIZATION] / 8 # Additional reads to the same IFMmaps between processing passes
            # PSUM reads and writes are the same
            gb_wr_psum = gb_rd_psum

            # VERIFICATION: OFM PER PASS
            psums_per_pass = self.ofm_h * self.ofm_w
            psums_per_pass *= n * p * r * t
            psums_per_pass *= ainfo[c.QUANTIZATION] / 8 # Additional reads to the same IFMmaps between processing passes
            # VERIFICATION: IFM PER PASS
            ifms_per_pass = self.ifm_h * self.ifm_w
            ifms_per_pass *= n * q * r
            ifms_per_pass *= ainfo[c.QUANTIZATION] / 8 # Additional reads to the same IFMmaps between processing passes

            ########################################
            # DRAM accesses
            ########################################
            # Total number of reads from DRAM, for each memory type
            dram_rd_ifm = (self.ifm_h * self.ifm_w) * self.num_channels * self.batch  # How to calculate this?
            # calculate how many accesses go to DRAM
            dram_rd_ifm *= ainfo[c.QUANTIZATION] / 8   # additional reads to the same IFMmaps between processing passes

            dram_rd_fil = (self.fil_h * self.fil_w) * self.num_channels * self.num_filters
            dram_rd_fil *= self.batch / n  # additional reads to the same IFMmaps between processing passes
            dram_rd_fil *= ainfo[c.QUANTIZATION] / 8   # additional reads to the same IFMmaps between processing passes

            dram_wr_ofm = self.ofm_w * self.ofm_h * self.num_filters *  self.batch
            dram_wr_ofm *= ainfo[c.QUANTIZATION] / 8   # additional reads to the same IFMmaps between processing passes

        else:
            logging.critical("DATAFLOW: "+dinfo[c.DATAFLOW])
            logging.critical("PASS_T: "+dinfo[c.PASS_T])
            raise
            assert(0)

        return gb_rd_ifm, gb_rd_fil, gb_rd_psum, gb_wr_psum, dram_rd_fil, dram_rd_ifm, dram_wr_ofm


    def next_power_of_2(self,x):
        return 1 if x == 0 else 2**(x - 1).bit_length()
    def previous_power_of_2(self,x):
        return 1 if x == 0 else 2**((x - 1).bit_length()-1)


    def param(self, dinfo, ainfo, pe_sets_w, pe_sets_h, pe_set_w, pe_set_h):
        '''
        Calculate de parameters for Row-stationary
        '''
        # Calculate the worst cases
        # Q
        a_df_q = math.floor(self.prf_ifm/self.fil_h) # This is max channels for the IFMs
        # P
        a_df_p = int(self.prf_psum) # filters per PE set
        a_df_p = self.previous_power_of_2(a_df_p)# Power of two
        # N: can not be larger than the batch
        a_df_n = math.floor(self.prf_ifm/(self.fil_h)) # IFMs are streamed. we only need to maintain "fil_h" ifms, independently of the number of channels
        if a_df_n > self.batch:
            a_df_n = self.batch
        # R
        pe_sets = int(pe_sets_h * pe_sets_w)
        if pe_sets == 0:
            pe_sets = 1 # Strip-mined (e.g., CONV1 in Alexnet)
        max_r  = pe_sets
        a_df_r = max_r
        # T
        logging.debug("df_n: "+str(self.df_n)+", a_df_n: "+str(a_df_n))
        logging.debug("df_p: "+str(self.df_p)+", a_df_p: "+str(a_df_p))
        logging.debug("df_q: "+str(self.df_q)+", a_df_q: "+str(a_df_q))
        logging.debug("df_r: "+str(self.df_r)+", a_df_r: "+str(a_df_r))
        logging.debug("df_t: "+str(self.df_t)+", a_df_t: "+str(a_df_t))
        pqueue = Q.PriorityQueue()
        strip_mined = pe_sets_w * pe_sets_h
        if strip_mined > 1:
            strip_mined = 1
        while True:
            for gb_ifm_banks in range(int(self.gb_ifm_psum_banks - 1), 0, -1):
                gb_psum_banks = self.gb_ifm_psum_banks - gb_ifm_banks
                gb_psum_cap = gb_psum_banks * self.gb_bank_size
                gb_ifm_cap  = gb_ifm_banks * self.gb_bank_size
                for r in range(a_df_r,0,-1):
                    for q in range(a_df_q,0,-1):
                        for i_p in range(int(math.log2(a_df_p)),0,-1):
                            p = 2**(i_p)
                            for n in range(a_df_n,0,-1):
                                for t in range(a_df_t,0,-1):

                                    if STRIP_MINED_INT:
                                        strip_mined = 1/math.ceil(1/strip_mined)

                                    eval_1 = math.floor(gb_ifm_cap/(self.ifm_h * self.ifm_w * n * q))
                                    eval_2 = math.floor(gb_psum_cap/((self.ofm_h * self.ofm_w * strip_mined)  * n * p * r))
                                    eval_3 = r * t
                                    eval_4 = self.fil_h * q * p
                                    if dinfo[c.PASS_T] == "igrad" and dinfo[c.DATAFLOW] == "gflow":
                                        eval_41 = (self.fil_h/self.stride) * q * r
                                        eval_42 = (self.fil_w * n)
                                    elif dinfo[c.PASS_T] == "fgrad" and dinfo[c.DATAFLOW] == "gflow":
                                        eval_41 = q * r
                                        eval_42 = n
                                    else:
                                        # always True
                                        eval_41 = 0
                                        eval_42 = 0

                                    eval_5 = self.ifm_h * (self.ifm_w * strip_mined) * n * q * r * self.quantization / 8 # IFMs per pass
                                    eval_6 = self.ofm_h * (self.ofm_w * strip_mined) * n * p * r * t * self.quantization / 8 # PSUMS per pass
                                    eval_7 = self.num_channels % (q) # has to be a multiple of the number of channels
                                    if (r <= eval_1) and (t <= eval_2) and (pe_sets == eval_3) and (self.prf_fil >= eval_4) and (self.prf_ifm >= eval_41) and (self.prf_psum >= eval_42) and (gb_ifm_cap >= eval_5) and (gb_psum_cap >= eval_6) and (eval_7 == 0):
                                        param = {
                                            c.DF_N: n,
                                            c.DF_Q: q,
                                            c.DF_R: r,
                                            c.DF_P: p,
                                            c.DF_T: t,
                                            c.GB_IFM_CAP: gb_ifm_cap,
                                            c.GB_PSUM_CAP: gb_psum_cap
                                        }

                                        # The spad accesses are the same for all combinations
                                        gb_rd_ifm, gb_rd_fil, gb_rd_psum, gb_wr_psum, dram_rd_fil, dram_rd_ifm, dram_wr_ofm = self.calculate_accesses(dinfo, ainfo, n, p, t, r, q)
                                        value_gb   = 6*(gb_rd_ifm + gb_rd_fil + gb_rd_psum + gb_wr_psum)
                                        value_dram = 200*(dram_rd_fil + dram_rd_ifm + dram_wr_ofm)
                                        value_spad = 0
                                        value_link = 0
                                        value = value_spad + value_link + value_gb + value_dram
                                        pqueue.put(opt(value,param))
            if not pqueue.empty():
                winner = pqueue.get()
                logging.debug("winner:" + str(winner.priority))
                logging.debug("      winner:" + str(winner.param))
                break
            else:
                logging.critical("NO POSSIBLE CONVINATION of PARAMETERS")
                logging.critical("1 > gb_ifm_capacity/(ifm_h*ifm_w*n*q) = "+str(1)+"/("+str(self.ifm_h)+"*"+str(self.ifm_w)+"*1*1)")
                logging.critical("1 > gb_psum_capacity/(ofm_h*ofm_w*n*q*r) = "+str(1)+"/("+str(self.ofm_h)+"*"+str(self.ofm_w)+"*1*1*1)")
                logging.critical("pe_sets = "+str(pe_sets))
                logging.critical("strip_mined = "+str(strip_mined))
                n=1
                q=1
                p=1
                r=1
                t=1
                gb_ifm_cap  = (self.ifm_h * self.ifm_w * n * q)
                gb_psum_cap = (self.gb_ifm_psum_banks * self.gb_bank_size) - gb_ifm_cap
                eval_1 = math.floor(gb_ifm_cap/(self.ifm_h * self.ifm_w * n * q))
                eval_2 = math.floor(gb_psum_cap*strip_mined/(self.ofm_h * self.ofm_w * n * p * r))
                eval_3 = r * t
                eval_4 = self.fil_h * q * p
                eval_5 = self.ifm_h * (self.ifm_w) * n * q * r * self.quantization / 8 # IFMs per pass
                eval_6 = self.ofm_h * (self.ofm_w * strip_mined) * n * p * r * t * self.quantization / 8 # PSUMS per pass
                eval_7 = self.num_channels % (q) # has to be a multiple of the number of channpe_sets:els
                logging.critical("strip_mined: "+str(strip_mined)+" pe_sets_w: "+str(pe_sets_w)+" pe_sets_h: "+str(pe_sets_h)+" pe_set_w: "+str(pe_set_w)+" pe_set_h: "+str(pe_set_h))
                logging.critical("eval_1: "+str(eval_1)+" eval_2: "+str(eval_2)+" eval_3: "+str(eval_3)+" eval_4: "+str(eval_4)+" eval_5: "+str(eval_5)+" eval_6: "+str(eval_6)+" eval_7: "+str(eval_7))
                logging.critical("r: "+str(r)+" t: "+str(t)+" pe_sets: "+str(pe_sets)+" prf_fil: "+str(self.prf_fil)+" gb_ifm_cap: "+str(gb_ifm_cap)+" gb_psum_cap: "+str(gb_psum_cap))

                if strip_mined > 1:
                    strip_mined = 1
                else:
                    strip_mined = strip_mined/2
                pe_sets = strip_mined
                #raise

        logging.debug("df_n: "+str(self.df_n)+", a_df_n: "+str(winner.param[c.DF_N]))
        logging.debug("df_p: "+str(self.df_p)+", a_df_p: "+str(winner.param[c.DF_P]))
        logging.debug("df_q: "+str(self.df_q)+", a_df_q: "+str(winner.param[c.DF_Q]))
        logging.debug("df_t: "+str(self.df_t)+", a_df_t: "+str(winner.param[c.DF_T]))
        logging.debug("df_r: "+str(self.df_r)+", a_df_r: "+str(winner.param[c.DF_R]))
        logging.debug("gb_ifm_cap: "+str(winner.param[c.GB_IFM_CAP]/1024)+" KB, gb_psum_cap: "+str(winner.param[c.GB_PSUM_CAP]/1024)+" KB")

        return winner.param


    def calculate(self, dinfo, ainfo, pe_sets_w, pe_sets_h, pe_set_w, pe_set_h):
        '''
        Calculate the parameters
        '''
        self.quantization = ainfo[c.QUANTIZATION] # To know how many parameters we can fit
        # Global buffer calculations
        self.gb_bank_size     = ainfo[c.GB_BANK_SIZE]
        self.gb_ifm_psum      = ainfo[c.GB_IFM_PSUM]
        self.gb_fil           = ainfo[c.GB_FIL]
        self.pgb_ifm_psum  = self.gb_ifm_psum * self.gb_bank_size
        self.pgb_fil  = self.gb_fil * self.gb_bank_size

        # Physical Register File
        self.prf_ifm  = ainfo[c.RF_IFM]
        self.prf_fil  = ainfo[c.RF_FIL]
        self.prf_psum = ainfo[c.RF_PSUM]
        logging.debug("prf_ifm: "+str(self.prf_ifm))
        logging.debug("prf_fil: "+str(self.prf_fil))
        logging.debug("prf_psum: "+str(self.prf_psum))

        # Bandwidth requirements to memory
        # The global input network (GIN) is optimized for a single-cycle multicast from the GLB to a group of PEs that receive the same filter weight, ifmap value, or psum
        self.bw_ifm = ainfo[c.IFM_BW]
        self.bw_fil = ainfo[c.FIL_BW]
        self.bw_ofm = ainfo[c.OFM_BW]

        # Specified in the config file
        self.gb_ifm_psum_capacity = ainfo[c.GB_IFM_PSUM] * ainfo[c.GB_BANK_SIZE] * 8 / ainfo[c.QUANTIZATION]
        self.gb_ifm_psum_banks    = ainfo[c.GB_IFM_PSUM]
        logging.debug("ifm_psum_capacity: "+str(self.gb_ifm_psum_capacity/1024)+" K-elements")
        # We need to read all the filters again
        logging.debug("pe_sets_w: "+str(pe_sets_w))
        logging.debug("pe_sets_h: "+str(pe_sets_h))

        if dinfo[c.DATAFLOW] == "conv" and dinfo[c.PASS_T] == "forward" :
            # Row stationary
            self.ifm_h = dinfo[c.IFM_H] - dinfo[c.IFM_PAD]
            self.ifm_w = dinfo[c.IFM_W] - dinfo[c.IFM_PAD]
            self.ofm_h = dinfo[c.OFM_H]
            self.ofm_w = dinfo[c.OFM_W]
            self.fil_h = dinfo[c.FIL_H]
            self.fil_w = dinfo[c.FIL_W]
            self.num_filters  = dinfo[c.NUM_FILTERS]
            self.num_channels = dinfo[c.NUM_CHANNELS]
            self.batch = dinfo[c.BATCH]
            self.df_n = dinfo[c.DF_N]
            self.df_p = dinfo[c.DF_P]
            self.df_t = dinfo[c.DF_T]
            self.df_r = dinfo[c.DF_R]
            self.df_q = dinfo[c.DF_Q]

            param = self.param(dinfo, ainfo, pe_sets_w, pe_sets_h, pe_set_w, pe_set_h)
            self.df_n = param[c.DF_N] #a_df_n
            self.df_p = param[c.DF_P] #a_df_p
            self.df_t = param[c.DF_T] #a_df_t
            self.df_r = param[c.DF_R] #a_df_r
            self.df_q = param[c.DF_Q] #a_df_q

            self.pgb_ifm = param[c.GB_IFM_CAP]
            self.pgb_psum = param[c.GB_PSUM_CAP]
            #raise
            # Configured SPAD sizes for this especific layer
            #Since only a sliding window of data has to be retained at a time, the required spad capacity depends only on the filter row size (S) but not the ifmap row size (W), and is equal to: 1) S for a row of filter weights; 2) S for a sliding window of ifmap values; and 3) 1 for the psum accumulation. In AlexNet, for example, possible values for S are 11 (layer CONV1), 5 (layer CONV2), and 3 (layers CONV3–CONV5). Therefore, the minimum spad capacity required for filter, ifmap, and psum is 11, 11, and 1, respectively, to support all layers.
            self.rf_ifm = self.fil_w * self.df_q
            self.rf_psum = self.df_p
            self.rf_fil = self.fil_h * self.df_q * self.df_p


        elif dinfo[c.DATAFLOW] == "conv" and dinfo[c.PASS_T] == "igrad":
            self.ifm_h = dinfo[c.OFM_H]
            self.ifm_w = dinfo[c.OFM_W]
            self.ofm_h = dinfo[c.IFM_H] - dinfo[c.IFM_PAD]
            self.ofm_w = dinfo[c.IFM_W] - dinfo[c.IFM_PAD]
            self.fil_h = dinfo[c.FIL_W]
            self.fil_w = dinfo[c.FIL_H]
            self.num_filters  =  dinfo[c.NUM_CHANNELS]
            self.num_channels = dinfo[c.NUM_FILTERS] # OFMAP CHANNELS
            self.batch = dinfo[c.BATCH]
            self.df_n  = dinfo[c.DF_N]
            self.df_p  = dinfo[c.DF_P]
            self.df_t  = dinfo[c.DF_T]
            self.df_r  = dinfo[c.DF_R]
            self.df_q  = dinfo[c.DF_Q]

            param = self.param(dinfo, ainfo, pe_sets_w, pe_sets_h, pe_set_w, pe_set_h)
            self.df_n = param[c.DF_N] #a_df_n
            self.df_p = param[c.DF_P] #a_df_p
            self.df_t = param[c.DF_T] #a_df_t
            self.df_r = param[c.DF_R] #a_df_r
            self.df_q = param[c.DF_Q] #a_df_q

            self.pgb_ifm = param[c.GB_IFM_CAP]
            self.pgb_psum = param[c.GB_PSUM_CAP]

            # Configured SPAD sizes for this especific layer
            self.rf_ifm = self.fil_w * self.df_q
            self.rf_psum = self.df_p
            self.rf_fil = self.fil_h * self.df_q * self.df_p

        elif dinfo[c.DATAFLOW] == "conv" and dinfo[c.PASS_T] == "fgrad":
            # TODO:
            self.ifm_h = dinfo[c.IFM_H] - dinfo[c.IFM_PAD]
            self.ifm_w = dinfo[c.IFM_W] - dinfo[c.IFM_PAD]
            self.ofm_h = dinfo[c.FIL_H]
            self.ofm_w = dinfo[c.FIL_W]
            self.fil_h = dinfo[c.OFM_W]
            self.fil_w = dinfo[c.OFM_H]
            self.num_filters  =  dinfo[c.NUM_FILTERS]
            self.num_channels = dinfo[c.BATCH] # OFMAP CHANNELS
            self.batch = dinfo[c.NUM_CHANNELS]
            self.df_n  = dinfo[c.DF_Q] * dinfo[c.DF_R]
            self.df_p  = dinfo[c.DF_P]
            self.df_t  = dinfo[c.DF_T]
            self.df_r  = 1
            self.df_q  = dinfo[c.DF_N]

            param = self.param(dinfo, ainfo, pe_sets_w, pe_sets_h, pe_set_w, pe_set_h)
            self.df_n = param[c.DF_N] #a_df_n
            self.df_p = param[c.DF_P] #a_df_p
            self.df_t = param[c.DF_T] #a_df_t
            self.df_r = param[c.DF_R] #a_df_r
            self.df_q = param[c.DF_Q] #a_df_q

            self.pgb_ifm = param[c.GB_IFM_CAP]
            self.pgb_psum = param[c.GB_PSUM_CAP]

            # Configured SPAD sizes for this especific layer
            self.rf_ifm = self.fil_w * self.df_q
            self.rf_psum = self.df_p
            self.rf_fil = self.fil_h * self.df_q * self.df_p

        if dinfo[c.DATAFLOW] == "lowering" and dinfo[c.PASS_T] == "forward" :
            # Row stationary
            self.ifm_h = dinfo[c.IFM_H] - dinfo[c.IFM_PAD]
            self.ifm_w = dinfo[c.IFM_W] - dinfo[c.IFM_PAD]
            self.ofm_h = dinfo[c.OFM_H]
            self.ofm_w = dinfo[c.OFM_W]
            self.fil_h = dinfo[c.FIL_H]
            self.fil_w = dinfo[c.FIL_W]
            self.num_filters  = dinfo[c.NUM_FILTERS]
            self.num_channels = dinfo[c.NUM_CHANNELS]
            self.batch = dinfo[c.BATCH]
            self.df_n = dinfo[c.DF_N]
            self.df_p = dinfo[c.DF_P]
            self.df_t = dinfo[c.DF_T]
            self.df_r = dinfo[c.DF_R]
            self.df_q = dinfo[c.DF_Q]

            param = self.param(dinfo, ainfo, pe_sets_w, pe_sets_h, pe_set_w, pe_set_h)
            self.df_n = param[c.DF_N] #a_df_n
            self.df_p = param[c.DF_P] #a_df_p
            self.df_t = param[c.DF_T] #a_df_t
            self.df_r = param[c.DF_R] #a_df_r
            self.df_q = param[c.DF_Q] #a_df_q

            self.pgb_ifm = param[c.GB_IFM_CAP]
            self.pgb_psum = param[c.GB_PSUM_CAP]
            #raise
            # Configured SPAD sizes for this especific layer
            #Since only a sliding window of data has to be retained at a time, the required spad capacity depends only on the filter row size (S) but not the ifmap row size (W), and is equal to: 1) S for a row of filter weights; 2) S for a sliding window of ifmap values; and 3) 1 for the psum accumulation. In AlexNet, for example, possible values for S are 11 (layer CONV1), 5 (layer CONV2), and 3 (layers CONV3–CONV5). Therefore, the minimum spad capacity required for filter, ifmap, and psum is 11, 11, and 1, respectively, to support all layers.
            self.rf_ifm = 1
            self.rf_psum = 1
            self.rf_fil = 1

        elif dinfo[c.DATAFLOW] == "lowering" and dinfo[c.PASS_T] == "igrad":
            self.ifm_h = dinfo[c.OFM_H]
            self.ifm_w = dinfo[c.OFM_W]
            self.ofm_h = dinfo[c.IFM_H] - dinfo[c.IFM_PAD]
            self.ofm_w = dinfo[c.IFM_W] - dinfo[c.IFM_PAD]
            self.fil_h = dinfo[c.FIL_W]
            self.fil_w = dinfo[c.FIL_H]
            self.num_filters  =  dinfo[c.NUM_CHANNELS]
            self.num_channels = dinfo[c.NUM_FILTERS] # OFMAP CHANNELS
            self.batch = dinfo[c.BATCH]
            self.df_n  = dinfo[c.DF_N]
            self.df_p  = dinfo[c.DF_P]
            self.df_t  = dinfo[c.DF_T]
            self.df_r  = dinfo[c.DF_R]
            self.df_q  = dinfo[c.DF_Q]

            param = self.param(dinfo, ainfo, pe_sets_w, pe_sets_h, pe_set_w, pe_set_h)
            self.df_n = param[c.DF_N] #a_df_n
            self.df_p = param[c.DF_P] #a_df_p
            self.df_t = param[c.DF_T] #a_df_t
            self.df_r = param[c.DF_R] #a_df_r
            self.df_q = param[c.DF_Q] #a_df_q

            self.pgb_ifm = param[c.GB_IFM_CAP]
            self.pgb_psum = param[c.GB_PSUM_CAP]

            # Configured SPAD sizes for this especific layer
            self.rf_ifm = 1
            self.rf_psum = 1
            self.rf_fil = 1

        elif dinfo[c.DATAFLOW] == "lowering" and dinfo[c.PASS_T] == "fgrad":
            # TODO:
            self.ifm_h = dinfo[c.IFM_H] - dinfo[c.IFM_PAD]
            self.ifm_w = dinfo[c.IFM_W] - dinfo[c.IFM_PAD]
            self.ofm_h = dinfo[c.FIL_H]
            self.ofm_w = dinfo[c.FIL_W]
            self.fil_h = dinfo[c.OFM_W]
            self.fil_w = dinfo[c.OFM_H]
            self.num_filters  =  dinfo[c.NUM_FILTERS]
            self.num_channels = dinfo[c.BATCH] # OFMAP CHANNELS
            self.batch = dinfo[c.NUM_CHANNELS]
            self.df_n  = dinfo[c.DF_Q] * dinfo[c.DF_R]
            self.df_p  = dinfo[c.DF_P]
            self.df_t  = dinfo[c.DF_T]
            self.df_r  = 1
            self.df_q  = dinfo[c.DF_N]

            param = self.param(dinfo, ainfo, pe_sets_w, pe_sets_h, pe_set_w, pe_set_h)
            self.df_n = param[c.DF_N]
            self.df_p = param[c.DF_P]
            self.df_t = param[c.DF_T]
            self.df_r = param[c.DF_R]
            self.df_q = param[c.DF_Q]

            self.pgb_ifm = param[c.GB_IFM_CAP]
            self.pgb_psum = param[c.GB_PSUM_CAP]

            # Configured SPAD sizes for this especific layer
            self.rf_ifm = 1
            self.rf_psum = 1
            self.rf_fil = 1

        elif dinfo[c.DATAFLOW] == "gflow" and dinfo[c.PASS_T] == "igrad":
            self.ifm_h = dinfo[c.OFM_H]
            self.ifm_w = dinfo[c.OFM_W]
            self.ofm_h = dinfo[c.IFM_H] - dinfo[c.IFM_PAD]
            self.ofm_w = dinfo[c.IFM_W] - dinfo[c.IFM_PAD]
            self.fil_h = dinfo[c.FIL_W]
            self.fil_w = dinfo[c.FIL_H]
            self.num_filters  =  dinfo[c.NUM_CHANNELS]
            self.num_channels = dinfo[c.NUM_FILTERS] # OFMAP CHANNELS
            self.batch = dinfo[c.BATCH]
            self.df_n  = dinfo[c.DF_N]
            self.df_p  = dinfo[c.DF_Q]
            self.df_t  = dinfo[c.DF_R]
            self.df_r  = dinfo[c.DF_T]
            self.df_q  = dinfo[c.DF_P]
            self.stride = dinfo[c.STRIDE]

            param = self.param(dinfo, ainfo, pe_sets_w, pe_sets_h, pe_set_w, pe_set_h)
            self.df_n = param[c.DF_N] #a_df_n
            self.df_p = param[c.DF_P] #a_df_p
            self.df_t = param[c.DF_T] #a_df_t
            self.df_r = param[c.DF_R] #a_df_r
            self.df_q = param[c.DF_Q] #a_df_q

            self.pgb_ifm = param[c.GB_IFM_CAP]
            self.pgb_psum = param[c.GB_PSUM_CAP]

            # Configured SPAD sizes for this especific layer
            self.rf_ifm  = math.ceil(self.fil_h/self.stride) #fil_h     # FIXME: This depends on the stride
            self.rf_psum = self.fil_w    # This is always the case in all cases I analyzed. Not totally sure why
            self.rf_fil  = 1        # We only need to maintain one at a time

            # Multicast Network characteristic
            # Row labels
            self.row_multicast_groups = self.ofm_h
            self.row_multicast_labels = math.ceil(self.fil_h/self.stride)
            self.row_multicast_label_bits = math.ceil(math.log(self.row_multicast_groups,2))

            # Column labels
            self.column_multicast_groups = self.ofm_w
            self.column_multicast_labels = math.ceil(self.fil_w/self.stride)
            self.column_multicast_label_bits = math.ceil(math.log(self.column_multicast_groups,2))

            print(">Row multicast groups: "+str(self.row_multicast_groups))
            print(">Row multicast labels: "+str(self.row_multicast_labels))
            print(">Row multicast label bits: "+str(self.row_multicast_label_bits)+" bits")
            print(">Column multicast groups: "+str(self.column_multicast_groups))
            print(">Column multicast labels: "+str(self.column_multicast_labels))
            print(">Column multicast label bits: "+str(self.column_multicast_label_bits)+" bits")

        elif dinfo[c.DATAFLOW] == "gflow" and dinfo[c.PASS_T] == "fgrad":
            self.ifm_h = dinfo[c.IFM_H] - dinfo[c.IFM_PAD]
            self.ifm_w = dinfo[c.IFM_W] - dinfo[c.IFM_PAD]
            self.ofm_h = dinfo[c.FIL_H]
            self.ofm_w = dinfo[c.FIL_W]
            self.fil_h = dinfo[c.OFM_W]
            self.fil_w = dinfo[c.OFM_H]
            self.num_filters  = dinfo[c.NUM_FILTERS]
            self.num_channels = dinfo[c.NUM_CHANNELS] # OFMAP CHANNELS
            self.batch = dinfo[c.BATCH]
            self.df_n  = dinfo[c.DF_N]
            self.df_p  = dinfo[c.DF_P]
            self.df_t  = dinfo[c.DF_T]
            self.df_r  = dinfo[c.DF_R]
            self.df_q  = dinfo[c.DF_Q]
            self.stride = dinfo[c.STRIDE]

            param = self.param(dinfo, ainfo, pe_sets_w, pe_sets_h, pe_set_w, pe_set_h)
            self.df_n = param[c.DF_N] #a_df_n
            self.df_p = param[c.DF_P] #a_df_p
            self.df_t = param[c.DF_T] #a_df_t
            self.df_r = param[c.DF_R] #a_df_r
            self.df_q = param[c.DF_Q] #a_df_q

            self.pgb_ifm = param[c.GB_IFM_CAP]
            self.pgb_psum = param[c.GB_PSUM_CAP]

            # Configured SPAD sizes for this especific layer
            self.rf_ifm  = 1     # This does not depend on the stride
            self.rf_psum = 1     # This is always the case in all cases I analyzed. Not totally sure why
            self.rf_fil  = 1     # We only need to maintain one at a time

            # Multicast characteristic
            if self.ofm_w > self.fil_w:
                # Is there such a case? in that case, the calculations are more complex
                raise

            # Row labels
            self.row_multicast_groups = self.ofm_h*2 - self.stride
            self.row_multicast_labels = math.ceil(self.ofm_h/self.stride)
            if self.row_multicast_groups >= 1:
                self.row_multicast_label_bits = math.ceil(math.log(self.row_multicast_groups,2)) #ifm_h - (ofm_h + stride -2)
            else:
                self.row_multicast_label_bits = 1

            # Column labels
            self.column_multicast_groups = self.ofm_w*2 - self.stride
            self.column_multicast_labels = math.ceil(self.ofm_w/self.stride)

            if self.column_multicast_groups >= 1:
                self.column_multicast_label_bits = math.ceil(math.log(self.column_multicast_groups,2))
            else:
                self.column_multicast_label_bits = 1
            print(">Row multicast groups: "+str(self.row_multicast_groups))
            print(">Row multicast labels: "+str(self.row_multicast_labels))
            print(">Row multicast label bits: "+str(self.row_multicast_label_bits)+" bits")
            print(">Column multicast groups: "+str(self.column_multicast_groups))
            print(">Column multicast labels: "+str(self.column_multicast_labels))
            print(">Column multicast label bits: "+str(self.column_multicast_label_bits)+" bits")

        ##############################################################

        if dinfo[c.DATAFLOW] == "conv" or dinfo[c.DATAFLOW] == "gflow" or dinfo[c.DATAFLOW] == "lowering":
            # row stationary dataflow
            print("Calculating ... Row Stationary Dataflow")

            # Data Read from the Global Buffer (taking into account the data reuse)
            # Overall, the computation of this layer uses eight processing passes. Each group of ifmaps is read from DRAM once, stored in the GLB, and reused in two consecutive passes with total eight filters to generate eight ofmap channels. However, this also requires the GLB to store psums from two consecutive passes so they do not go to DRAM. In this case, the GLB needs to store m = 8 ofmap channels. Each filter weight is read from DRAM into the PE array once for every four passes.
            ########################################
            # Global Buffer
            ########################################

            self.gb_rd_ifm = (self.ifm_h * self.ifm_w) * self.num_channels * self.batch# total IFM maps from memory
            # Additionally, if we can fit more than one PE set per array, we can reuse across PE sets
            self.gb_rd_ifm *= self.num_filters / (self.df_p * self.df_t) # additional reads to the same IFMmaps between processing passes
            self.gb_rd_ifm *= ainfo[c.QUANTIZATION] / 8   # additional reads to the same IFMmaps between processing passes
            self.gb_rd_fil  = 0
            # The PSUMS generated for each 2D conv is: (OFM_H * OFM_W) * p
            self.gb_rd_psum = self.ofm_h * self.ofm_w * self.df_p * self.df_n * self.df_t # OFMs generated in each pass
            self.gb_rd_psum *= self.num_filters / (self.df_p*self.df_t)  # additional reads to the same IFMmaps between processing passes
            self.gb_rd_psum *= self.num_channels / (self.df_r*self.df_q) # additional reads to the same IFMmaps between processing passes
            self.gb_rd_psum *= self.batch / self.df_n # additional reads to the same IFMmaps between processing passes
            self.gb_rd_psum *= ainfo[c.QUANTIZATION] / 8 # Additional reads to the same IFMmaps between processing passes
            # PSUM reads and writes are the same
            self.gb_wr_psum = self.gb_rd_psum

            # VERIFICATION: OFM PER PASS
            self.psums_per_pass = self.ofm_h * self.ofm_w
            self.psums_per_pass *= self.df_n * self.df_p * self.df_r * self.df_t
            self.psums_per_pass *= ainfo[c.QUANTIZATION] / 8 # Additional reads to the same IFMmaps between processing passes
            # VERIFICATION: IFM PER PASS
            self.ifms_per_pass = self.ifm_h * self.ifm_w
            self.ifms_per_pass *= self.df_n * self.df_q * self.df_r
            self.ifms_per_pass *= ainfo[c.QUANTIZATION] / 8 # Additional reads to the same IFMmaps between processing passes

            ########################################
            # DRAM accesses
            ########################################

            # Total number of reads from DRAM, for each memory type
            self.dram_rd_ifm = (self.ifm_h * self.ifm_w) * self.num_channels * self.batch  # How to calculate this?
            # calculate how many accesses go to DRAM
            self.dram_rd_ifm *= ainfo[c.QUANTIZATION] / 8   # additional reads to the same IFMmaps between processing passes

            self.dram_rd_fil = (self.fil_h * self.fil_w) * self.num_channels * self.num_filters
            self.dram_rd_fil *= self.batch / self.df_n  # additional reads to the same IFMmaps between processing passes
            self.dram_rd_fil *= ainfo[c.QUANTIZATION] / 8   # additional reads to the same IFMmaps between processing passes

            self.dram_wr_ofm = self.ofm_w * self.ofm_h * self.num_filters *  self.batch
            self.dram_wr_ofm *= ainfo[c.QUANTIZATION] / 8   # additional reads to the same IFMmaps between processing passes

        else:
            print("DATAFLOW: "+dinfo[c.DATAFLOW])
            print("PASS_T: "+dinfo[c.PASS_T])
            raise
            assert(0)

    def get_bandwidth(self, net):
        '''
        Get the bandwidth requirements for a specific data network (ifm, ofm, fil, psum)
        '''

    def get_rfsize(self, mem):
        '''
        Get the register file size for a specific data type (ifm, ofm, fil, psum)
        '''
        if mem == c.IFM_MEM:
            return self.rf_ifm
        elif mem == c.FILTER_MEM:
            return self.rf_fil
        elif mem == c.PSUM_MEM:
            return self.rf_psum
        else:
            assert(0)

    def get_param(self, param):
        '''
        Get the param
        '''
        if param == c.DF_N:
            return self.df_n
        elif param == c.DF_P:
            return self.df_p
        elif param == c.DF_Q:
            return self.df_q
        elif param == c.DF_R:
            return self.df_r
        elif param == c.DF_T:
            return self.df_t
        else:
            raise

    def get_prfsize(self, mem):
        '''
        Get the register file size for a specific data type (ifm, ofm, fil, psum)
        '''
        if mem == c.IFM_MEM:
            return self.prf_ifm
        elif mem == c.FILTER_MEM:
            return self.prf_fil
        elif mem == c.PSUM_MEM:
            return self.prf_psum
        else:
            assert(0)

    def get_gbsize(self, mem):
        '''
        Get the register file size for a specific data type (ifm, ofm, fil, psum)
        '''
        if mem == c.GB_IFM:
            return self.pgb_ifm
        elif mem == c.GB_FIL:
            return self.pgb_fil
        elif mem == c.GB_PSUM:
            return self.pgb_psum
        else:
            assert(0)

    def get_gballoc(self, mem):
        '''
        Get the register file size for a specific data type (ifm, ofm, fil, psum)
        '''
        if mem == c.GB_IFM:
            return self.ifms_per_pass
        elif mem == c.GB_FIL:
            return 0
        elif mem == c.GB_PSUM:
            return self.psums_per_pass
        else:
            assert(0)

    def get_gb_accesses(self, dtype):
        '''
        Get the data accessed from the global buffer, per data type, in bytes
        '''
        if dtype == c.GB_RD_IFM:
            return self.gb_rd_ifm
        elif dtype == c.GB_RD_FIL:
            return self.gb_rd_fil
        elif dtype == c.GB_RD_PSUM:
            return self.gb_rd_psum
        elif dtype == c.GB_WR_PSUM:
            return self.gb_wr_psum
        else:
            assert(0)


    def get_dram_accesses(self, dtype):
        '''
        Get the data accessed from the global buffer, per data type, in bytes
        '''
        if dtype == c.DRAM_RD_IFM:
            return self.dram_rd_ifm
        elif dtype == c.DRAM_RD_FIL:
            return self.dram_rd_fil
        elif dtype == c.DRAM_WR_OFM:
            return self.dram_wr_ofm
        else:
            assert(0)
