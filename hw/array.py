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
from . import gbuffer
import math
from numpy import nan
import hw.constants as c
from hw.queue import queue
from hw.energy_model import dram_access_energy
from hw.memory_model import memory_model
from hw.energy_model import active_energy
from hw.DRAMPower import DRAMPower

STRIP_MINED_INT = False
# Use DRAMPower
DRAMPOWER = True

# Keep performance stats
class stats_performance:
    def __init__(self):
        self.fullpipe_cycles = 0
        self.total_mul_by_zero = 0
        self.total_mul = 0
        self.txdata_cycles = 0
        self.pe_set_w = 0
        self.pe_set_h = 0
        self.total_pe_set = 0 # Number of PEs in a PE set
        self.ifm_elements = 0
        self.ofm_elements = 0
        self.filter_elements = 0
        self.total_simulated_cycles = 0
        self.cycles_gbuffer2array = 0
        self.cycles_array2gbuffer = 0
        self.cycles_per_ofm = 0 # This is the number of cycles that takes to calculate one OFM when the pipeline is full
        self.max_array2gbuffer_bw = 0
        self.total_mul = 0
        self.total_mul_by_zero = 0
        self.gbuff_reads = 0
        self.gbuff_reads_zero = 0
        self.gbuff_writes = 0
        self.gbuff_writes_zero = 0

        # For Systolic array only
        self.ifm_artificial_padding_remove = 0
        self.fil_artificial_padding_remove = 0


# Keep energy stats
class stats_energy:
    def __init__(self):
        # ENERGY STATS
        self.e_total = 0
        self.e_mem_ifm = 0
        self.e_mem_fil = 0
        self.e_mem_psum = 0
        self.e_mul = 0
        self.e_sum = 0
        self.e_noc = 0
        self.e_gbuff = 0


class array(object):
    def __init__(self, name, h, w, hw, dataflow_info, quantization, pe_type):
        self.name = name
        self.w = w
        self.h = h
        self.matrix = [[ None for x in range(w)] for y in range(h) ]
        self.cycle = 0
        # Stats
        self.max_array_to_gbuffer_bw = 0
        self.ifm_bw = hw[c.IFM_BW]
        self.fil_bw = hw[c.FIL_BW]
        self.ofm_bw = hw[c.OFM_BW]
        self.pe_type = pe_type

        self.dataflow_info = dataflow_info

        self.quantization = quantization

        self.running = True # Determine if we reach the max cycle (from the input files)
        self.ofm_buffer_queues = [[queue("", 256, False,False) for x in range(w)] for y in range(h)] # To model ofm bandwidth

        self.ofm_init = 0
        self.ofm_end  = 0

        self.num_ofms = 0
        self.fname_stats = "SASIM_stats"
        self.enable_traces = False # This is enabled in the command line
        self.f_traces = None

        # Set the limit to finish the simulation
        # For finishing the systolic array execution
        self.finish = 0
        if self.dataflow_info[c.PASS_T] == "forward":
            self.finish = self.dataflow_info[c.OFM_W]*self.dataflow_info[c.OFM_H]
        elif self.dataflow_info[c.PASS_T] == "igrad":
            self.finish = self.dataflow_info[c.IFM_W]*self.dataflow_info[c.IFM_H]
        elif self.dataflow_info[c.PASS_T] == "fgrad":
            self.finish = self.dataflow_info[c.FIL_W]*self.dataflow_info[c.FIL_H]
        else:
            assert(0)

        self.pe_sets_w = 0
        self.pe_sets_h = 0
        self.pe_set_w = 0
        self.pe_set_h = 0

    def set_hw(self, accelerator_info):
        '''
        Set accelerator info
        '''
        self.accelerator_info = accelerator_info
        self.quantization = accelerator_info[c.QUANTIZATION]
        self.phy_array_h = accelerator_info[c.ARRAY_H]
        self.phy_array_w = accelerator_info[c.ARRAY_W]
        self.bw_i = accelerator_info[c.IFM_BW]
        self.bw_o = accelerator_info[c.OFM_BW]
        self.bw_f = accelerator_info[c.FIL_BW]
        self.layer_type = accelerator_info[c.LAYER_TYPE]

        # This is the case where we dont need to load the filter from memory
        # Is a max operation, no filter
        if self.accelerator_info[c.LAYER_TYPE] == c.POOLING_LAYER and self.dataflow_info[c.PASS_T] == "forward":
            for w in range(self.w):
                for h in range(self.h):
                    self.matrix[h][w].set_pooling_forward()


    def enable_traces(self, f_traces):
        '''
        Enable the memory Traces
        '''
        self.enable_traces = True
        self.f_traces = open(f_traces,"w")

    def set_name_stats(self, fname, dname):
        '''
        Set the name of the output file stats
        the name should not include extension
        '''
        self.fname_stats = fname # file name
        self.dname_stats = dname # directory name


    def add_pe(self, h, w, pe):
        '''
        Add one PE to the array
        '''
        assert w < self.w
        assert h < self.h
        self.matrix[h][w] = pe

        if self.matrix[h][w].pe_type == c.SYSTOLIC:
            if w == self.w-1:
                self.matrix[h][w].set_limit_right()
            if h == self.h-1:
                self.matrix[h][w].set_limit_bottom()



    def conf_ofm(self, ofm, pe_ofm, c_ofm, ofm_bw):
        '''
        Configure which PEs are writing to memory and when
        '''
        self.ofm    = ofm
        self.pe_ofm = pe_ofm
        self.c_ofm  = c_ofm
        self.ofm_bw = ofm_bw

    def conf_noc_filter(self, filter):
        '''
        Configure the filter NoC
        '''
        for h in range(self.h):
            for w in range(self.w):
                if type(filter[h][w]) is not type([]):
                    print("[ERROR] noc_filter should be a list")
                    raise
        self.noc_filter = filter

    def conf_noc_ifmap(self, ifmap):
        '''
        Configure the ifmap NoC
        '''
        for h in range(self.h):
            for w in range(self.w):
                if type(ifmap[h][w]) is not type([]):
                    print("[ERROR] noc_ifmap should be a list")
                    raise
        self.noc_ifmap = ifmap

    def conf_gbuffer(self, gbuf):
        '''
        Configure the on-chip global buffer
        '''
        self.gbuf = gbuf

    def __check_bw(self,seq, bw):
        '''
        Check if it is on the bandwidth limits
        '''
        for cycle in range(len(seq)):
            if type(seq[cycle]) is type([]):
                if len(seq[cycle]) > bw:
                    print("[ERROR] BW limits exceded")
                    raise

    def conf_datamov(self, ifmap_seq, filter_seq):
        '''
        Determines the dataflow from the global buffer to the PE array
        '''
        if len(ifmap_seq) != len(filter_seq):
            print("ifmap_seq: "+str(len(ifmap_seq))+", filter_seq: "+str(len(filter_seq)))
            raise
        self.__check_bw(ifmap_seq,self.ifm_bw)
        self.__check_bw(filter_seq,self.fil_bw)
        self.ifmap_seq = ifmap_seq
        self.filter_seq = filter_seq


    def debug(self):
        '''
        Print debuging information
        '''
        return

    def print_result(self):
        '''
        Print results (ofm)
        '''
        print("---Results---")
        self.gbuf.print_ofm()

    def __cycles(self, seq):
        '''
        Count the number of cycles in transmiting data
        '''
        cycles = 0
        for i in range(len(seq)):
            if seq[i] is not c.NAN and seq[i] is not [c.NAN]:
                cycles += 1
        return cycles

    def __elements(self, seq):
        '''
        Count the number of elements in transmiting data
        '''
        cycles = 0
        for i in range(len(seq)):
            if type(seq[i]) is type([]) :
                for a in range(len(seq[i])):
                    if seq[i] is not c.NAN and seq[i] is not [c.NAN] and seq[i] is not "":
                        cycles += 1
            elif seq[i] is not c.NAN and seq[i] is not [c.NAN] and seq[i] is not "":
                cycles +=1
        return cycles


    # Merge in igrad calculation
    def merge2_h(self, h_factor, pe_set_h):
        # We have to group several vertical virtual PEs in one physical PE
        #h_factor has to be power of 2
        # Improve this algorithm, we are been very conservative
        h_factor *= 2
        merged = int(pe_set_h / 2)
        unmerged = pe_set_h % 2
        pe_set_h = merged + unmerged
        return h_factor,pe_set_h

    def calc_total(self, p, e, f_stats):
        '''
        0) Consider channels number of filters, etc.all
        1) We need to know the type of dataflow for mapping the virtual PEs to the Physical structure
        '''

        print("\n---------FINAL STATS (PHYSICAL MAPPING)----------------")
        print("Dimensions of the accelerator: "+str(self.phy_array_w)+"(w)x"+str(self.phy_array_h)+"(h) ("+str(self.phy_array_h*self.phy_array_w)+" PEs)")
        print("Bandwidths of the accelerator: ")
        print("    ifm: " + str(self.bw_i if self.bw_i != -1 else "infinite"))
        print("    filter: " + str(self.bw_f if self.bw_f !=-1 else "infinite"))
        print("    ofm: " + str(self.bw_o if self.bw_o != -1 else "infinite"))
        print("Num channels: "+str(self.dataflow_info[c.NUM_CHANNELS]))
        print("Num filters: "+str(self.dataflow_info[c.NUM_FILTERS]))
        print("Batch: "+str(self.dataflow_info[c.BATCH]))
        dataflow = self.dataflow_info[c.DATAFLOW]
        pass_t   = self.dataflow_info[c.PASS_T]
        print("dataflow: "+str(dataflow))
        print("pass_t: "+str(pass_t))

        num_conv = self.dataflow_info[c.NUM_CHANNELS] * self.dataflow_info[c.NUM_FILTERS] * self.dataflow_info[c.BATCH]

        self.pe_set_w = self.w
        self.pe_set_h = self.h

        if STRIP_MINED_INT:
            if self.pe_set_w > self.phy_array_w:
                w = math.ceil(self.pe_set_w/float(self.phy_array_w))
                self.pe_sets_w = 1/float(w)
            else:
                self.pe_sets_w = int(self.phy_array_w / self.pe_set_w)

            if self.pe_set_h > self.phy_array_h:
                h = math.ceil(self.pe_set_h/self.phy_array_h)
                self.pe_sets_h = 1/float(h)
            else:
                self.pe_sets_h = int(self.phy_array_h / self.pe_set_h)
        else:
            self.pe_sets_w = self.phy_array_w / self.pe_set_w
            self.pe_sets_h = self.phy_array_h / self.pe_set_h

        print("pe_sets_w: "+str(self.pe_sets_w))
        print("pe_sets_h: "+str(self.pe_sets_h))

        # TODO: fix this
        # Alexnet CONV1 does not fit in eyeriss. Readjust the dimensions according to the Eyeriss paper
        self.num_pe_sets_per_array = self.pe_sets_w * self.pe_sets_h

        if STRIP_MINED_INT and self.num_pe_sets_per_array < 1:
            self.num_pe_sets_per_array = 1/(math.ceil(1/self.num_pe_sets_per_array))

        print(str(self.num_pe_sets_per_array)+" PE set fits into the physical array")
        total_cycles = (p.fullpipe_cycles *num_conv)/self.num_pe_sets_per_array
        total_energy = e.e_total*num_conv
        ocu_pes = self.num_pe_sets_per_array * self.pe_set_w * self.pe_set_h
        ocupation = ocu_pes/(self.phy_array_w*self.phy_array_h)

        # print the stats
        print("[PE]")
        print("    Num PE sets per array: "+str(self.num_pe_sets_per_array))
        c_per_phype = total_cycles / (self.phy_array_w*self.phy_array_h)
        c_per_ocu_pe = total_cycles / (self.num_pe_sets_per_array*self.w*self.h)
        print("    Cycles per physical PE: "+str(c_per_phype)) # This only includes the occuped PEs
        print("    Cycles per used PE: "+str(c_per_ocu_pe)) # This only includes the occuped PEs
        print("    Used PEs: "+str(ocu_pes)+" (" + str(ocupation*100)+"% ocupation)")
        parameters = self.dataflow_info[c.NUM_FILTERS] * self.dataflow_info[c.FIL_H] * self.dataflow_info[c.FIL_W] * self.dataflow_info[c.NUM_CHANNELS]
        ifms = self.dataflow_info[c.NUM_CHANNELS] * self.dataflow_info[c.IFM_H] * self.dataflow_info[c.IFM_W] * self.dataflow_info[c.BATCH]
        ofms = self.dataflow_info[c.NUM_FILTERS] * self.dataflow_info[c.OFM_H] * self.dataflow_info[c.OFM_W] * self.dataflow_info[c.BATCH]

        memory = (parameters + ifms + ofms) * self.quantization
        edram = dram_access_energy()
        dram_energy = memory/512 * edram
        print("[Memory]")
        print("    Parameters: "+str(parameters))
        print("    ifms: "+str(ifms))
        print("    ofms: "+str(ofms))
        print("    DRAM footprint: "+str(memory/8) +" ("+str(memory/(8*1024*1024))+" MB)")
        print("         IFM: "+str(ifms*self.quantization/8) +" B ("+str(ifms*self.quantization/(8*1024*1024))+" MB)")
        print("         FIL: "+str(parameters*self.quantization/8) +" B ("+str(parameters*self.quantization/(8*1024*1024))+" MB)")
        print("         OFM: "+str(ofms*self.quantization/8) +" B ("+str(ofms*self.quantization/(8*1024*1024))+" MB)")
        print("    DRAM access energy: " + str(dram_energy))
        bw_ifm_max = 1
        bw_filter_max = 1

        print("[BW requirements][BW accelerator]: ")
        ifm_max = ((((self.gbuf.get_ifm_elements() - p.ifm_artificial_padding_remove) * self.num_pe_sets_per_array * self.dataflow_info[c.BATCH]) / p.fullpipe_cycles)+0.5) # +0.5 is for round UP
        ifm_max_total = ifm_max #* num_pe_sets_per_array
        print("    ifm: ["+str(ifm_max_total)+"]["+str(self.bw_i if self.bw_i != -1 else "infinite")+"]")
        filter_max = (((self.gbuf.get_filter_elements()*self.num_pe_sets_per_array) / p.fullpipe_cycles)+0.5) # +0.5 is for round UP
        filter_max_total = filter_max #* num_pe_sets_per_array

        print("    filter: ["+str(filter_max_total)+"]["+str(self.bw_f if self.bw_f != -1 else "infinite")+"]")
        ofm_max = (((self.gbuf.get_ofm_elements()*self.num_pe_sets_per_array*self.dataflow_info[c.BATCH]) / p.fullpipe_cycles)+0.5) # +0.5 is for round UP
        ofm_max_total = ofm_max #* num_pe_sets_per_array
        print("    ofm: [" + str(ofm_max_total)+"]["+str(self.bw_o if self.bw_o != -1 else "infinite")+"]")

        # We calculate the number of cycles with the restrictions of the hardware arquitecture
        ifms_per_conv = (self.gbuf.get_ifm_elements() - p.ifm_artificial_padding_remove) * self.dataflow_info[c.BATCH]  #self.dataflow_info[c.IFM_H] * self.dataflow_info[c.IFM_W]
        ofms_per_conv = self.gbuf.get_ofm_elements() * self.dataflow_info[c.BATCH] #self.dataflow_info[c.OFM_H] * self.dataflow_info[c.OFM_W]
        parameters_per_conv = self.gbuf.get_filter_elements() - p.fil_artificial_padding_remove#self.dataflow_info[c.FIL_H] * self.dataflow_info[c.FIL_W]

        fullpipe_cycles = p.fullpipe_cycles
        cycles_ifm = (max((ifms_per_conv*self.num_pe_sets_per_array / self.bw_i), fullpipe_cycles) if self.bw_i != -1 else fullpipe_cycles)
        cycles_ofm = (max((ofms_per_conv*self.num_pe_sets_per_array / self.bw_o), fullpipe_cycles) if self.bw_o != -1 else fullpipe_cycles)
        cycles_filter = (max((parameters_per_conv*self.num_pe_sets_per_array / self.bw_f), fullpipe_cycles) if self.bw_f != -1 else fullpipe_cycles)

        print("ifm_cycles: "+str(cycles_ifm/self.num_pe_sets_per_array))
        print("ofm_cycles: "+str(cycles_ofm/self.num_pe_sets_per_array))
        print("filter_cycles: "+str(cycles_filter/self.num_pe_sets_per_array))
        print("fullpipe_cycles: "+str(p.fullpipe_cycles/self.num_pe_sets_per_array))

        total_cycles_conv = max(cycles_ifm, cycles_ofm, cycles_filter, p.fullpipe_cycles)
        print("Total cycles conv: "+str(total_cycles_conv))


        print("Num_pe_sets_per_array: "+str(self.num_pe_sets_per_array))

        total_cycles_accelerator = total_cycles_conv * num_conv / self.num_pe_sets_per_array

        total_mseconds = total_cycles*1000/(self.accelerator_info[c.CLOCK]*1000000)
        print("[Cycles]:")
        print("    Ideal      : "+ str(total_cycles) +" ("+str(total_mseconds)+" ms)")
        print("    Accelerator: "+ str(total_cycles_accelerator) +" ("+str(total_cycles_accelerator*1000/(self.accelerator_info[c.CLOCK]*1000000))+" ms)")
        print("Total energy (array): "+ str(total_energy))
        power = total_energy*1000/(total_mseconds * 1000000000000) # in wats
        print("Power: "+ str(power*1000)+" mW")
        # Total MACS
        total_MACs = p.total_mul * self.dataflow_info[c.NUM_CHANNELS] * self.dataflow_info[c.NUM_FILTERS] * self.dataflow_info[c.BATCH]
        print("Total MACs: "+ str(total_MACs) + " ("+str(total_MACs/1000000000)+" GMACs)")

        # Printing in the file
        f_stats.write("acc_w;"+str(self.w)+"\n")
        f_stats.write("acc_h;"+str(self.h)+"\n")
        f_stats.write("acc_clock;"+str(self.accelerator_info[c.CLOCK])+"\n")
        f_stats.write("acc_dram_footprint_MB;"+str((memory/(8*1024*1024)))+"\n")
        f_stats.write("acc_bw_ifm_req;"+str(ifm_max_total)+"\n")
        f_stats.write("acc_bw_ifm_acc;"+str(self.bw_i)+"\n")
        f_stats.write("acc_bw_ofm_req;"+str(ofm_max_total)+"\n")
        f_stats.write("acc_bw_ofm_acc;"+str(self.bw_o)+"\n")
        f_stats.write("acc_bw_filter_req;"+str(filter_max_total)+"\n")
        f_stats.write("acc_bw_filter_acc;"+str(self.bw_f)+"\n")
        f_stats.write("acc_cycles_ideal;"+str(total_cycles)+"\n")
        f_stats.write("acc_cycles_acc;"+str(total_cycles_accelerator)+"\n")

        return total_cycles_accelerator

    def sum_dec(self, n):
        result = 0
        for i in range(n):
            result += i
        return result

    def print_stats(self):
        '''
        Print PE stats
        '''
        p = stats_performance()
        e = stats_energy()
        p.fullpipe_cycles   = 0
        p.total_mul_by_zero = 0
        p.total_mul         = 0

        # Get the Full pipeline ccyles of a 2D Convolution
        for w in range(self.w):
            for h in range(self.h):
                p.total_mul_by_zero += self.matrix[h][w].get_mul_by_zero()
                p.total_mul         += self.matrix[h][w].get_mul()
                cyc = self.matrix[h][w].get_cycles_fullpipe()
                if cyc > p.fullpipe_cycles:
                    p.fullpipe_cycles = cyc

        p.ifm_artificial_padding_remove = 0
        p.fil_artificial_padding_remove = 0
        p.filter_elements = self.dataflow_info[c.FIL_W] * self.dataflow_info[c.FIL_H]
        p.ifm_elements = self.dataflow_info[c.IFM_W] * self.dataflow_info[c.IFM_H]
        p.ofm_elements = self.dataflow_info[c.OFM_W] * self.dataflow_info[c.OFM_H]

        if self.pe_type == c.SYSTOLIC:
            if self.dataflow_info[c.PASS_T] == "forward":
                p.total_mul = p.ofm_elements * p.filter_elements
                p.fullpipe_cycles = p.filter_elements + 1
            elif self.dataflow_info[c.PASS_T] == "igrad":
                p.total_mul = p.ofm_elements * p.filter_elements
                p.fullpipe_cycles = p.filter_elements + 1
            elif self.dataflow_info[c.PASS_T] == "fgrad":
                padded_filter = (self.dataflow_info[c.OFM_W] + ((self.dataflow_info[c.OFM_W]-1)*(self.dataflow_info[c.STRIDE]-1)))  * (self.dataflow_info[c.OFM_H] + ((self.dataflow_info[c.OFM_H]-1)*(self.dataflow_info[c.STRIDE]-1)))
                p.total_mul = p.ofm_elements * p.filter_elements
                p.fullpipe_cycles = (p.ofm_elements)+ 1
            else:
                raise

        # Print PE stats
        fname_detailed = self.dname_stats+"/"+self.fname_stats+".full.stats"
        detailed_stats_file = open(fname_detailed, "w")
        for w in range(self.w):
            for h in range(self.h):
                cyc = self.matrix[h][w].print_stats(p.fullpipe_cycles, detailed_stats_file)

        # Overall stats
        p.txdata_cycles = max(self.__cycles(self.ifmap_seq),self.__cycles(self.filter_seq))
        print("[OVERALL - one 2D convolution] "+self.name)
        p.pe_set_h = self.h
        p.pe_set_w = self.w
        p.total_pe_set = self.h * self.w
        print("    Total PEs                         : "+str(self.h*self.w)+" ("+str(self.h)+"x"+str(self.w)+")")
        print("    IFM elements                               : "+str(p.ifm_elements) + " ("+str(self.dataflow_info[c.IFM_W])+"x"+str(self.dataflow_info[c.IFM_H])+")")
        print("    IFM PAD                                    : "+str(self.dataflow_info[c.IFM_PAD]))
        print("    OFM elements                               : "+str(p.ofm_elements) + " ("+str(self.dataflow_info[c.OFM_W])+"x"+str(self.dataflow_info[c.OFM_H])+")")
        print("    Filter elements                            : "+str(p.filter_elements)+ " ("+str(self.dataflow_info[c.FIL_W])+"x"+str(self.dataflow_info[c.FIL_H])+")")
        p.total_simulated_cycles = self.cycle + 1
        print("    Total simulated cycles                     : "+str(self.cycle + 1))
        p.cycles_array2gbuffer = round((self.num_ofms/self.ofm_bw),1)
        print("    Cycles data from array (OFM) to GBUFF      : "+str(self.ofm_end - self.ofm_init)+(" (ideal: "+str(round((self.num_ofms/self.ofm_bw),1))+")"))
        print("    Cycles per 2D conv (pipeline full)         : "+str(p.fullpipe_cycles))
        p.max_array2gbuffer_bw = self.max_array_to_gbuffer_bw
        print("    Max array-to-gbuffer bandwidth             : "+str(self.max_array_to_gbuffer_bw))
        print("    Total multiplications                      : "+str(p.total_mul))
        if p.total_mul == 0:
            print("    Total multiplications by zero              : "+str(p.total_mul_by_zero)+" (0%)")
        else:
            print("    Total multiplications by zero              : "+str(p.total_mul_by_zero)+" ("+str(100*p.total_mul_by_zero/p.total_mul)+"%)")
        # Calculate energy
        e.e_total = 0
        e.e_mem_ifm = 0
        e.e_mem_fil = 0
        e.e_mem_psum = 0
        e.e_mul = 0
        e.e_sum = 0
        e.e_noc = 0
        e_noc_i_ifm = 0
        e_noc_o_ifm = 0
        e_noc_i_fil = 0
        e_noc_o_fil = 0
        e_noc_i_psum = 0
        e_noc_o_psum = 0
        for w in range(self.w):
            for h in range(self.h):
                e.e_mem_ifm  += self.matrix[h][w].get_energy(c.IFM_MEM)
                e.e_mem_fil  += self.matrix[h][w].get_energy(c.FILTER_MEM)
                e.e_mem_psum += self.matrix[h][w].get_energy(c.PSUM_MEM)
                e.e_mul      += self.matrix[h][w].get_energy(c.MUL)
                e.e_sum      += self.matrix[h][w].get_energy(c.SUM)
                e.e_noc      += self.matrix[h][w].get_energy(c.LINK)
                e_noc_i_ifm += self.matrix[h][w].get_energy(c.IN_IFM)
                e_noc_i_fil += self.matrix[h][w].get_energy(c.IN_FILTER)
                e_noc_i_psum += self.matrix[h][w].get_energy(c.IN_PSUM)
                e_noc_o_psum += self.matrix[h][w].get_energy(c.OUT_PSUM)
                if self.pe_type == c.SYSTOLIC:
                    e_noc_o_ifm += self.matrix[h][w].get_energy(c.OUT_IFM)
                    e_noc_o_fil += self.matrix[h][w].get_energy(c.OUT_FILTER)

        e_spad = e.e_mem_ifm + e.e_mem_fil + e.e_mem_psum
        e_mul = e.e_mul
        e_sum = e.e_sum
        e_noc = e.e_noc
        e.e_total = e.e_mem_ifm + e.e_mem_fil + e.e_mem_psum + e.e_mul + e.e_sum + e.e_noc

        # TODO: Energy of GB, energy of DRAM
        # Write in CSV format the overall stats
        fname_summary = self.dname_stats+"/"+self.fname_stats+".summary.stats"
        summary_stats_file = open(fname_summary, "w")
        summary_stats_file.write("name;"+self.name+"\n")
        summary_stats_file.write("ifm_elements;"+str(self.gbuf.get_ifm_elements())+"\n")
        summary_stats_file.write("ofm_elements;"+str(self.gbuf.get_ofm_elements())+"\n")
        summary_stats_file.write("filter_elements;"+str(self.gbuf.get_filter_elements())+"\n")
        summary_stats_file.write("num_channels;"+str(self.dataflow_info[c.NUM_CHANNELS])+"\n")
        summary_stats_file.write("num_filters;"+str(self.dataflow_info[c.NUM_FILTERS])+"\n")
        summary_stats_file.write("pe_total;"+str(self.h*self.w)+"\n")
        summary_stats_file.write("pe_w;"+str(self.w)+"\n")
        summary_stats_file.write("pe_h;"+str(self.h)+"\n")
        summary_stats_file.write("cycles_buf2arr;"+str(p.txdata_cycles)+"\n")
        summary_stats_file.write("cycles_arr2buf;"+str(self.ofm_end-self.ofm_init)+"\n")
        summary_stats_file.write("cycles_total;"+str(self.cycle+1)+"\n")
        summary_stats_file.write("cycles_pipefull;"+str(p.fullpipe_cycles)+"\n")
        summary_stats_file.write("bw_arr2buf;"+str(self.max_array_to_gbuffer_bw)+"\n")
        summary_stats_file.write("mul_total;"+str(p.total_mul)+"\n")
        summary_stats_file.write("mul_zero;"+str(p.total_mul_by_zero)+"\n")
        summary_stats_file.write("gbuff_rd_nozero;"+str(self.gbuf.get_reads_ifm()+self.gbuf.get_reads_filter())+"\n")
        summary_stats_file.write("gbuff_rd_zero;"+str(self.gbuf.get_reads_ifm(True)+self.gbuf.get_reads_filter(True))+"\n")
        summary_stats_file.write("energy_array_pj;"+str(e.e_total)+"\n")
        summary_stats_file.write("energy_gbuff_pj;"+str(e.e_gbuff)+"\n")
        summary_stats_file.write("energy_mem_ifm_pj;"+str(e.e_mem_ifm)+"\n")
        summary_stats_file.write("energy_mem_fil_pj;"+str(e.e_mem_fil)+"\n")
        summary_stats_file.write("energy_mem_psum_pj;"+str(e.e_mem_psum)+"\n")
        summary_stats_file.write("energy_mul_pj;"+str(e.e_mul)+"\n")
        summary_stats_file.write("energy_sum_pj;"+str(e.e_sum)+"\n")
        summary_stats_file.write("energy_noc_pj;"+str(e.e_noc)+"\n")


        # Calculate the final stats based on the physical dimensions
        # of the accelerator
        total_cycles_accelerator = self.calc_total(p,e, summary_stats_file)


        print("[FULL LAYER] "+self.name)
        # Calculates the memory size and bandwidth requirements
        # Taking into account the data reuse of each dataflow
        spad_accesses_ifm  = 0
        spad_accesses_fil  = 0
        spad_accesses_psum = 0
        spad_link_accesses = 0
        for w in range(self.w):
            for h in range(self.h):
                spad_accesses_ifm  += self.matrix[h][w].get_accesses(c.IFM_MEM)
                spad_accesses_fil  += self.matrix[h][w].get_accesses(c.FILTER_MEM)
                spad_accesses_psum += self.matrix[h][w].get_accesses(c.PSUM_MEM)
                spad_link_accesses += self.matrix[h][w].get_accesses(c.LINK)
        num_conv = self.dataflow_info[c.NUM_CHANNELS] * self.dataflow_info[c.NUM_FILTERS] * self.dataflow_info[c.BATCH]
        spad_accesses_ifm  *= num_conv
        spad_accesses_fil  *= num_conv
        spad_accesses_psum *= num_conv
        spad_accesses = spad_accesses_ifm + spad_accesses_fil + spad_accesses_psum
        spad_link_accesses *= num_conv

        mm = memory_model(self.dataflow_info, self.accelerator_info, self.pe_sets_w, self.pe_sets_h, self.pe_set_w, self.pe_set_h)
        print("[SIZE-SPAD]")
        print("    filter_size: "+str(mm.get_rfsize(c.FILTER_MEM))+"/"+str(mm.get_prfsize(c.FILTER_MEM))+" ("+str(int(100*mm.get_rfsize(c.FILTER_MEM)/mm.get_prfsize(c.FILTER_MEM)))+"%)")
        print("    ifm_size: "+str(mm.get_rfsize(c.IFM_MEM))+"/"+str(mm.get_prfsize(c.IFM_MEM))+" ("+str(int(100*mm.get_rfsize(c.IFM_MEM)/mm.get_prfsize(c.IFM_MEM)))+"%)")
        print("    psum_size: "+str(mm.get_rfsize(c.PSUM_MEM))+"/"+str(mm.get_prfsize(c.PSUM_MEM))+" ("+str(int(100*mm.get_rfsize(c.PSUM_MEM)/mm.get_prfsize(c.PSUM_MEM)))+"%)")
        print("[SIZE-BUFFER]")
        print("    filter_size: "+str(mm.get_gbsize(c.GB_FIL)/1024)+"/"+str(mm.get_gbsize(c.GB_FIL)/1024)+" KB ("+str(int(100*mm.get_gbsize(c.GB_FIL)/mm.get_gbsize(c.GB_FIL)))+"%)")
        print("    ifm_size: "+str(mm.get_gballoc(c.GB_IFM)/1024)+"/"+str(mm.get_gbsize(c.GB_IFM)/1024)+" KB ("+str(int(100*mm.get_gballoc(c.GB_IFM)/mm.get_gbsize(c.GB_IFM)))+"%)")
        print("    ofm_size: "+str(mm.get_gballoc(c.GB_PSUM)/1024)+"/"+str(mm.get_gbsize(c.GB_PSUM)/1024)+" KB ("+str(int(100*mm.get_gballoc(c.GB_PSUM)/mm.get_gbsize(c.GB_PSUM)))+"%)")

        # Write in CSV format the overall stats
        summary_stats_file.write("size_spad_filter;"+str(mm.get_rfsize(c.FILTER_MEM))+"\n")
        summary_stats_file.write("size_spad_ifm;"+str(mm.get_rfsize(c.IFM_MEM))+"\n")
        summary_stats_file.write("size_spad_psum;"+str(mm.get_rfsize(c.PSUM_MEM))+"\n")
        summary_stats_file.write("size_gb_filter;"+str(mm.get_gbsize(c.GB_FIL))+"\n")
        summary_stats_file.write("size_gb_ifm;"+str(mm.get_gbsize(c.GB_IFM))+"\n")
        summary_stats_file.write("size_gb_psum;"+str(mm.get_gbsize(c.GB_PSUM))+"\n")

        summary_stats_file.write("df_n;"+str(mm.get_param(c.DF_N))+"\n")
        summary_stats_file.write("df_p;"+str(mm.get_param(c.DF_P))+"\n")
        summary_stats_file.write("df_q;"+str(mm.get_param(c.DF_Q))+"\n")
        summary_stats_file.write("df_r;"+str(mm.get_param(c.DF_R))+"\n")
        summary_stats_file.write("df_t;"+str(mm.get_param(c.DF_T))+"\n")

        total_spad_accesses = spad_accesses_ifm + spad_accesses_fil + spad_accesses_psum
        print("[ACCESSES-SPAD]")
        print("    SPAD_TOTAL_ACCESSES: "+str((total_spad_accesses + spad_link_accesses)/float(1024*1024))+" MB")
        print("    SPAD_IFM: "+str(spad_accesses_ifm/float(1024*1024))+" MB")
        print("    SPAD_FIL: "+str(spad_accesses_fil/float(1024*1024))+" MB")
        print("    SPAD_PSUM: "+str(spad_accesses_psum/float(1024*1024))+" MB")
        print("    SPAD_LINK: "+str(spad_link_accesses/float(1024*1024))+" MB")

        summary_stats_file.write("accesses_spad_total;"+str(total_spad_accesses + spad_link_accesses)+"\n")
        summary_stats_file.write("accesses_spad_ifm;"+str(spad_accesses_ifm)+"\n")
        summary_stats_file.write("accesses_spad_fil;"+str(spad_accesses_fil)+"\n")
        summary_stats_file.write("accesses_spad_psum;"+str(spad_accesses_psum)+"\n")
        summary_stats_file.write("accesses_spad_link;"+str(spad_link_accesses)+"\n")

        gb_rd_ifm  = mm.get_gb_accesses(c.GB_RD_IFM)
        gb_rd_fil  = mm.get_gb_accesses(c.GB_RD_FIL)
        gb_rd_psum = mm.get_gb_accesses(c.GB_RD_PSUM)
        gb_wr_psum = mm.get_gb_accesses(c.GB_WR_PSUM)
        total_gb_accesses = (gb_rd_ifm+gb_rd_fil+gb_rd_psum+gb_wr_psum)

        print("[ACCESSES-GB]")
        print("    GB_TOTAL_ACCESSES: "+str((total_gb_accesses)/float(1024*1024))+" MB")
        print("    GB_IFM_READ: "+str(gb_rd_ifm/float(1024*1024))+" MB")
        print("    GB_FIL_READ: "+str(gb_rd_fil/float(1024*1024))+" MB")
        print("    GB_PSUM_READ: "+str(gb_rd_psum/float(1024*1024))+" MB")
        print("    GB_PSUM_WRITE: "+str(gb_wr_psum/float(1024*1024))+" MB")

        # Write in CSV format the overall stats
        summary_stats_file.write("accesses_gb_total;"+str(total_gb_accesses)+"\n")
        summary_stats_file.write("accesses_gb_ifm;"+str(gb_rd_ifm)+"\n")
        summary_stats_file.write("accesses_gb_fil;"+str(gb_rd_fil)+"\n")
        summary_stats_file.write("accesses_gb_psum;"+str(gb_rd_psum + gb_wr_psum)+"\n")

        dram_rd_ifm  = mm.get_dram_accesses(c.DRAM_RD_IFM)
        dram_rd_fil  = mm.get_dram_accesses(c.DRAM_RD_FIL)
        dram_wr_ofm  = mm.get_dram_accesses(c.DRAM_WR_OFM)
        total_dram_accesses = (dram_rd_ifm+dram_rd_fil+dram_wr_ofm)
        print("[ACCESSES-DRAM]")
        print("    DRAM_TOTAL: "+str((total_dram_accesses)/float(1024*1024))+" MB")
        print("    DRAM_IFM_READ: "+str(dram_rd_ifm/float(1024*1024))+" MB")
        print("    DRAM_FIL_READ: "+str(dram_rd_fil/float(1024*1024))+" MB")
        print("    DRAM_OFM_WRITE: "+str(dram_wr_ofm/float(1024*1024))+" MB")

        # Write in CSV format the overall stats
        summary_stats_file.write("accesses_dram_total;"+str(total_dram_accesses)+"\n")
        summary_stats_file.write("accesses_dram_ifm;"+str(dram_rd_ifm)+"\n")
        summary_stats_file.write("accesses_dram_fil;"+str(dram_rd_fil)+"\n")
        summary_stats_file.write("accesses_dram_psum;"+str(dram_wr_ofm)+"\n")

        # Get the energy from DRAMPower
        if DRAMPOWER:
            dramp = DRAMPower(dram_rd_ifm, dram_rd_fil, dram_wr_ofm, total_cycles_accelerator, self.accelerator_info[c.CLOCK])
            dramp.generate_traces()
            e_dram = dramp.simulate_energy()
        else:
            e_dram = (total_dram_accesses )* active_energy(c.E_DRAM)

        e_gb = total_gb_accesses * active_energy(c.E_GB)

        e_array = e.e_total * num_conv
        e_spad  = e_spad * num_conv
        e_spad_ifm = e.e_mem_ifm * num_conv
        e_spad_fil = e.e_mem_fil * num_conv
        e_spad_psum = e.e_mem_psum * num_conv
        e_noc  = e_noc * num_conv
        e_mul = e_mul * num_conv
        e_sum = e_sum * num_conv
        e_total = e_array + e_gb + e_dram

        e_noc_i_ifm *= num_conv
        e_noc_o_ifm *= num_conv
        e_noc_i_fil *= num_conv
        e_noc_o_fil *= num_conv
        e_noc_i_psum *= num_conv
        e_noc_o_psum *= num_conv

        print("[ENERGY]")
        print("    TOTAL: "+str(e_total/1000000) + " mJ")
        print("    ARRAY: "+str(e_array)+" nJ ({:.1f}".format(100*e_array/e_total)+"%)")
        print("             SPAD: "+str(e_spad)+" nJ ({:.1f}".format(100*e_spad/e_total)+"%)")
        print("                 spad-ifm: "+str(e_spad_ifm)+" nJ ({:.1f}".format(100*e_spad_ifm/e_total)+"%)")
        print("                 spad-fil: "+str(e_spad_fil)+" nJ ({:.1f}".format(100*e_spad_fil/e_total)+"%)")
        print("                 spad-psum: "+str(e_spad_psum)+" nJ ({:.1f}".format(100*e_spad_psum/e_total)+"%)")
        print("              NoC: "+str(e_noc)+" nJ ({:.1f}".format(100*e_noc/e_total)+"%)")
        print("                 noc-i-ifm: "+str(e_noc_i_ifm)+" nJ ({:.1f}".format(100*e_noc_i_ifm/e_total)+"%)")
        print("                 noc-o-ifm: "+str(e_noc_o_ifm)+" nJ ({:.1f}".format(100*e_noc_o_ifm/e_total)+"%)")
        print("                 noc-i-fil: "+str(e_noc_i_fil)+" nJ ({:.1f}".format(100*e_noc_i_fil/e_total)+"%)")
        print("                 noc-o-fil: "+str(e_noc_o_fil)+" nJ ({:.1f}".format(100*e_noc_o_fil/e_total)+"%)")
        print("                 noc-i-psum: "+str(e_noc_i_psum)+" nJ ({:.1f}".format(100*e_noc_i_psum/e_total)+"%)")
        print("                 noc-o-psum: "+str(e_noc_o_psum)+" nJ ({:.1f}".format(100*e_noc_o_psum/e_total)+"%)")
        print("              MUL: "+str(e_mul)+" nJ ({:.1f}".format(100*e_mul/e_total)+"%)")
        print("              SUM: "+str(e_sum)+" nJ ({:.1f}".format(100*e_sum/e_total)+"%)")
        print("       GB: "+str(e_gb)+" nJ ({:.1f}".format(100*e_gb/e_total)+"%)")
        print("     DRAM: "+str(e_dram)+" nJ ({:.1f}".format(100*e_dram/e_total)+"%)")

        summary_stats_file.write("total_energy_array;"+str(e_array)+"\n")
        summary_stats_file.write("total_energy_gb;"+str(e_gb)+"\n")
        summary_stats_file.write("total_energy_dram;"+str(e_dram)+"\n")
        summary_stats_file.write("total_energy_spad;"+str(e_spad)+"\n")
        summary_stats_file.write("total_energy_noc;"+str(e_noc)+"\n")
        summary_stats_file.write("total_energy_sum;"+str(e_sum)+"\n")
        summary_stats_file.write("total_energy_mul;"+str(e_mul)+"\n")

        print("[ENERGY_PER_BYTE]")
        epb_spad = e_spad/total_spad_accesses
        epb_noc = e_noc/spad_link_accesses
        epb_gb = e_gb/total_gb_accesses
        epb_dram = e_dram/total_dram_accesses
        print("     SPAD: "+str(epb_spad)+" nJ")
        print("      NoC: "+str(epb_noc)+" nJ")
        print("       GB: "+str(epb_gb)+" nJ")
        print("     DRAM: "+str(epb_dram)+" nJ")
        # Write in CSV format the overall stats
        summary_stats_file.write("energy_per_byte_spad;"+str(epb_spad)+"\n")
        summary_stats_file.write("energy_per_byte_noc;"+str(epb_noc)+"\n")
        summary_stats_file.write("energy_per_byte_gb;"+str(epb_gb)+"\n")
        summary_stats_file.write("energy_per_byte_dram;"+str(epb_dram)+"\n")
        summary_stats_file.close()

    def print_trace(self,val):
        '''
        Print in the memory trace
        '''
        self.f_traces.write(str(val))
        self.f_traces.write("\n")


    #################################
    # Common functions ( Spatial Architecture VS Systolic Array)
    #################################
    def check_end(self, cycle):
        '''
        Check for the end condition
        '''
        for w in range(self.w):
            for h in range(self.h):
                if cycle >= self.matrix[h][w].get_max_cycle():
                    self.running = False
    def check_end_systolic(self, cycle):
        '''
        Check for the end condition
        '''
        delay = 3
        print("length: "+str(self.matrix[0][0].get_length_filter()))
        #raise

        filter_total_length = self.matrix[0][0].get_length_filter()
        if cycle >= (self.w + filter_total_length + delay):
            self.running = False

    def print_debug(self):
        '''
        Print debuging information
        '''
        for w in range(self.w):
            for h in range(self.h):
                self.matrix[h][w].print_debug() # Print debugging information

    ############################
    # Advance function
    ############################
    def advance(self, cycle):
        if self.pe_type == c.SYSTOLIC:
            return self.advance_systolic(cycle)
        else:
            return self.advance_spatial(cycle)

    ############################
    # Systolic Array
    ############################
    def advance_systolic(self, cycle):
        '''
        Systolic array implementation
        '''
        self.check_end_systolic(cycle)
        if self.running:
            self.print_debug()

            # Write to the global buffer
            cbandwidth = 0 # Bandwidth required in the current cycle
            bw_left = self.ofm_bw
            mem_out = [[False for w in range(self.w)] for h in range(self.h)]
            for w in range(self.w):
                for h in range(self.h):
                    # Write to memory in order, from previous cycles
                    if bw_left > 0:
                        if not self.ofm_buffer_queues[h][w].empty():
                            # send to memory
                            [addr,val] = self.ofm_buffer_queues[h][w].get()
                            self.gbuf.write_ofm(addr, val)
                            if self.enable_traces:
                                self.print_trace(addr)
                            cbandwidth += 1
                            bw_left -= 1
                            self.num_ofms +=1
                            if self.ofm_end == 0:
                                self.ofm_init = cycle
                            self.ofm_end = cycle

                    to_mem_addr = str(self.matrix[h][w].to_memory(self.cycle-w))
                    if to_mem_addr is not "":
                        ofm = self.matrix[h][w].get_queue(c.OUT_PSUM)
                        if ofm is "":
                            print("[ERROR] Nothing to write to memory from PE"+str(h)+str(w)+". to_mem_addr = \""+str(to_mem_addr)+"\" . Debug your code! ")
                        mem_out[h][w] = True
                        self.gbuf.write_ofm(to_mem_addr, ofm)
                        if self.enable_traces:
                            self.print_trace(to_mem_addr)
                        cbandwidth += 1
                        bw_left -= 1
                        self.num_ofms +=1
                        if self.ofm_end == 0:
                            self.ofm_init = cycle
                        self.ofm_end = cycle
            self.max_array_to_gbuffer_bw = max(self.max_array_to_gbuffer_bw, cbandwidth)

            #############################
            # Key difference between a Spatial Architecture
            # and a systolic array
            # Each PE transmits values from top to bottom and form left  to the right
            # Multiplies Top by Left, and transmit Diagonal
            # Partial sums are accumulated localy in psum
            #############################
            for w in range(self.w):
                for h in range(self.h):
                    ofilter = self.matrix[h][w].get_queue(c.OUT_FILTER)
                    if w < (self.w-1):
                        if ofilter != "" and (ofilter is not None) and (ofilter != -1):
                            self.matrix[h][w+1].put_queue(c.IN_FILTER, ofilter)

            # Advance the state of all PEs
            for w in range(self.w):
                for h in range(self.h):
                    self.matrix[h][w].advance_systolic(cycle) #advance

            self.cycle+=1

        if not self.running:
            return -1
        return 0

    ############################
    # Spatial Architecture
    ############################
    def advance_spatial(self,cycle):
        '''
        Advance one cycle
        '''
        # Check for end condition
        self.check_end(cycle)

        # Debugging cycle by cycle
        if self.running:
            self.print_debug()

            # IFMs into the array
            pos_ifm = self.ifmap_seq[self.cycle]
            if type(pos_ifm) != type([]):
                pos_ifm = [pos_ifm]

            c_bw = len(pos_ifm)
            if c_bw > self.ifm_bw:
                print("[ERROR] maximum bandwidth exceeded. Check your compiler")
                raise
            for bw in range(len(pos_ifm)):
                if pos_ifm[bw] is not c.NAN:
                    gbuff_ifm =  self.gbuf.read_ifm()
                    if self.enable_traces:
                        self.print_trace(gbuff_ifm)
                    # send to the PE array
                    for w in range(self.w):
                        for h in range(self.h):
                            if pos_ifm[bw] in self.noc_ifmap[h][w]:
                                self.matrix[h][w].put_queue(c.IN_IFM, gbuff_ifm)


            # FILTERs into the array
            pos_filter = self.filter_seq[self.cycle]
            if type(pos_filter) != type([]):
                pos_filter = [pos_filter]
            c_bw = len(pos_filter)
            if c_bw > self.fil_bw:
                print("[ERROR] maximum bandwidth exceeded. Check your compiler")
                raise

            for bw in range(c_bw):
                if pos_filter[bw] is not c.NAN:
                    gbuff_filter = self.gbuf.read_filter()
                    if self.enable_traces:
                        self.print_trace(gbuff_filter)
                    # send to the PE array
                    for w in range(self.w):
                        for h in range(self.h):
                            if pos_filter[bw] in self.noc_filter[h][w]:
                                self.matrix[h][w].put_queue(c.IN_FILTER, gbuff_filter)

            # Write to the global buffer
            cbandwidth = 0 # Bandwidth required in the current cycle
            bw_left = self.ofm_bw
            mem_out = [[False for w in range(self.w)] for h in range(self.h)]
            for w in range(self.w):
                for h in range(self.h):
                    # Write to memory in order, from previous cycles
                    writen = False
                    if bw_left > 0:
                        if not self.ofm_buffer_queues[h][w].empty():
                            # send to memory
                            [addr,val] = self.ofm_buffer_queues[h][w].get()
                            self.gbuf.write_ofm(addr, val)
                            if self.enable_traces:
                                self.print_trace(addr)
                            cbandwidth += 1
                            bw_left -= 1
                            writen = True
                            self.num_ofms +=1
                            if self.ofm_end == 0:
                                self.ofm_init = cycle
                            self.ofm_end = cycle

                    to_mem_addr = str(self.matrix[h][w].to_memory(self.cycle))
                    if to_mem_addr is not "":
                        ofm = self.matrix[h][w].get_queue(c.OUT_PSUM)
                        if ofm is "":
                            print("[ERROR] Nothing to write to memory from PE"+str(h)+str(w)+". to_mem_addr = \""+str(to_mem_addr)+"\" . Debug your code! ")
                        mem_out[h][w] = True
                        if writen or (not writen and bw_left==0):
                            self.ofm_buffer_queues[h][w].put([to_mem_addr, ofm])
                        else:
                            self.gbuf.write_ofm(to_mem_addr, ofm)
                            if self.enable_traces:
                                self.print_trace(to_mem_addr)
                            cbandwidth += 1
                            bw_left -= 1
                            self.num_ofms +=1
                            if self.ofm_end == 0:
                                self.ofm_init = cycle
                            self.ofm_end = cycle
            self.max_array_to_gbuffer_bw = max(self.max_array_to_gbuffer_bw, cbandwidth)


            for w in range(self.w):
                for h in range(self.h):
                    if h != 0: # If 0, it is at the top of the array, so no more vertical comunication
                        if mem_out[h][w] is False:
                            opsum = self.matrix[h][w].get_queue(c.OUT_PSUM)
                            self.matrix[h-1][w].put_queue(c.IN_PSUM, opsum)


            # Advance the state of all PEs
            for w in range(self.w):
                for h in range(self.h):
                    self.matrix[h][w].advance(cycle) #advance

            self.cycle+=1
            if self.cycle == len(self.ifmap_seq):
                self.cycle = 0
                raise

        #####################################
        # Empty the output queues
        #####################################
        if not self.running:
            self.cycle+=1
            # Write to memory the remaining elements
            active = False
            bw_left = self.ofm_bw
            for w in range(self.w):
                for h in range(self.h):
                    if not self.ofm_buffer_queues[h][w].empty():
                        # send to memory
                        [addr,val] = self.ofm_buffer_queues[h][w].get()
                        self.gbuf.write_ofm(addr, val)
                        if self.ofm_end == 0:
                            self.ofm_init = cycle
                        self.ofm_end = cycle
                        bw_left -= 1
                        active = True
                        if bw_left == 0:
                            return 0
            if not active:
                print("Exiting...")
                return -1
        return 0


    ############################
    # Spatial Architecture
    ############################
    def advance_spatial_pooling(self,cycle):
        '''
        Advance one cycle
        '''
        # Check for end condition
        self.check_end(cycle)

        # Debugging cycle by cycle
        if self.running:
            self.print_debug()

            # IFMs into the array
            pos_ifm = self.ifmap_seq[self.cycle]
            if type(pos_ifm) != type([]):
                pos_ifm = [pos_ifm]

            c_bw = len(pos_ifm)
            if c_bw > self.ifm_bw:
                print("[ERROR] maximum bandwidth exceeded. Check your compiler")
                raise
            for bw in range(len(pos_ifm)):
                if pos_ifm[bw] is not c.NAN:
                    gbuff_ifm =  self.gbuf.read_ifm()
                    if self.enable_traces:
                        self.print_trace(gbuff_ifm)
                    # send to the PE array
                    for w in range(self.w):
                        for h in range(self.h):
                            if pos_ifm[bw] in self.noc_ifmap[h][w]:
                                self.matrix[h][w].put_queue(c.IN_IFM, gbuff_ifm)

            cbandwidth = 0 # Bandwidth required in the current cycle
            bw_left = self.ofm_bw
            mem_out = [[False for w in range(self.w)] for h in range(self.h)]
            for w in range(self.w):
                for h in range(self.h):
                    # Write to memory in order, from previous cycles
                    writen = False
                    if bw_left > 0:
                        if not self.ofm_buffer_queues[h][w].empty():
                            # send to memory
                            [addr,val] = self.ofm_buffer_queues[h][w].get()
                            self.gbuf.write_ofm(addr, val)
                            if self.enable_traces:
                                self.print_trace(addr)
                            cbandwidth += 1
                            bw_left -= 1
                            writen = True
                            self.num_ofms +=1
                            if self.ofm_end == 0:
                                self.ofm_init = cycle
                            self.ofm_end = cycle

                    to_mem_addr = str(self.matrix[h][w].to_memory(self.cycle))
                    if to_mem_addr is not "":
                        ofm = self.matrix[h][w].get_queue(c.OUT_PSUM)
                        if ofm is "":
                            print("[ERROR] Nothing to write to memory from PE"+str(h)+str(w)+". to_mem_addr = \""+str(to_mem_addr)+"\" . Debug your code! ")
                        mem_out[h][w] = True
                        if writen or (not writen and bw_left==0):
                            self.ofm_buffer_queues[h][w].put([to_mem_addr, ofm])
                        else:
                            self.gbuf.write_ofm(to_mem_addr, ofm)
                            if self.enable_traces:
                                self.print_trace(to_mem_addr)
                            cbandwidth += 1
                            bw_left -= 1
                            self.num_ofms +=1
                            if self.ofm_end == 0:
                                self.ofm_init = cycle
                            self.ofm_end = cycle
            self.max_array_to_gbuffer_bw = max(self.max_array_to_gbuffer_bw, cbandwidth)


            for w in range(self.w):
                for h in range(self.h):
                    if h != 0: # If 0, it is at the top of the array, so no more vertical comunication
                        if mem_out[h][w] is False:
                            opsum = self.matrix[h][w].get_queue(c.OUT_PSUM)
                            self.matrix[h-1][w].put_queue(c.IN_PSUM, opsum)


            # Advance the state of all PEs
            for w in range(self.w):
                for h in range(self.h):
                    self.matrix[h][w].advance(cycle) #advance

            self.cycle+=1
            if self.cycle == len(self.ifmap_seq):
                self.cycle = 0
                raise

        #####################################
        # Empty the output queues
        #####################################
        if not self.running:
            self.cycle+=1
            # Write to memory the remaining elements
            active = False
            bw_left = self.ofm_bw
            for w in range(self.w):
                for h in range(self.h):
                    if not self.ofm_buffer_queues[h][w].empty():
                        # send to memory
                        [addr,val] = self.ofm_buffer_queues[h][w].get()
                        self.gbuf.write_ofm(addr, val)
                        if self.ofm_end == 0:
                            self.ofm_init = cycle
                        self.ofm_end = cycle
                        bw_left -= 1
                        active = True
                        if bw_left == 0:
                            return 0
            if not active:
                print("Exiting...")
                return -1
        return 0


