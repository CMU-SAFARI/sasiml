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
from hw.memory import memory
from hw.multiplier import multiplier
from hw.mux import mux2
from hw.sum import sum
from hw.queue import queue
import hw.constants as c
import numpy as np
from numpy import nan
from hw.energy_model import active_energy
from hw.energy_model import idle_energy

warning_mode = False
debug_mode   = False

class pe(object):
    # the mem_ arguments are are the memory access indices for each step
    def __init__(self, mem_ifm_wr, mem_ifm_rd,
                       mem_filter_wr, mem_filter_rd,
                       mem_psum_wr, mem_psum_rd,
                       mux_seq, ofm, out_psum, size, name, pe_type, debug, debug_pe, pos_h = 0, pos_w = 0, max_cycle=-1, length_filter = -1):
        '''
        PE Hardcoded for now
        '''
        self.debug_pe = debug_pe
        self.debug = debug

        # For systolic arrays
        self.pos_h = pos_h
        self.pos_w = pos_w
        self.length_filter = length_filter

        # Scratchpad memories
        if pe_type == c.SYSTOLIC:
            delay = 3*pos_w # mul+add
            max_comp = length_filter
            if self.pos_w == 0:
                self.mem_filter = memory(size[c.FILTER_MEM], mem_ifm_rd, [], ("MEM_FILTER"), self.debug[c.FILTER_MEM],-1,max_comp)
                self.mem_ifm = memory(size[c.IFM_MEM], mem_ifm_rd, [], ("MEM_IFM"), self.debug[c.IFM_MEM],-1,max_comp)
            else:
                self.mem_filter = memory(size[c.FILTER_MEM], mem_ifm_rd, [], ("MEM_FILTER"), self.debug[c.FILTER_MEM],delay,max_comp)
                self.mem_ifm = memory(size[c.IFM_MEM], mem_ifm_rd, [], ("MEM_IFM"), self.debug[c.IFM_MEM],delay,max_comp)
            self.mem_psum = memory(size[c.PSUM_MEM], [], mem_psum_wr, ("MEM_PSUM"), self.debug[c.PSUM_MEM], delay, max_comp)
        else:
            self.mem_ifm = memory(size[c.IFM_MEM], mem_ifm_rd, mem_ifm_wr, ("MEM_IFM"), self.debug[c.IFM_MEM])
            self.mem_filter = memory(size[c.FILTER_MEM], mem_filter_rd, mem_filter_wr, ("MEM_FILTER"), self.debug[c.FILTER_MEM])
            self.mem_psum = memory(size[c.PSUM_MEM], mem_psum_rd, mem_psum_wr, ("MEM_PSUM"), self.debug[c.PSUM_MEM])

        # Computation
        self._2smul = multiplier(size[c.MUL],size[c.ZERO_CLOCKGATE], name, self.debug[c.MUL])
        self._1sadd = sum(size[c.SUM], name, self.debug[c.SUM])
        self.reg_add = 0

        # Multiplexer
        self.mux = mux2(mux_seq, name, self.debug[c.MUX_MUL])

        # OFM to memory (to GBUFFER)
        self.ofm = ofm

        self.out_psum = out_psum

        # I/O Queues
        self.qifm = queue(("IN_IFM"), size[c.IN_IFM],  self.debug[c.IN_IFM])
        self.qfilter = queue(("IN_FILTER"), size[c.IN_FILTER], self.debug[c.IN_FILTER])
        self.qipsum = queue(("IN_PSUM"), size[c.IN_PSUM], self.debug[c.IN_PSUM])
        self.qopsum = queue(("OUT_PSUM"), size[c.OUT_PSUM], self.debug[c.OUT_PSUM])
        self.name = name
        self.pe_type = pe_type
        # New: Exclusive for the systolic array
        # Propagate the filter and the ifms to adjacent PEs
        if pe_type == c.SYSTOLIC:
            self.qoifm = queue(("OUT_IFM"), size[c.OUT_IFM], self.debug[c.OUT_IFM])
            self.qofilter = queue(("OUT_FILTER"), size[c.OUT_FILTER], self.debug[c.OUT_FILTER])

        # Calculate the max cycle to be simulated
        if max_cycle == -1:
            self.max_cycle = len(mem_ifm_rd) - 1
        else:
            self.max_cycle = max_cycle

        self.prev_cycle = -1

        # All vectors should be the same size
        if self.pe_type != c.SYSTOLIC:
            if not (len(mem_ifm_rd) == len(mem_filter_rd) == len(mem_psum_wr) == len(mem_psum_rd)
                    == len(mux_seq) == len(ofm) == len(out_psum)):
                    print("[ERROR]["+name+"]All vectors should be the same size")
                    print("mem_ifm_wr: "+str(len(mem_ifm_wr)))
                    print("mem_ifm_rd: "+str(len(mem_ifm_rd)))
                    print("mem_filter_wr: "+str(len(mem_filter_wr)))
                    print("mem_filter_rd: "+str(len(mem_filter_rd)))
                    print("mem_psum_wr: "+str(len(mem_psum_wr)))
                    print("mem_psum_rd: "+str(len(mem_psum_rd)))
                    print("mux_seq: "+str(len(mux_seq)))
                    print("ofm: "+str(len(ofm)))
                    print("out_psum: "+str(len(out_psum)))
                    raise

        # For stats
        self.c_start_full    = -1 # Indicates the end of the gradient calculation when the pipeline is full.
        self.c_end_full      = -1 # Indicates the end of the gradient calculation when the pipeline is full.
        self.occ_mul         = 0 # Occupation of the multiplier
        self.occ_sum         = 0 # Occupation of the multiplier
        self.max_iqifm_occ     = 0
        self.max_iqfilter_occ     = 0
        self.max_iqpsum_occ     = 0

        self.limit_bottom = False
        self.limit_right = False

        # Layer type
        self.pooling_forward = False


        # For systolic array
        delay = 2 + 1 + 1 # Delay from PE to PE
        self.to_mem_cycle = self.length_filter*self.length_filter + delay  # self.filter_size+delay


    def get_length_filter(self):
        return self.length_filter


    def load_reg(self, reg, init_val):
        '''
        Set the initial values on registers
        '''
        if debug_mode:
            print("Loading initial values in memory ... ["+str(self.name)+"][REG:"+str(reg)+"] init_val: "+str(init_val))
        if reg == c.IFM_MEM:
            for i in range(len(init_val)):
                self.mem_ifm.write(init_val[i])
        elif reg == c.FILTER_MEM:
            for i in range(len(init_val)):
                self.mem_filter.write(init_val[i])
        else:
            print("[ERROR] Not implemented "+str(reg)+" init function")
            assert(0)

    def print_w(self, text):
        '''
        Print text only in warning mode
        '''
        if warning_mode:
            print("[WARNING]"+text)

    def print_d(self, text):
        '''
        Print text only in debug mode
        '''
        if warning_mode:
            print("[DEBUG]"+text)

    def set_pooling_forward(self):
        '''
        Set the data type
        '''
        self.pooling_forward = True

    def set_limit_right(self):
        '''
        Indicates if the PE is in the border of the array
        '''
        self.limit_right = True

    def set_limit_bottom(self):
        '''
        Indicates if the PE is in the border of the array
        '''
        self.limit_bottom = True

    def unique_elements(self, array):
        '''
        Count the number of unique elements in an array
        '''
        # Exclude nan
        unique = []
        for i in range(len(array)):
            if array[i] is not c.NAN:
                unique.append(array[i])
        a = np.array(unique)
        a_unique = np.unique(a)
        return len(a_unique)

    def get_toMem(self):
        '''
        Return the number of ofms that this PE sends to memory
        '''
        count = 0
        for i in range(len(self.ofm)):
            if self.ofm[i] != "":
                count += 1
        return count

    def get_mul_by_zero(self):
        '''
        Return the number of multiplications by zero
        '''
        return self._2smul.get_mul_by_zero()

    def get_mul(self):
        '''
        Return the number of multiplications
        '''
        return self._2smul.get_mul()


    def get_energy(self, element):
        '''
        Return the total energy comsumed by the PE
        '''
        if element == c.IFM_MEM:
            return self.mem_ifm.get_energy()
        elif element == c.FILTER_MEM:
            return self.mem_filter.get_energy()
        elif element == c.PSUM_MEM:
            return self.mem_psum.get_energy()
        elif element ==  c.MUL:
            return self._2smul.get_energy()
        elif element ==  c.SUM:
            return self._1sadd.get_energy()
        elif element ==  c.LINK:
            if self.pe_type == c.SYSTOLIC:
                return self.qifm.get_energy() + self.qfilter.get_energy() + self.qipsum.get_energy() + self.qopsum.get_energy() + self.qoifm.get_energy() + self.qofilter.get_energy()
            else:
                return self.qifm.get_energy() + self.qfilter.get_energy() + self.qipsum.get_energy() + self.qopsum.get_energy()
        elif element == c.IN_IFM:
            return self.qifm.get_energy()
        elif element == c.IN_PSUM:
            return self.qipsum.get_energy()
        elif element == c.IN_FILTER:
            return self.qfilter.get_energy()
        elif element == c.OUT_IFM:
            return self.qoifm.get_energy()
        elif element == c.OUT_PSUM:
            return self.qopsum.get_energy()
        elif element == c.OUT_FILTER:
            return self.qofilter.get_energy()
        raise

    def get_accesses(self, element):
        '''
        Get the number of accesses to the SPAD
        '''
        if element == c.IFM_MEM:
            return (self.mem_ifm.get_reads()[0] + self.mem_ifm.get_writes()[0])
        elif element == c.FILTER_MEM:
            return (self.mem_filter.get_reads()[0] + self.mem_filter.get_writes()[0])
        elif element == c.PSUM_MEM:
            return (self.mem_psum.get_reads()[0] + self.mem_psum.get_writes()[0])
        elif element == c.LINK:
            if self.pe_type == c.SYSTOLIC:
                qifm_accesses = self.qifm.get_reads() + self.qifm.get_writes() + self.qoifm.get_reads() + self.qoifm.get_writes()
                qfil_accesses = self.qfilter.get_reads() + self.qfilter.get_writes() + self.qofilter.get_reads() + self.qofilter.get_writes()
            else:
                qifm_accesses = self.qifm.get_reads() + self.qifm.get_writes()
                qfil_accesses = self.qfilter.get_reads() + self.qfilter.get_writes()

            qpsum_accesses = self.qipsum.get_reads() + self.qipsum.get_writes() + self.qopsum.get_reads() + self.qopsum.get_writes()
            return (qifm_accesses + qpsum_accesses + qfil_accesses)
        raise


    def print_stats(self, ofm_cycles, sf):
        '''
        Print stats
        '''
        sf.write("["+self.name+"]:")
        omul = self.occ_mul/(ofm_cycles)
        osum = self.occ_sum/(ofm_cycles)

        max_memifm_capacity     = self.mem_ifm.get_capacity()
        max_memfilter_capacity  = self.mem_filter.get_capacity()
        max_mempsum_capacity    = self.mem_psum.get_capacity()

        memifm_wr,memifm_wr_zeros        = self.mem_ifm.get_writes()
        memfilter_wr,memfilter_wr_zeros  = self.mem_filter.get_writes()
        mempsum_wr,mempsum_wr_zeros      = self.mem_psum.get_writes()

        memifm_rd,memifm_rd_zeros        = self.mem_ifm.get_reads()
        memfilter_rd,memfilter_rd_zeros  = self.mem_filter.get_reads()
        mempsum_rd,mempsum_rd_zeros      = self.mem_psum.get_reads()

        e_unit = "pJ"

        sf.write("    [MUL]"+"\n") #
        sf.write("         ->OCCUPATION              : "+str(omul)+"\n") #
        sf.write("         ->ENERGY                  : "+str(self._2smul.get_energy())+e_unit+"\n")
        sf.write("         ->MUL                     : "+str(self._2smul.get_mul())+"\n")
        if self._2smul.get_mul() != 0:
            percent = str(100*self._2smul.get_mul_by_zero()/self._2smul.get_mul())
        else:
            percent = "--"
        sf.write("         ->MUL_BY_ZERO             : "+str(self._2smul.get_mul_by_zero())+" ("+percent+"%)"+"\n")
        sf.write("    [SUM]"+"\n") #
        sf.write("         ->OCCUPATION              : "+str(osum)+"\n") #
        sf.write("         ->ENERGY                  : "+str(self._1sadd.get_energy())+e_unit+"\n")
        sf.write("    [MEM-IFM]"+"\n")
        sf.write("         ->USED_CAPACITY           : "+str(max_memifm_capacity)+"\n")
        sf.write("         ->ENERGY                  : "+str(self.mem_ifm.get_energy())+e_unit+"\n")
        sf.write("         ->READS                   : "+str(memifm_rd)+" ("+str(memifm_rd_zeros)+" zeros)"+"\n")
        sf.write("         ->WRITES                  : "+str(memifm_wr)+" ("+str(memifm_wr_zeros)+" zeros)"+"\n")
        sf.write("    [MEM-FILTER]: "+"\n")
        sf.write("         ->USED_CAPACITY           : "+str(max_memfilter_capacity)+"\n")
        sf.write("         ->ENERGY                  : "+str(self.mem_filter.get_energy())+e_unit+"\n")
        sf.write("         ->READS                   : "+str(memfilter_rd)+" ("+str(memfilter_rd_zeros)+" zeros)"+"\n")
        sf.write("         ->WRITES                  : "+str(memfilter_wr)+" ("+str(memfilter_wr_zeros)+" zeros)"+"\n")
        sf.write("    [MEM-PSUM]"+"\n")
        sf.write("         ->USED_CAPACITY           : "+str(max_mempsum_capacity)+"\n")
        sf.write("         ->ENERGY                  : "+str(self.mem_psum.get_energy())+e_unit+"\n")
        sf.write("         ->READS                   : "+str(mempsum_rd)+" ("+str(mempsum_rd_zeros)+" zeros)"+"\n")
        sf.write("         ->WRITES                  : "+str(mempsum_wr)+" ("+str(mempsum_wr_zeros)+" zeros)"+"\n")
        sf.write("    [Q-IFM]"+"\n")
        sf.write("         ->ENERGY                  : "+str(self.qifm.get_energy())+e_unit+"\n")
        sf.write("    [Q-FILTER]"+"\n")
        sf.write("         ->ENERGY                  : "+str(self.qfilter.get_energy())+e_unit+"\n")
        sf.write("    [Q-IN_PSUM]"+"\n")
        sf.write("         ->ENERGY                  : "+str(self.qipsum.get_energy())+e_unit+"\n")
        sf.write("    [Q-OUT_PSUM]"+"\n")
        sf.write("         ->ENERGY                  : "+str(self.qopsum.get_energy())+e_unit+"\n")
        if self.pe_type == c.SYSTOLIC:
            sf.write("    [Q-OUT_IFM]"+"\n")
            sf.write("         ->ENERGY                  : "+str(self.qoifm.get_energy())+e_unit+"\n")
            sf.write("    [Q-OUT_FILTER]"+"\n")
            sf.write("         ->ENERGY                  : "+str(self.qofilter.get_energy())+e_unit+"\n")
        sf.write("    [I-QUEUE-IFM]    max occupation: "+str(self.max_iqifm_occ)+"\n")
        sf.write("    [I-QUEUE-FILTER] max occupation: "+str(self.max_iqfilter_occ)+"\n")
        sf.write("    [I-QUEUE-PSUM]   max occupation: "+str(self.max_iqpsum_occ)+"\n")

    def get_max_cycle(self):
        return self.max_cycle

    def to_memory(self, cycle):
        if self.pe_type == c.SYSTOLIC:
            if cycle == self.to_mem_cycle:
                return self.ofm[0]
            else:
                return ""
        return self.ofm[cycle]

    def size_queue(self, id):
        if id is c.IN_IFM:
            return self.qifm.qsize()
        if id is c.IN_FILTER:
            return self.qfilter.qsize()
        if id is c.IN_PSUM:
            return self.qipsum.qsize()
        if id is c.OUT_PSUM:
            return self.qopsum.qsize()
        if id is c.OUT_IFM:
            return self.qoifm.qsize()
        if id is c.OUT_FILTER:
            return self.qofilter.qsize()
        raise

    def get_queue(self, id):
        if id is c.IN_IFM:
            return self.qifm.get()
        if id is c.IN_FILTER:
            return self.qfilter.get()
        if id is c.IN_PSUM:
            return self.qipsum.get()
        if id is c.OUT_PSUM:
            return self.qopsum.get()
        if id is c.OUT_IFM:
            return self.qoifm.get()
        if id is c.OUT_FILTER:
            return self.qofilter.get()
        raise

    def put_queue(self, id, data):
        if id is c.IN_IFM:
            return self.qifm.put(data)
        if id is c.IN_FILTER:
            return self.qfilter.put(data)
        if id is c.IN_PSUM:
            return self.qipsum.put(data)
        if id is c.OUT_PSUM:
            if data is "":
                self.print_w("put empty data into OUT_PSUM queue")
            return self.qopsum.put(data)
        if id is c.OUT_IFM:
            if data is "":
                self.print_w("put empty data into OUT_IFM queue")
            return self.qoifm.put(data)
        if id is c.OUT_FILTER:
            if data is "":
                self.print_w("put empty data into OUT_FILTER queue")
            return self.qofilter.put(data)
        raise

    def print_state(self):
        print("State of "+str(self.name))
        self.mem_ifm.print_debug()
        self.mem_filter.print_debug()
        self.mem_psum.print_debug()
        self._2smul.print_debug()
        self._1sadd.print_debug()
        self.qopsum.print_debug()
        self.qipsum.print_debug()

        if self.pe_type == SYSTOLIC:
            self.qoifm.print_debug()
            self.qofilter.print_debug()


    def print_debug(self):
        '''
        Print debuging information
        '''
        return


    def get_cycles_fullpipe(self):
        '''
        Get the number of cycles on processing one input, when the pipeline is full
        '''
        if self.c_end_full == self.c_start_full:
                self.c_end_full += 1
        return self.c_end_full - self.c_start_full + 1

    def __stats_queues(self):
        # Queues
        self.max_iqifm_occ = max(self.max_iqifm_occ, self.size_queue(c.IN_IFM))
        self.max_iqfilter_occ = max(self.max_iqfilter_occ, self.size_queue(c.IN_FILTER))
        self.max_iqpsum_occ = max(self.max_iqpsum_occ, self.size_queue(c.IN_PSUM))

    def __stats__(self, cycle):
        '''
        Update  the variables:
            1) Begin and end of full pipeline cycles
            2) Multiplier occupation
            3) Adder occupation
        '''

        #Check when the data arrives in the multiplier for the first time
        if self.c_start_full is -1:
            if self.mem_ifm.isfirst_rd(cycle) and self.mem_filter.isfirst_rd(cycle):
                self.c_start_full = cycle

        # Multiplier occupation
        if  self.c_start_full is not -1:
            if self.c_end_full is -1:
                if (self.c_start_full + 1) :
                    self.occ_mul += self._2smul.get_occ();
            elif (cycle <= (self.c_end_full + 1) and cycle >= (self.c_start_full + 1)) :
                    self.occ_mul += self._2smul.get_occ();

        # SUM occupation
        if  self.c_start_full is not -1:
            if self.c_end_full is -1:
                if(cycle >= (self.c_start_full + self._2smul.get_num_stages() + 1)) :
                    self.occ_sum += self._1sadd.get_occ();
            elif(cycle <= (self.c_end_full + self._2smul.get_num_stages() + 1) and cycle >= (self.c_start_full + self._2smul.get_num_stages() + 1)) :
                self.occ_sum += self._1sadd.get_occ();

        #Check when the data arrives in the multiplier for the last time
        if self.c_end_full is -1:
            if self.mem_ifm.islast_rd(cycle) and self.mem_filter.islast_rd(cycle):
                self.c_end_full = cycle + 1

        # For calculating the full pipeline occupation

    def advance(self, cycle):
        '''
        Advance 1 cycle
        '''
        if self.pe_type is c.NEW_PE:
            return self.advance_new(cycle)
        elif self.pe_type is c.EYERISS_PE:
            if self.pooling_forward == True:
                return self.advance_eyeriss_pooling(cycle)
            else:
                return self.advance_eyeriss(cycle)
        elif self.pe_type is c.SYSTOLIC:
            return self.advance_systolic(cycle)
        else:
            raise

    def __rd_queue(self, queue, num=-1):
        '''
        Read from the queue into array
        If num=-1, read everything
        '''
        _input = []
        if num == -1:
            qsize = self.size_queue(queue)
        else:
            size = self.size_queue(queue)
            if num <= size:
                qsize = num
            else:
                qsize = size

        for i in range(0, qsize, 1):
                element = self.get_queue(queue)
                _input.append(element)
        if len(_input) == 0:
            _input = [""]

        return _input

    def advance_eyeriss_pooling(self, cycle):
        '''
        Advance  1 cycle
        Clasical Eyeriss PE architecture
        '''
        # STATS Queues: need to be collected before get the queues
        self.__stats_queues()

        if cycle != self.prev_cycle + 1:
            print(self.name+": cycle: "+str(cycle)+" prev_cycle: "+str(self.prev_cycle))
            raise

        self.prev_cycle = cycle


        # get the sate
        # Read from the IN_IFM queue to memory
        # Read from the IN_FILTER queue to memory
        i_fmap = self.__rd_queue(c.IN_IFM)
        #i_filter = 1

        # IFMAP Memory
        _0_mul = self.mem_ifm.read()  # Read and advance
        self.mem_ifm.write(i_fmap)    # Write and advance

        if _0_mul == None:
            self.mem_ifm.print_debug()
            raise

        #################
        # Filter Memory #
        #################
        _1_mul = 1 # Read and advance
        # Multiplier
        _out_mul = self._2smul.get_output() # get the output
        self._2smul.advance(_0_mul, _1_mul) # Multiply and advance


        # Out mem PSUM
        _out_mempsum = self.mem_psum.read()
        # Multiplexer output
        if self.mux.next_state() == 1:
            _in_qpsum = self.get_queue(c.IN_PSUM)
            if _in_qpsum == "":
                self.print_d("["+str(self.name)+"] MUX is 1, but the input psum Queue is empty")
        else:
            _in_qpsum = ""
        self.mux.advance(_out_mul, _in_qpsum)
        _out_mux = self.mux.get_output() # It is after advance because it is combinatorial logic
        # Adder
        _out_add = self._1sadd.advance(_out_mux, _out_mempsum)

        # Output
        if self.out_psum[cycle]:
            self.put_queue(c.OUT_PSUM, _out_add)
        self.mem_psum.write(_out_add)

        #Debug
        if self.debug_pe:
            print("["+self.name+"]",end='')
            self._2smul.print_debug()
            self._1sadd.print_debug()
            self.mem_psum.print_debug()
            self.qopsum.print_debug()
            self.qipsum.print_debug()
            print("")

        # STATS
        self.__stats__(cycle)

        return 0

    def advance_eyeriss(self, cycle):
        '''
        Advance  1 cycle
        Clasical Eyeriss PE architecture
        '''
        # STATS Queues: need to be collected before get the queues
        self.__stats_queues()

        if cycle != self.prev_cycle + 1:
            print(self.name+": cycle: "+str(cycle)+" prev_cycle: "+str(self.prev_cycle))
            raise

        self.prev_cycle = cycle


        # get the sate
        # Read from the IN_IFM queue to memory
        # Read from the IN_FILTER queue to memory
        i_fmap = self.__rd_queue(c.IN_IFM)
        i_filter = self.__rd_queue(c.IN_FILTER)

        # IFMAP Memory
        _0_mul = self.mem_ifm.read()  # Read and advance
        self.mem_ifm.write(i_fmap)    # Write and advance

        if _0_mul == None:
            self.mem_ifm.print_debug()
            raise

        #################
        # Filter Memory #
        #################
        _1_mul = self.mem_filter.read() # Read and advance
        self.mem_filter.write(i_filter) # Write and advance

        # Multiplier
        _out_mul = self._2smul.get_output() # get the output
        self._2smul.advance(_0_mul, _1_mul) # Multiply and advance


        # Out mem PSUM
        _out_mempsum = self.mem_psum.read() # mux_psum.get_output()

        # Multiplexer output
        if self.mux.next_state() == 1:
            _in_qpsum = self.get_queue(c.IN_PSUM)
            if _in_qpsum == "":
                self.print_d("["+str(self.name)+"] MUX is 1, but the input psum Queue is empty")
        else:
            _in_qpsum = ""
        self.mux.advance(_out_mul, _in_qpsum)
        _out_mux = self.mux.get_output() # It is after advance because it is combinatorial logic
        # Adder
        _out_add = self._1sadd.advance(_out_mux, _out_mempsum)

        # Output
        if self.out_psum[cycle]:
            self.put_queue(c.OUT_PSUM, _out_add)
        self.mem_psum.write(_out_add)

        #Debug
        if self.debug_pe:
            print("["+self.name+"]",end='')
            self._2smul.print_debug()
            self._1sadd.print_debug()
            self.mem_psum.print_debug()
            self.qopsum.print_debug()
            self.qipsum.print_debug()
            print("")

        # STATS
        self.__stats__(cycle)

        return 0

    def put_filter_mem(self, element):
        self.mem_filter.write(element)

    def advance_systolic(self, cycle):
        '''
        Advance  1 cycle
        Clasical Systolic Array PE architecture
        '''
        # STATS Queues: need to be collected before get the queues
        self.__stats_queues()

        if cycle != self.prev_cycle + 1:
            print(self.name+": cycle: "+str(cycle)+" prev_cycle: "+str(self.prev_cycle))
            raise

        self.prev_cycle = cycle

        # get the sate
        # Read from the IN_IFM queue to memory
        # Read from the IN_FILTER queue to memory
        # read just one element from the queue

        if self.pos_w != 0:
            [_1_mul] = self.__rd_queue(c.IN_FILTER, 1) # read just one element from theRRRUUU
            self.mem_filter.read()  # Read and advance
        else:
            _1_mul = self.mem_filter.read()  # Read and advance
        _0_mul = self.mem_ifm.read()  # Read and advance

        # Multiplier
        _out_mul = self._2smul.get_output() # get the output

        self._2smul.advance(_0_mul, _1_mul) # Multiply and advance

        # Next cycle
        in_add = self.reg_add
        self.reg_add = _out_mul

        # Out mem PSUM
        _out_mem_psum = self.mem_psum.read() #= mux_psum.get_output()

        _out_add = self._1sadd.advance(_out_mem_psum, in_add)

        # It should be controlled by the signals
        self.mem_psum.write(_out_add)

        # Output IFM
        if not self.limit_bottom:
            if _0_mul != "" and (_0_mul is not None):
                self.put_queue(c.OUT_IFM, _0_mul)

        # Output FILTER
        if not self.limit_right:
            if _1_mul != "" and (_1_mul is not None):
                self.put_queue(c.OUT_FILTER, _1_mul)

        # Output to Memory
        if (cycle+1-self.pos_w) == self.to_mem_cycle:
                self.put_queue(c.OUT_PSUM, _out_add)

        #Debug
        if self.debug_pe:
            print("["+self.name+"]",end='')
            self._2smul.print_debug()
            self._1sadd.print_debug()
            self.mem_psum.print_debug()
            self.qopsum.print_debug()
            self.qipsum.print_debug()
            self.qoifm.print_debug()
            self.qofilter.print_debug()
            print("")

        # STATS
        self.__stats__(cycle)

        return 0

