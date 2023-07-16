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

import numpy as np

class agen(object):
    '''
    Address generator
    '''
    def __init__(self, seq_rd, seq_wr, name, delay = -1, max_comp = -1):
        if seq_wr == []:
            self.systolic_ifm_filter= True
        else:
            self.systolic_ifm_filter= False
            self.seq_wr = seq_wr
            self.size_wr = len(seq_wr)

        if seq_rd == []:
            self.systolic_psum= True
        else:
            self.systolic_psum= False
            self.seq_rd = seq_rd
            self.size_rd = len(seq_rd)

        if seq_rd == [] and seq_wr == []:
            self.fix_pos = True
        else:
            self.fix_pos = False

        if self.systolic_ifm_filter:
            self.seq_wr = self.seq_rd
            self.size_wr = len(seq_wr)
        if self.systolic_psum:
            self.seq_rd = self.seq_wr
            self.size_rd = len(seq_rd)

        self.num_reads = 0
        self.max_comp = max_comp
        self.p_rd = 0
        self.p_rd_cycle = 0
        self.p_wr = 0
        self.p_wr_cycle = 0
        self.last = "" # To identify the last element to be used
        self.remaining_comp = max_comp
        if delay != -1:
            self.cycles_to_start = delay
            self.unique_pos = delay # different to -1
        else:
            self.unique_pos = -1
        self.name = name

        # Calculate first and last read
        self.first_rd = -1
        self.last_rd  = -1
        end  = -1
        for cycle in range(len(seq_rd)):
            if self.seq_rd[cycle] != -1:
                if self.first_rd == -1:
                    self.first_rd = cycle
                self.last_rd  = cycle

    def next_rd(self):
        ret_res = 0
        if self.unique_pos == -1:
            if self.systolic_psum:
                res = self.seq_rd[self.p_rd+1]
            else:
                if self.max_comp != -1:
                    if self.remaining_comp != 0:
                        res = self.seq_rd[self.p_rd]
                        self.remaining_comp -= 1
                    else:
                        res = c.NAN
                else:
                    res = self.seq_rd[self.p_rd]
        else:
            if self.cycles_to_start >= 0 or self.remaining_comp <= 0:
                self.cycles_to_start -= 1
                res = c.NAN
            else:
                self.remaining_comp -= 1
                self.num_reads +=1
                res = 0
        if type(res) == type([]):
            ret_res = res[self.p_rd_cycle]
            if self.p_rd_cycle == len(res)-1:
                self.p_rd_cycle=0
                self.p_rd +=1
            else:
                self.p_rd_cycle += 1
        else:
            self.p_rd +=1
            self.p_wr_cycle=0
            ret_res = res

        if self.p_rd == self.size_rd:
            self.p_rd = 0
        return ret_res

    def next_wr(self):


        ret_res = 0
        if self.unique_pos == -1:
            # case for initialized values
            if len(self.seq_wr) == 0:
                # Initialization
                res = self.p_wr
                self.p_wr += 1
                return res
            res = self.seq_wr[self.p_wr]
        else:
            # Systolic array
            if self.cycles_to_start >= 1 or self.remaining_comp <= 0:
                res = c.NAN
            else:
                res = 0

        if type(res) == type([]):
            ret_res = res[self.p_wr_cycle]
            if self.p_wr_cycle == len(res)-1:
                self.p_wr_cycle=0
                self.p_wr +=1
            else:
                self.p_wr_cycle += 1
        else:
            self.p_wr +=1
            self.p_wr_cycle=0
            ret_res = res

        if self.p_wr == self.size_wr:
            self.p_wr = 0
        return ret_res

    # Used for calculating stats
    def num_accesses(self):
        '''
        Return the number of accesses to the memory (reads and write)
        '''
        if self.max_comp != -1:
            return self.max_comp

        srd = self.seq_rd
        swr = self.seq_wr
        srd = np.asarray(srd)
        swr = np.asarray(swr)
        srd = [x for x in srd if ~np.isnan(x)]
        swr = [x for x in swr if ~np.isnan(x)]
        return len(srd)+len(swr)

    def get_active_seq(self):
        '''
        Get the number of cycles that the memory is active
        and the idle cycles in the active section
        '''
        active = 0
        idle = 0
        isactive = False
        for i in range(len(self.seq_rd)):
            if self.isfirst_rd(i):
                isactive = True
            if isactive:
                if self.seq_rd[i] is not c.NAN:
                    active +=1
                else:
                    idle +=1
            if self.islast_rd(i):
                isactive = False
        return active, idle

    def islast_rd(self,cycle):
        '''
        Check if the read in this cycle is the last one
        Useful for calculating the full pipeline cycles
        '''
        if cycle == self.last_rd -1:
            return True
        return False

    def isfirst_rd(self,cycle):
        '''
        Check if the read in this cycle is the firs one
        Useful for calculating the full pipeline cycles
        '''
        if cycle == self.first_rd:
            return True
        return False

class memory(object):
    '''
    Memory
    '''
    def __init__(self,size, seq_rd, seq_wr, name, debug=False, unique_pos = -1, max_comp = -1):
        self.size = size
        # Define and initialize an empty the array
        self.array = [None for x in range(self.size)]
        self.buffer = 0
        self.addr_gen = agen(seq_rd, seq_wr, name, unique_pos, max_comp)
        self.debug = debug
        self.name = name
        self.e_active = active_energy(c.E_SPAD)
        self.e_idle = idle_energy(c.E_SPAD)
        self.energy = 0

        # For stats
        self.zeros_rd  = 0
        self.zeros_wr  = 0
        self.writes = 0
        self.reads = 0
        self.max_capacity = 0

    def print_debug(self):
        if self.debug:
            print("["+self.name+"] ", end='')
            for x in range(self.size):
                if self.array[x] is not "" and self.array[x] is not None:
                    print(self.array[x]+" ", end='')

    def get_energy(self):
        '''
        Return the total energy consumed during  1 ofm generation (full pipeline)
        '''
        return self.energy

    def get_reads(self):
        '''
        Return the reads and the zero reads
        '''
        return self.reads, self.zeros_rd

    def get_writes(self):
        '''
        Return the writes and the zero writes
        '''
        return self.writes, self.zeros_wr

    def get_active_seq(self):
        '''
        Return the active and idle cycles when the pipeline is full
        '''
        return self.addr_gen.get_active_seq()

    def get_capacity(self):
        '''
        Return the maximum capacity used
        '''
        return self.max_capacity

    def write(self, data):
        '''
        Write data
        '''
        if type(data) is not type([]):
            data = [data]

        for d in range(len(data)):
            pos = self.addr_gen.next_wr()
            if pos is c.NAN:
                # Idle energy is already added in the read function
                # This function is accessed every cycle
                return -1
            if pos >= self.size:
                print("[WARNING] oveflow of the "+self.name+" memory capacity. Size: "+str(self.size)+", pos: "+str(pos))
                assert(pos < self.size)

            pos = int(pos)
            # Capacity stats
            if self.array[pos] == None:
                # update the maximum capacity
                self.max_capacity += 1
            # Access stats
            if data[d] == 0 or data[d] == '':
                self.zeros_wr += 1
            # Write the data into memory
            self.writes += 1
            # Energy stats
            self.energy += self.e_active
            self.array[pos] = data[d]
            assert data[d] != None
        return 0


    def read(self):
        '''
        Read data
        '''
        pos = int(self.addr_gen.next_rd())

        if pos is c.NAN:
            # This function is accessed every cycle
            self.energy += self.e_idle # We count this only once in read()
            return ""

        if pos >= self.size:
            print("[WARNING] oveflow of the "+self.name+" memory capacity. Size: "+str(self.size)+", pos: "+str(pos))
            assert(pos < self.size)
        value = self.array[pos]

        # Enery stats
        self.energy += self.e_active

        # Access stats
        if value == 0 or value == '':
            # Zeros read value.
            self.zeros_rd += 1

        # Write the data into mem
        self.reads  += 1

        return value

    # Functions used for stats
    def islast_rd(self, cycle):
        return self.addr_gen.islast_rd(cycle)

    def isfirst_rd(self, cycle):
        return self.addr_gen.isfirst_rd(cycle)
