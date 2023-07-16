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
from hw.energy_model import active_energy
from hw.energy_model import idle_energy
import hw.constants as c
class gbuffer(object):
    def __init__(self, ifmap, filter, size,  debug=False):
        self.ifmap = ifmap
        self.filter = filter
        self.debug = debug
        self.num_filter = 0
        self.num_filter_zero = 0
        self.num_filter_nonzero = 0
        self.num_ifm = 0
        self.num_ifm_zero = 0
        self.num_ifm_nonzero = 0
        self.num_ofm = 0
        self.ofm = {}
        self.e_active = active_energy(c.E_GB)
        self.e_idle = idle_energy(c.E_GB)
        self.energy = 0
        self.nones = 0

    def get_ifm_elements(self):
        return self.num_ifm_zero + self.num_ifm_nonzero
        #return self.num_ifm_nonzero

    def get_filter_elements(self):
        return self.num_filter_zero + self.num_filter_nonzero
        #return self.num_filter_nonzero

    def get_ofm_elements(self):
        return self.num_ofm

    def get_reads_ifm(self, zero=False):
        '''
        Get the total number of reads
        '''
        if zero:
            return self.num_ifm_zero
        else:
            return self.num_ifm_nonzero

    def get_reads_filter(self, zero=False):
        '''
        Get the total number of reads
        '''
        if zero:
            return self.num_filter_zero
        else:
            return self.num_filter_nonzero

    def get_writes(self):
        '''
        Get the total number of writes
        '''
        return self.num_ofm

    def read_ifm(self):

        if self.num_ifm >= len(self.ifmap):
            return None

        res = self.ifmap[self.num_ifm]
        self.num_ifm += 1
        if res == None:
            self.nones +=1
            return None

        if res != 0:
            self.num_ifm_nonzero += 1
            self.energy += self.e_active
        else:
            self.num_ifm_zero += 1
            self.energy += self.e_idle
        return res

    def read_filter(self):
        if self.num_filter >= len(self.filter):
            return None

        res = self.filter[self.num_filter]

        if res == None:
            return None

        self.num_filter += 1
        if res != 0:
            self.num_filter_nonzero += 1
            self.energy += self.e_active
        else:
            self.num_filter_zero += 1
            self.energy += self.e_idle
        return res

    def write_ofm(self, addr, data):
        self.ofm[addr] = data
        self.num_ofm += 1
        self.energy += self.e_active
        if self.debug:
            print("GBUF [WR]["+str(addr)+"]["+str(data)+"]")

    def print_ofm(self):
        for key, value in self.ofm.items():
            print("["+key+"] "+str(value))

    def get_energy(self):
        '''
        Return the total energy consumed during  1 ofm generation (full pipeline)
        '''
        return self.energy
