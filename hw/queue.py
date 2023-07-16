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
from queue import Queue
from numpy import nan
from hw.energy_model import active_energy
from hw.energy_model import idle_energy
import hw.constants as c

class queue(object):
    def __init__(self, name, size, debug=False, energy_model=True):
        self.debug = debug
        self.size = size
        self.queue = Queue(size)
        self.name = name
        self.reads  = 0
        self.writes = 0
        if energy_model:
            self.e_active_queue= active_energy(c.E_QUEUE)
            self.e_idle_queue = idle_energy(c.E_QUEUE)
        else:
            self.e_active_queue= 0
            self.e_idle_queue = 0
        self.energy = 0

    def qsize(self):
        return self.queue.size()

    def put(self, data):
        if data is "" or data is c.NAN:
            return

        if not self.queue.full():
            self.energy += self.e_active_queue
            self.writes +=1
            self.queue.put(data)
        else:
            print("[ERROR][" + self.name + "] FULL QUEUE ("+str(self.qsize())+")")
            self.print_debug()
            raise

    def get(self):
        if not self.queue.empty():
            self.energy += self.e_active_queue
            self.reads +=1
            return self.queue.get()
        else:
            return ""

    def full(self):
        return self.queue.full()

    def qsize(self):
        return self.queue.qsize()

    def empty(self):
        return self.queue.empty()

    def print_debug(self):
        if self.debug:
            temp = Queue(self.size)
            print("["+self.name + "] ", end='')
            while not self.queue.empty():
                elem = self.queue.get()
                temp.put(elem)
                if elem is c.NAN:
                    print("c.NAN,", end='')
                else:
                    print(str(elem)+",", end='')
            self.queue = temp

    def get_energy(self):
        '''
        Return the total energy consumed during  1 ofm generation (full pipeline)
        '''
        return self.energy

    def get_reads(self):
        '''
        '''
        return self.reads

    def get_writes(self):
        '''
        '''
        return self.writes
