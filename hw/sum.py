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


class sum(object):
    def __init__(self, num_stages, name, debug=False):
        self.num_stages = num_stages
        self.stage = ["" for x in range(self.num_stages)]
        self.buff = [0] * 2
        self.debug = debug
        self.name = name

        # for stats
        self.first = -1
        self.last  = -1
        self.occ   = False
        # Energy
        self.e_active = active_energy(c.SUM)
        self.e_idle   = idle_energy(c.SUM)
        self.energy   = 0

    def get_num_stages(self):
        '''
        Return the number of stages
        '''
        return self.num_stages

    def get_occ(self):
        '''
        Return 1 if it is busy in the first mul stage
        '''
        if self.stage[0] is "":
            return 0
        return 1

    def get_energy(self):
        '''
        Return the energy
        '''
        return self.energy

    def get_output(self):
        '''
        New finished sum
        '''
        return self.stage[self.num_stages-1]

    def print_debug(self):
        if self.debug:
            print("[SUM] ", end='')
            for s in range(self.num_stages):
                print(""+str(s)+":"+str(self.stage[s])+" ", end='')

    def advance(self, data1, data2):
        '''
        Advance 1 cycle
        '''
        self.occ = True

        for x in range(self.num_stages-1, 0, -1):
            self.stage[x] = self.stage[x-1]

        if data1 is "" and data2 is "" and data2 is not None and data1 is not None:
            self.stage[0] = "" # Idle
            self.energy += self.e_idle
            self.occ = False
        if data1 is "" and data2 is not "" and data2 is not None and data1 is not None:
            if type(data2) is str:
                self.stage[0] = str(data2)
            else:
                self.stage[0] = data2

            self.energy += self.e_active
        if data2 is "" and data1 is not "" and data2 is not None and data1 is not None:
            if type(data1) is str:
                self.stage[0] = str(data1)
            else:
                self.stage[0] = data1
            self.energy += self.e_active
        if data2 is not "" and data1 is not "" and data2 is not None and data1 is not None:
            if type(data2) is str or type(data1) is str:
                self.stage[0] = str(data1)+"+"+str(data2)
            else:
                self.stage[0] = data1+data2

            self.energy += self.e_active

        return self.stage[self.num_stages-1]

    def get_occupation(self):
        '''
        Return the ocupation of the adder
        '''
        if self.occ:
            return 1
        return 0

