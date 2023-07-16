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
import numpy as np

class multiplier(object):
    def __init__(self, num_stages, zero_clockgate, name, debug=False):
        self.num_stages = num_stages
        self.stage = ["" for x in range(self.num_stages)]
        self.debug = debug
        self.mul_by_zero = 0
        self.mul = 0
        self.name = name
        # Energy
        self.e_active = active_energy(c.MUL)
        self.e_idle   = idle_energy(c.MUL)
        self.energy   = 0
        self.zero_clockgate = zero_clockgate


    def get_energy(self):
        '''
        Return the energy
        '''
        return self.energy

    def get_num_stages(self):
        '''
        Return the number of stages of this multiplier
        '''
        return self.num_stages

    def get_occ(self):
        '''
        Return 1 if it is busy in the first mul stage
        '''
        if self.stage[0] is "":
            return 0
        return 1

    def get_mul_by_zero(self):
        '''
        Return the number of multiplications by zero
        '''
        return self.mul_by_zero

    def get_mul(self):
        '''
        Return the number of multiplications
        '''
        return self.mul

    def get_output(self):
        '''
        Get the output
        '''
        return self.stage[self.num_stages-1]

    def print_debug(self):
        if self.debug:
            print("[MUL] ", end='')
            for s in range(self.num_stages):
                print(str(s)+":"+str(self.stage[s])+" ", end='')

    def advance(self, data1, data2):
        '''
        Advance 1 cycle
        '''
        for x in range(self.num_stages-1, 0, -1):
            self.stage[x] = self.stage[x-1]

        if data1 is "" or data2 is "":
            self.energy += self.e_idle
            self.stage[0] = ""
        elif (data1 == 0) or (data2 == 0) or (data1 == None) or (data2 == None):
            if self.zero_clockgate is True:
                self.energy += self.e_idle
            else:
                self.energy += self.e_active
            self.stage[0] = ""    # In multiplications by zero, the multiplier is clock gated. So, no energy comsumption

            if (data1 == 0) or (data2 == 0):
                self.mul_by_zero += 1
                self.mul += 1 # number of multiplications
        else:
            self.energy += self.e_active
            self.mul += 1 # number of multiplications

            if type(data2) is str or type(data1) is str:
                self.stage[0] = str(data2) + "*" + str(data1)
            elif type(data2) is np.str_ or type(data1) is np.str_:
                if data1 == "0" or data2 == "0":
                    self.mul_by_zero += 1
                    self.stage[0] = ""    # In multiplications by zero, the multiplier is clock gated. So, no energy comsumption
                else:
                    self.stage[0] = str(data2) + "*" + str(data1)
            else:
                print("data1 type: "+str(type(data1)))
                print("data1: "+str(data1))
                print("data2 type: "+str(type(data2)))
                print("data2: "+str(data2))

                self.stage[0] = data2*data1

