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
class mux2(object):
    def __init__(self, seq, name, debug=False):
        self.input = [0] * 2
        self.select = 0
        self.seq = seq
        self.size_seq = len(seq)
        self.seq_p = 0
        self.debug = debug
        self.name = name

    def state(self):
        '''
        Return the state of the mux
        '''
        return self.seq[self.seq_p]

    def next_state(self):
        '''
        Return the state of the mux
        '''
        next_seq = self.seq_p
        next_seq += 1
        if next_seq == self.size_seq:
            next_seq = 0
        return self.seq[next_seq]

    def advance(self, data1, data2):
        '''
        advance to the next cycle
        '''
        self.input[0] = data1
        self.input[1] = data2

        self.seq_p += 1
        if self.seq_p == self.size_seq:
            self.seq_p = 0
        self.select = self.seq[self.seq_p]

        return self.seq[self.seq_p]

    def get_output(self):
        '''
        New finished sum
        '''
        return self.input[self.select]
