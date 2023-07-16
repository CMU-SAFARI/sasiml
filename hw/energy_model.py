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
    Current energy parameters are taken from "Computing's Energy Problem (and what we can do about it)" Mark Horowitz, 2014 International Solid-State Circuits Conference
    Technology: 45nm, 0.9V
    MUL (FP):
        - 16 bits: 1.1pJ
        - 32 bits: 3.7pJ
    ADD (FP):
        - 16 bits: 0.4pJ
        - 32 bits: 0.9pJ

    Memory:
        - 8KB:  10pJ
        - 32KB: 20pJ
        - 1MB:  100pJ
        - DRAM: 1.3-2.6nJ


'''
import hw.constants as c

#class energy_model:
#'''
#Energy model
#'''
# Energy consumed by the memory  (Registers at the PE and global buffer)
# nJ per access, depending on the size of the memory
mem_active = {
    156000: 30, # GB
    52012: 2, # ifm
    52224: 5, # filter
    52024: 3, # psum
    4096: 20,
    224:  5, # filter
    64:   4, # queues
    24:   3, # psum
    12:   2,  # ifm
    4:    1,
}
mem_idle = {
    156000: 0, # GB
    52012: 0, # ifm
    52224: 0, # filter
    52024: 0, # psum
    4096: 0, #
    224:  0,  #
    64:   0, # psum
    24:   0,  #
    12:   0,
    4:    0,
}

# Energy consumed by each Multiplier operation
# pJ per access, depending on the number of stages
mul_active = {
    2: 1.1
}
mul_idle = {
    2: 0,
}

# Energy consumed by each Sum operation
# pJ per access, depending on the number of stages
sum_active = {
    1: 0.4
}
sum_idle = {
    1: 0,
}


# SPAD (Register file)
e_spad  = 0.9 #pJ
# Energy of each hop: check this
e_link = e_spad * 2
e_queue = e_spad
# Global buffer
e_gb = e_spad * 6 #pJ
# DRAM access energy
e_dram = e_spad * 200

def active_energy(element):
    if element is c.E_SPAD:
        return e_spad
    if element is c.E_GB:
        return e_gb
    if element is c.E_DRAM:
        return e_dram
    if element is c.E_LINK:
        return e_link
    if element is c.E_QUEUE:
        return e_link
    if element is c.MUL:
        return mul_active[2]
    if element is c.SUM:
        return sum_active[1]
    if element is c.LINK:
        return e_link
    raise

def idle_energy(element):
    if element is c.E_SPAD:
        return 0
    if element is c.E_GB:
        return 0
    if element is c.E_DRAM:
        return 0
    if element is c.E_LINK:
        return 0
    if element is c.E_QUEUE:
        return 0
    if element is c.MUL:
        return mul_idle[2]
    if element is c.SUM:
        return sum_idle[1]
    raise

def dram_access_energy():
    return e_dram
