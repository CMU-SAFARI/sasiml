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
import timeit
import numpy
from numpy import nan
import numpy as np
import modules.constants as s

from modules.sanity_check import check_dimension

from modules.common import del_nan
from modules.common import index_nonan
from modules.common import index_nonan2
from modules.common import all_same_size
from modules.common import fill_1Darray
from functools import reduce

import sys
sys.path.insert(0,'..')
import hw.constants as c

# Initialize the memories or distribute the data?
initialize_memories = True

# Print debugging information
debug_mode = False

class multiply(object):
    '''
    Maps a regular Matrix Multiplication to a systolic array
    To keep the name of the variables the same, we do the following mapping:
    1) ifm    -> propagate vertically. Mat 1
    1) filter -> propagate horizontally. Mat 2
    '''

    def __init__(self, mat1, mat2, ofm, _filter_size, hw, pe_type, num_channels, num_filters, batch):
        if pe_type != s.SYSTOLIC:
            print("PE_TYPE: "+str(pe_type))
            raise
        self.pe_type = pe_type

        # Hardware configuration
        self.hw = hw

        assert(self.hw[c.IFM_BW] == 1)
        assert(self.hw[c.FIL_BW] == 1)
        assert(self.hw[c.OFM_BW] == 1)

        # Filter size. To calculate when to output to memory
        self.filter_size = _filter_size
        self.ofm_size    = len(ofm)*len(ofm[0])

        self.num_channels = num_channels
        self.num_filters  = num_filters
        self.batch        = batch

        # Dimensions of the PE array
        self.array_w = len(mat1[0])
        self.array_h = len(mat2[0])
        self.ofm     = ofm

        # The actual ifmap, filter, ofmap and stride
        self.mat1 = mat1
        self.mat2 = mat2

        # Create the data structures to save the signals
        self.mem_ifm_wr         = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mem_ifm_rd         = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mem_filter_wr      = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mem_filter_rd      = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mem_psum_wr        = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mem_psum_rd        = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mux_seq            = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.out_psum           = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.out_ifm            = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.out_filter         = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.ofm_seq            = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.multicast_filter   = [[0  for w in range(self.array_w)] for h in range(self.array_h)]

        # Initial values on IFM and Filter memories
        self.mem_ifm_init           = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mem_filter_init        = [[[] for w in range(self.array_w)] for h in range(self.array_h)]

        # Multicast groups
        self.num_groups_ifm   = self.array_w
        self.num_groups_fil   = 1
        self.pe_ifm_group     = [[] for i in range(self.num_groups_ifm)]
        self.pe_fil_group     = [[] for i in range(self.num_groups_fil)]

        # Multicast sequences
        self.ifm_seq_multicast = []
        self.filter_seq_multicast = []




    def print_d(self, text):
        '''
        Print text only in debug mode
        '''
        if debug_mode:
            print(text)

    def __gen_multicast_groups(self):
        '''
        Generate the ifmap and filter.
        ifmap: mat1. It goes to the first row.
        filter: mat2. It goes to the first column
        '''
        # Create multicast groups
        # Distribute IFM horizontally
        group = 0
        for w in range(self.array_w):
            self.pe_ifm_group[group].append([0,w])
            group +=1

        # Distribute Filter vertically
        group = 0
        #for h in range(self.array_h):
        for w in range(self.array_w):
            self.pe_fil_group[group].append([0,w])

        if debug_mode:
            for i in range(self.array_w):
                print("multicast_ifm["+str(i)+"] "+str(self.pe_ifm_group[i]))
            print("fil[0] "+str(self.pe_fil_group[0]))

    def __gen_mem_ifm_wr(self):
        '''
        Generate mem_ifm_wr signals (mat1)
        Data writen in the internal ifm PE registers from the global buffer
        '''
        cycle = 0
        same_cycle = self.hw[c.IFM_BW]-1 #Simulate BW
        w_bw = 0
        for w in range(len(self.mat1[0])):
            # Simulate bw > 1
            if same_cycle < self.hw[c.IFM_BW]-1:
                same_cycle += 1
            else:
                w_bw = w
                same_cycle = 0

            for h in range(len(self.mat1)):
                # Simulate BW > 1
                cycle = int(w_bw/self.hw[c.IFM_BW])*len(self.mat1)+h
                # init this cycle to nan
                if same_cycle == 0:
                    for y in range(self.array_w):
                        for x in range(self.array_h):
                            self.mem_ifm_wr[x][y].append([s.NAN])
                # update the PEs that receive values
                for g in self.pe_ifm_group[w]:
                    # Save values in consequtive positions
                    if self.mem_ifm_wr[g[0]][g[1]][cycle] == [s.NAN]:
                        self.mem_ifm_wr[g[0]][g[1]][cycle] = [0]#
                    else:
                        self.mem_ifm_wr[g[0]][g[1]][cycle].append(0)#

                # Only one ifm multicast group
                if same_cycle == 0:
                    self.ifm_seq_multicast.append([0])
                else:
                    self.ifm_seq_multicast[cycle].append(0)
        if debug_mode:
            for x in range(len(self.mem_ifm_wr)):
                for y in range(len(self.mem_ifm_wr[0])):
                    print("mem_ifm_wr["+str(x)+"]["+str(y)+"]    "+str(self.mem_ifm_wr[x][y]))

    def __init_mem_ifm(self):
        '''
        Initialize the memory of the PEs
        '''
        pos = [[0 for x in range(self.array_w)] for y in range(self.array_h)]
        for w in range(len(self.mat1[0])):
            for h in range(len(self.mat1)):
                for g in self.pe_ifm_group[w]:
                    # Save values in consequtive positions
                    if self.mat1[h][w] != None:
                        self.mem_ifm_init[g[0]][g[1]].append(self.mat1[h][w]) #

        if debug_mode:
            for x in range(len(self.mem_ifm_init)):
                for y in range(len(self.mem_ifm_init[0])):
                    print("mem_ifm_init["+str(x)+"]["+str(y)+"]    "+str(self.mem_ifm_init[x][y]))

    def __gen_mem_filter_wr(self):
        '''
        Generate mem_filter_wr signals
        Data writen in the internal filter PE registers from the global buffer
        '''
        cycle = 0
        # Keeps the position where the data is written
        same_cycle = self.hw[c.FIL_BW]-1 # Simulate BW
        w_bw = 0
        for h in range(len(self.mat2)):
            # Simulate bw > 1
            if same_cycle < self.hw[c.FIL_BW]-1:
                same_cycle += 1
            else:
                h_bw = h
                same_cycle = 0

            for w in range(len(self.mat2[0])):
                # Simulate BW > 1
                cycle = int(h_bw/self.hw[c.FIL_BW])*len(self.mat2[0])+w
                # init this cycle to nan
                if same_cycle == 0:
                    for y in range(self.array_w):
                        for x in range(self.array_h):
                            self.mem_filter_wr[x][y].append([s.NAN])
                # update the PEs that receive values
                for g in self.pe_fil_group[w]:
                    if self.mem_filter_wr[g[0]][g[1]][cycle] == [s.NAN]:
                        self.mem_filter_wr[g[0]][g[1]][cycle] = [0]#
                    else:
                        self.mem_filter_wr[g[0]][g[1]][cycle].append(0)#
                    # Write the elements of the filter in consequtive positions
                if same_cycle == 0:
                    self.filter_seq_multicast.append([w])
                else:
                    self.filter_seq_multicast[cycle].append(w)

        if debug_mode:
            for x in range(len(self.mem_filter_wr)):
                for y in range(len(self.mem_filter_wr[0])):
                    print("mem_filter_wr["+str(x)+"]["+str(y)+"] "+str(self.mem_filter_wr[x][y]))

    def __init_mem_filter(self):
        '''
        Initialize the filter memory
        '''
        pos = [[0 for x in range(self.array_w)] for y in range(self.array_h)]
        for w in range(len(self.mat2[0])):
            for h in range(len(self.mat2)):
                # update the PEs that receive values
                for g in self.pe_fil_group[w]:
                    if self.mat1[h][w] != None:
                        self.mem_filter_init[g[0]][g[1]].append(self.mat2[h][w]) #
                    # Write the elements of the filter in consequtive positions
        #if debug_mode:
        for x in range(len(self.mem_filter_init)):
            for y in range(len(self.mem_filter_init[0])):
                print("mem_filter_init["+str(x)+"]["+str(y)+"]    "+str(self.mem_filter_init[x][y]))


    def __index_window(self, window, array, idx_begin=False):

        '''
        Calculate the index of the end of the window
        By default, returning the index of the last element of the window
        '''
        count = 0
        wcount = 0
        window_size = len(self.fil[0])
        if idx_begin == False:
            for i in range(0, len(array), 1):
                if array[i] is not s.NAN:
                    count += 1
                    if count == window_size:
                        if  wcount == window:
                            return i
                        wcount += 1
                        count = 0
        elif idx_begin:
            for i in range(len(array)):
                if array[i] is not s.NAN:
                    if count == 0:
                        # begining of the window
                        bwin = i
                    count +=1
                    if count == window_size:
                        if  wcount == window:
                            return bwin
                        wcount += 1
                        count = 0

        print("[ERROR] the simulator shouldn't readch this point")
        raise

    def __adjust_array_same_size(self, array, val=s.NAN):
        '''
        Adjust all the vectors to have the same size by introducing val at the end
        '''
        max_len = 0
        for x in range(self.array_h):
            for y in range(self.array_w):
                max_len = max(max_len, len(array[x][y]))
        for x in range(self.array_h):
            for y in range(self.array_w):
                for i in range(max_len-len(array[x][y])):
                    array[x][y].append(val)

    def __gen_mem_ifm_fil_rd2(self):
        '''
        Generate mem_ifm_rd and mem_fil_rd
        '''
        # Get the values in the same order they were writen in memory
        for y in range(self.array_w):
            # For each individual PE
            # Populate the mul first, with no nan
            for i in range(len(self.mat1)):
                self.mem_ifm_rd[0][y].append(i)

            for a in range(len(self.mat1)):
                self.mem_filter_rd[0][y].append(a)

        self.mem_filter_rd[0][0].append(-1)
        self.mem_filter_rd[0][0].append(-1)
        self.mem_filter_rd[0][0].append(-1)
        self.mem_filter_rd[0][0].append(-1)
        self.mem_filter_rd[0][0].append(-1)
        self.mem_filter_rd[0][0].append(-1)
        self.mem_ifm_rd[0][0].append(-1)
        self.mem_ifm_rd[0][0].append(-1)
        self.mem_ifm_rd[0][0].append(-1)
        self.mem_ifm_rd[0][0].append(-1)
        self.mem_ifm_rd[0][0].append(-1)

        # Makes all the arrays the same size
        self.__adjust_array_same_size(self.mem_ifm_rd)
        self.__adjust_array_same_size(self.mem_filter_rd)

        if debug_mode:
            for y in range(len(self.mem_filter_rd[0])):
                for x in range(len(self.mem_filter_rd)):
                    print("mem_filter_rd["+str(x)+"]["+str(y)+"] "+str(self.mem_filter_rd[x][y]))
            for y in range(len(self.mem_ifm_rd[0])):
                for x in range(len(self.mem_ifm_rd)):
                    print("mem_ifm_rd["+str(x)+"]["+str(y)+"]    "+str(self.mem_ifm_rd[x][y]))


    def __gen_mem_ifm_fil_rd(self):
        '''
        Generate mem_ifm_rd and mem_fil_rd
        '''
        # Get the values in the same order they were writen in memory
        filter    = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        ifm       = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        for x in range(self.array_h):
            for y in range(self.array_w):
                #plain_ifm = []
                for g in range(len(self.mem_ifm_wr[x][y])):
                    for b in range(len(self.mem_ifm_wr[x][y][g])):
                        if (self.mem_ifm_wr[x][y][g][b] is not s.NAN) and (self.mem_ifm_wr[x][y][g][b] != [s.NAN]):
                            ifm[x][y].append(self.mem_ifm_wr[x][y][g][b])
                for g in range(len(self.mem_filter_wr[x][y])):
                    for b in range(len(self.mem_filter_wr[x][y][g])):
                        if (self.mem_filter_wr[x][y][g][b] is not s.NAN) and (self.mem_filter_wr[x][y][g][b] != [s.NAN]):
                            filter[x][y].append(self.mem_filter_wr[x][y][g][b])

        # Generate read signals, without bubbles
        for x in range(self.array_h):
            for y in range(self.array_w):
                # For each individual PE
                # Populate the mul first, with no nan
                cycle = 0
                len_filter = len(filter[x][y])
                for i in range(len_filter):
                    self.mem_ifm_rd[x][y].append(i)
                    self.mem_filter_rd[x][y].append(i)
                    cycle +=1
        # Makes all the arrays the same size
        self.__adjust_array_same_size(self.mem_ifm_rd)
        self.__adjust_array_same_size(self.mem_filter_rd)



        if debug_mode:
            for y in range(len(self.mem_filter_rd[0])):
                for x in range(len(self.mem_filter_rd)):
                    print("mem_filter_rd["+str(x)+"]["+str(y)+"] "+str(self.mem_filter_rd[x][y]))
            for y in range(len(self.mem_ifm_rd[0])):
                for x in range(len(self.mem_ifm_rd)):
                    print("mem_ifm_rd["+str(x)+"]["+str(y)+"]    "+str(self.mem_ifm_rd[x][y]))

    def __gen_mem_psum(self):
        '''
        1) Generate mem_psum_wr
        We write in PSUM all the elements of the window
        with the exception of the last row of elements, which do not save the last element of the window
        2) Generate mem_psum_rd: it is just  mem_psum_wr shifted by one
        '''
        pos_mem_psum_wr = 0
        delay = self.hw[c.MUL] + self.hw[c.SUM]  # Delay from PE to PE
        for x in range(self.array_h):
            for y in range(self.array_w):
                for d in range(delay):
                    self.mem_psum_wr[x][y].append(c.NAN) # write always in possition 0
                for i in range(len(self.mat1)):
                    self.mem_psum_wr[x][y].append(0) # write always in possition 0
                self.mem_psum_wr[x][y].append(c.NAN) # write always in possition 0

        if debug_mode:
            for x in range(self.array_h):
                for y in range(self.array_w):
                    print("mem_psum_wr["+str(x)+"]["+str(y)+"]   "+str(self.mem_psum_wr[x][y]))
            for x in range(self.array_h):
                for y in range(self.array_w):
                    print("mem_psum_rd["+str(x)+"]["+str(y)+"]   "+str(self.mem_psum_rd[x][y]))

    def __gen_ofm_seq(self):
        '''
        Generate the sequence of outputs, cycle by cycle
        Only the first row generate outputs
        This is the same sequence than mux
        '''
        # All PEs write to memory
        print("ofm: "+str(len(self.ofm[0]))+"x"+str(len(self.ofm)))
        print("ofm_seq: "+str(len(self.ofm_seq[0]))+"x"+str(len(self.ofm_seq)))
        print("array_h: "+str(self.array_h)+" array_w: "+str(self.array_w))
        for y in range(self.array_h):
            for x in range(self.array_w):
                # Each PE outputs exactly one value to memory
                ofm_w = len(self.ofm[0])
                _y = int(x/ofm_w)
                _x = x % ofm_w
                print("_x: "+str(_x)+" _y: "+str(_y))
                print("ofm_seq[0]["+str((y*self.array_w)+x)+"]")
                new_element = self.ofm[_y][_x]
                self.ofm_seq[0][(y*self.array_w)+x].append(new_element)
        delay = self.hw[c.MUL] + self.hw[c.SUM] + 1 # Delay from PE to PE

        for x in range(self.array_h):
            for y in range(self.array_w):
                print("ofm_seq["+str(x)+"]["+str(y)+"]   "+str(self.ofm_seq[x][y]))

    def __fit_in_memory(self, seq, capacity):
        '''
        Fit the sequences in physical memory
        '''
        for h in range(self.array_h):
            for w in range(self.array_w):
                for q in range(len(seq[h][w])):
                    if type(seq[h][w][q]) != type([]):
                        seq[h][w][q] = [seq[h][w][q]]
                    for i in range(len(seq[h][w][q])):
                        if seq[h][w][q][i] is not s.NAN:
                            seq[h][w][q][i] = seq[h][w][q][i]%capacity


    def size_array(self,matrix):
        '''
        Return the number of elements of an array
        '''
        shape = numpy.shape(matrix)
        print("dimensions: "+str(shape))
        return reduce(lambda x, y: x*y, shape)
        return

    def gen_signals(self):
        '''
        Generate the signals
        Row stationary dataflow
        '''
        self.print_d("--> Generating Signals <--")
        self.print_d("--> (1/8) Generating multicast groups")
        start = timeit.default_timer()
        self.__gen_multicast_groups()
        stop = timeit.default_timer()
        self.print_d("time: "+str(stop-start)+"s")
        self.print_d("--> (2/8) Generating mem_ifm write sequences")
        if initialize_memories:
            self.__init_mem_ifm()
            self.__init_mem_filter()
            self.__gen_mem_ifm_fil_rd2()
        self.print_d("time: "+str(stop-start)+"s")
        stop = timeit.default_timer()
        self.print_d("time: "+str(stop-start)+"s")
        stop = timeit.default_timer()
        self.print_d("time: "+str(stop-start)+"s")
        self.__gen_mem_psum() # FIXME
        stop = timeit.default_timer()
        self.print_d("time: "+str(stop-start)+"s")
        self.print_d("--> (8/8) Generating ofm sequences")
        self.__gen_ofm_seq()
        stop = timeit.default_timer()
        self.print_d("time: "+str(stop-start)+"s")

        # Make all the arrays the same size
        max_size = self.filter_size + self.array_w + self.hw[c.MUL] + self.hw[c.SUM] + 1 + 1# Delay from PE to PE

        stop = timeit.default_timer()
        self.print_d("time: "+str(stop-start)+"s")
        self.print_d("--> Last signals calculations <--")
        signals = {}
        signals[s.MEM_IFM_WR]    = [] #self.mem_ifm_wr
        signals[s.MEM_IFM_RD]    = self.mem_ifm_rd
        signals[s.MEM_FILTER_WR] = [] #self.mem_filter_wr
        signals[s.MEM_FILTER_RD] = [] #self.mem_filter_rd
        signals[s.MEM_PSUM_WR]   = self.mem_psum_wr
        signals[s.MEM_PSUM_RD]   = [] #self.mem_psum_rd
        signals[s.MUX_SEQ]       = [] #self.mux_seq
        signals[s.OUT_PSUM]      = [] #self.out_psum
        signals[s.OFM_SEQ]       = self.ofm_seq


        # Mem init
        signals[s.MEM_IFM_INIT]     = self.mem_ifm_init
        signals[s.MEM_FILTER_INIT]  = self.mem_filter_init

        print("size ofm_seq: "+str(self.size_array(self.ofm_seq)))
        print("size mem_ifm_init: "+str(self.size_array(self.mem_ifm_init)))
        print("size mem_filter_init: "+str(self.size_array(self.mem_filter_init)))
        print("size mem_ifm_rd: "+str(self.size_array(self.mem_ifm_rd)))
        print("size mem_ifm_wr: "+str(self.size_array(self.mem_ifm_wr)))
        print("size mem_filter_rd: "+str(self.size_array(self.mem_filter_rd)))
        print("size mem_filter_wr: "+str(self.size_array(self.mem_filter_wr)))
        print("size mem_psum_wr "+str(self.size_array(self.mem_psum_wr)))
        print("size mem_psum_rd "+str(self.size_array(self.mem_psum_rd)))
        print("size out psum "+str(self.size_array(self.out_psum)))
        print("size mux seq "+str(self.size_array(self.mux_seq)))

        signals[s.PE_TYPE]       = self.pe_type
        signals[s.HW]            = self.hw
        signals[s.ARRAY_W]       = self.array_w
        signals[s.ARRAY_H]       = self.array_h

        signals[s.NUM_CHANNELS]  = self.num_channels
        signals[s.NUM_FILTERS]   = self.num_filters
        signals[s.BATCH]         = self.batch


        # Calculate the string of IFM values
        ifm_w = len(self.mat1[0])
        ifm_h = len(self.mat1)
        ifm_1D = ["" for i in range(ifm_w * ifm_h)]
        bw = self.ofm_size
        idx = 0
        for w in range(0, ifm_w, bw):
            for h in range(ifm_h):
                for b in range(bw):
                    if (w+b) < len(self.mat1[0]):
                        ifm_1D[idx] =self.mat1[h][w+b]
                        idx += 1

        signals[s.IFM] = ifm_1D

        # Calculate the string of FILTER values
        fil_w = len(self.mat2[0])
        fil_h = len(self.mat2)
        fil_1D = ["" for i in range(fil_w * fil_h)]
        idx = 0
        for w in range(0, fil_w, self.hw[c.FIL_BW]):
            for h in range(fil_h):
                for b in range(self.hw[c.FIL_BW]):
                    if (w+b) < fil_w:
                        fil_1D[idx] = self.mat2[h][w+b]
                        idx += 1
        signals[s.FILTER] = fil_1D
        print("size fil_1D "+str(self.size_array(fil_1D)))
        print("size ifm_1D "+str(self.size_array(ifm_1D)))

        # Multicast ifm groups
        multicast_ifm = [[[] for i in range(self.array_w)] for a in range(self.array_h)]
        for x in range(self.array_h):
            for y in range(self.array_w):
                for i in range(len(self.pe_ifm_group)):
                    for a in range(len(self.pe_ifm_group[i])):
                        if self.pe_ifm_group[i][a] == [x,y] :
                            multicast_ifm[x][y].append(i)
        signals[s.MULTICAST_IFM] = multicast_ifm


        if debug_mode:
           for i in range(len(multicast_ifm)):
                print("multicast_ifm: "+str(multicast_ifm[i]))


        # Multicast filter groups
        multicast_fil = [[[] for i in range(self.array_w)] for a in range(self.array_h)]
        for x in range(self.array_h):
            for y in range(self.array_w):
                for i in range(len(self.pe_fil_group)):
                    for a in range(len(self.pe_fil_group[i])):
                        if self.pe_fil_group[i][a] == [x,y] :
                            multicast_fil[x][y].append(i)
        signals[s.MULTICAST_FILTER] = multicast_fil
        print("size multicast_fil "+str(self.size_array(multicast_fil)))


        if debug_mode:
            for i in range(len(multicast_fil)):
                print("multicast_fil: "+str(multicast_fil[i]))

        fill_1Darray(max_size, s.NAN, self.filter_seq_multicast)
        fill_1Darray(max_size, s.NAN, self.ifm_seq_multicast)

        signals[s.FILTER_SEQ_MULTICAST] = self.filter_seq_multicast
        signals[s.IFM_SEQ_MULTICAST]    = self.ifm_seq_multicast

        return signals



