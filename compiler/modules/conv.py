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
from numpy import nan
import numpy as np
import modules.constants as s

from modules.sanity_check import check_dimension

from modules.common import del_nan
from modules.common import index_nonan
from modules.common import index_nonan2
from modules.common import all_same_size
from modules.common import fill_1Darray

import sys
sys.path.insert(0,'..')
import hw.constants as c

# Initialize the memories or distribute the data?
initialize_memories = True

# Print debugging information
debug_mode = False

class conv(object):
    '''
    Maps a regular convolution to the hardware using a row-stationary dataflow
    '''

    def __init__(self, ifm, fil, ofm, stride, hw, pe_type, num_channels, num_filters, batch, info):
        if pe_type == s.NEWARCH:
            # TODO: New architecture
            raise

        self.info = info

        self.pe_type = pe_type

        # Hardware configuration
        self.hw = hw

        self.num_channels = num_channels
        self.num_filters  = num_filters
        self.batch        = batch

        # Sanity check
        check_dimension(ifm)
        check_dimension(fil)
        check_dimension(ofm)

        # Dimensions of the PE array
        self.array_w = len(ofm[0])
        self.array_h = len(fil)

        # The actual ifmap, filter, ofmap and stride
        self.ifm = ifm
        self.fil = fil
        self.ofm = ofm
        self.stride = stride

        # Create the data structures to save the signals
        self.mem_ifm_wr         = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mem_ifm_rd         = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mem_filter_wr      = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mem_filter_rd      = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mem_psum_wr        = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mem_psum_rd        = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mux_seq            = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.out_psum           = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.ofm_seq            = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.multicast_filter   = [[0 for w in range(self.array_w)] for h in range(self.array_h)]

        # Initial values on IFM and Filter memories
        self.mem_ifm_init           = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mem_filter_init        = [[[] for w in range(self.array_w)] for h in range(self.array_h)]

        # Multicast groups
        self.pe_ifm_group = [[] for i in range(len(self.ifm))]
        self.pe_fil_group = [[] for i in range(len(self.fil))]

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
        Generate the ifmap and filter multicast groups
        '''
        # Create multicast groups
        # Distribute IFM diagonally
        group = 0
        for w in range(self.array_w):
            h_init_group = group
            for h in range(self.array_h):
                self.pe_ifm_group[group].append([h,w])
                group += 1
            group = h_init_group + self.stride

        # Distribute Filters horizontally
        for i in range(len(self.fil)):
            for a in range(self.array_w):
                self.pe_fil_group[i].append([i,a])

        if debug_mode:
            for i in range(len(self.ifm)):
                print("multicast_ifm["+str(i)+"] "+str(self.pe_ifm_group[i]))
            for i in range(len(self.fil)):
                print("fil["+str(i)+"] "+str(self.pe_fil_group[i]))

    def __init_mem_ifm(self):
        '''
        Initialize the memory of the PEs
        '''
        pos = [[0 for x in range(self.array_w)] for y in range(self.array_h)]
        #self.ifm[w+b][h]
        for w in range(len(self.ifm[0])):
            for h in range(len(self.ifm)):
                for g in self.pe_ifm_group[h]:
                    # Save values in consequtive positions
                    self.mem_ifm_init[g[0]][g[1]].append(self.ifm[h][w]) # Value!!! TODO

        if debug_mode:
            for x in range(len(self.mem_ifm_init)):
                for y in range(len(self.mem_ifm_init[0])):
                    print("mem_ifm_init["+str(x)+"]["+str(y)+"]    "+str(self.mem_ifm_init[x][y]))


    def __gen_mem_ifm_wr(self):
        '''
        Generate mem_ifm_wr signals
        Data writen in the internal ifm PE registers from the global buffer
        '''
        cycle = 0
        pos = [[0 for x in range(self.array_w)] for y in range(self.array_h)]
        same_cycle = self.hw[c.IFM_BW]-1 # Simulate BW
        w_bw = 0
        for w in range(len(self.ifm[0])):
            # Simulate bw > 1
            if same_cycle < self.hw[c.IFM_BW]-1:
                same_cycle += 1
            else:
                w_bw = w
                same_cycle = 0

            for h in range(len(self.ifm)):
                # Simulate BW > 1
                cycle = int(w_bw/self.hw[c.IFM_BW])*len(self.ifm)+h
                # init this cycle to nan
                if same_cycle == 0:
                    for y in range(self.array_w):
                        for x in range(self.array_h):
                            self.mem_ifm_wr[x][y].append([s.NAN])
                # update the PEs that receive values
                for g in self.pe_ifm_group[h]:
                    # Save values in consequtive positions
                    if self.mem_ifm_wr[g[0]][g[1]][cycle] == [s.NAN]:
                        self.mem_ifm_wr[g[0]][g[1]][cycle] = [pos[g[0]][g[1]]] #
                    else:
                        self.mem_ifm_wr[g[0]][g[1]][cycle].append(pos[g[0]][g[1]]) #
                    pos[g[0]][g[1]] +=1 # Write the elements of the filter in consequtive positions

                if same_cycle == 0:
                    self.ifm_seq_multicast.append([h])
                else:
                    self.ifm_seq_multicast[cycle].append(h)
        if debug_mode:
            for x in range(len(self.mem_ifm_wr)):
                for y in range(len(self.mem_ifm_wr[0])):
                    print("mem_ifm_wr["+str(x)+"]["+str(y)+"]    "+str(self.mem_ifm_wr[x][y]))

    def __init_mem_filter(self):
        '''
        Initialize the filter memory
        '''
        pos = [[0 for x in range(self.array_w)] for y in range(self.array_h)]
        for w in range(len(self.fil[0])):
            for h in range(len(self.fil)):
                # update the PEs that receive values
                for g in self.pe_fil_group[h]:
                    self.mem_filter_init[g[0]][g[1]].append(self.fil[h][w]) #
                    # Write the elements of the filter in consequtive positions
        if debug_mode:
            for x in range(len(self.mem_filter_init)):
                for y in range(len(self.mem_filter_init[0])):
                    print("mem_filter_init["+str(x)+"]["+str(y)+"]    "+str(self.mem_filter_init[x][y]))

    def __gen_mem_filter_wr(self):
        '''
        Generate mem_filter_wr signals
        Data writen in the internal filter PE registers from the global buffer
        '''
        cycle = 0
        # Keeps the position where the data is written
        pos = [[0 for x in range(self.array_w)] for y in range(self.array_h)]
        same_cycle = self.hw[c.FIL_BW]-1 # Simulate BW
        w_bw = 0
        for w in range(len(self.fil[0])):
            # Simulate bw > 1
            if same_cycle < self.hw[c.FIL_BW]-1:
                same_cycle += 1
            else:
                w_bw = w
                same_cycle = 0

            for h in range(len(self.fil)):
                # Simulate BW > 1
                cycle = int(w_bw/self.hw[c.FIL_BW])*len(self.fil)+h
                # init this cycle to nan
                if same_cycle == 0:
                    for y in range(self.array_w):
                        for x in range(self.array_h):
                            self.mem_filter_wr[x][y].append([s.NAN])
                # update the PEs that receive values
                for g in self.pe_fil_group[h]:
                    if self.mem_filter_wr[g[0]][g[1]][cycle] == [s.NAN]:
                        self.mem_filter_wr[g[0]][g[1]][cycle] = [pos[g[0]][g[1]]] #
                    else:
                        self.mem_filter_wr[g[0]][g[1]][cycle].append(pos[g[0]][g[1]]) #
                    # Write the elements of the filter in consequtive positions
                    pos[g[0]][g[1]] +=1
                if same_cycle == 0:
                    self.filter_seq_multicast.append([h])
                else:
                    self.filter_seq_multicast[cycle].append(h)

        if debug_mode:
            for x in range(len(self.mem_filter_wr)):
                for y in range(len(self.mem_filter_wr[0])):
                    print("mem_filter_wr["+str(x)+"]["+str(y)+"] "+str(self.mem_filter_wr[x][y]))


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
        filter    = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        ifm       = [[[] for w in range(self.array_w)] for h in range(self.array_h)]

        #mem_filter_init

        for x in range(self.array_h):
            for y in range(self.array_w):
                plain_ifm = []
                for g in range(len(self.mem_ifm_init[x][y])):
                    if (self.mem_ifm_init[x][y][g] is not s.NAN) and (self.mem_ifm_init[x][y][g] != [s.NAN]):
                        plain_ifm.append(self.mem_ifm_init[x][y][g])
                ifm[x][y] = plain_ifm.copy()

                plain_filter = []
                for g in range(len(self.mem_filter_init[x][y])):
                        if (self.mem_filter_init[x][y][g] is not s.NAN) and (self.mem_filter_init[x][y][g] != [s.NAN]):
                            plain_filter.append(self.mem_filter_init[x][y][g])
                filter[x][y] = plain_filter.copy()

        # Generate read signals, without bubbles
        for x in range(self.array_h):
            for y in range(self.array_w):
                # For each individual PE
                # Populate the mul first, with no nan
                cycle = 0
                len_filter = len(filter[x][y])
                window = 0
                num_windows = int(((len(self.ifm[0])-len(self.fil[0]))/self.stride) +1)
                for window in range(num_windows):
                    for i in range(len_filter):
                        self.mem_ifm_rd[x][y].append(i+(window*self.stride))
                        self.mem_filter_rd[x][y].append(i)
                        cycle +=1

        # Fine tunning
        for x in range(self.array_h):
            for y in range(self.array_w):
                cycle = 0

        ############################################################
        # Insert bubles, one after each window
        ############################################################
        num_windows = int(((len(self.ifm[0])-len(self.fil[0]))/self.stride) +1)
        for x in range(self.array_h-1,-1,-1): # Start from the bottom
            for y in range(self.array_w):
                num_bubbles = 0
                for w in range(num_windows):
                    end_window = self.__index_window(w,self.mem_ifm_rd[x][y])
                    self.mem_ifm_rd[x][y].insert(end_window+1, s.NAN)
                    self.mem_filter_rd[x][y].insert(end_window+1, s.NAN)


        ############################################################
        # Last row of PEs start first
        #   - They do not have any bubbles in the midle
        # Next rows of PEs start +1 cycles later (to account for the latency of receiving psums from the other PEs)
        #   - They have X bubles after
        ############################################################
        for x in range(self.array_h-2,-1,-1): # Start from the bottom
            for y in range(self.array_w):
                 for i in range(self.array_h -1 - x):
                    self.mem_filter_rd[x][y].insert(0,s.NAN)
                    self.mem_ifm_rd[x][y].insert(0,s.NAN)

        # Makes all the arrays the same size
        self.__adjust_array_same_size(self.mem_ifm_rd)
        self.__adjust_array_same_size(self.mem_filter_rd)

        if debug_mode:
            for w in range(len(self.mem_filter_rd[0])):
                for h in range(len(self.mem_filter_rd)):
                    print("mem_filter_rd["+str(h)+"]["+str(w)+"] "+str(self.mem_filter_rd[h][w]))
            for w in range(len(self.mem_ifm_rd[0])):
                for h in range(len(self.mem_ifm_rd)):
                    print("mem_ifm_rd["+str(h)+"]["+str(w)+"]    "+str(self.mem_ifm_rd[h][w]))

    def __gen_mem_ifm_fil_rd(self):
        '''
        Generate mem_ifm_rd and mem_fil_rd
        '''
        # Get the values in the same order they were writen in memory
        filter    = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        ifm       = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        for x in range(self.array_h):
            for y in range(self.array_w):
                plain_ifm = []
                for g in range(len(self.mem_ifm_wr[x][y])):
                    for b in range(len(self.mem_ifm_wr[x][y][g])):
                        if (self.mem_ifm_wr[x][y][g][b] is not s.NAN) and (self.mem_ifm_wr[x][y][g][b] != [s.NAN]):
                            plain_ifm.append(self.mem_ifm_wr[x][y][g][b])
                ifm[x][y] = plain_ifm.copy()

                plain_filter = []
                for g in range(len(self.mem_filter_wr[x][y])):
                    for b in range(len(self.mem_filter_wr[x][y][g])):
                        if (self.mem_filter_wr[x][y][g][b] is not s.NAN) and (self.mem_filter_wr[x][y][g][b] != [s.NAN]):
                            plain_filter.append(self.mem_filter_wr[x][y][g][b])
                filter[x][y] = plain_filter.copy()

        # Generate read signals, without bubbles
        for x in range(self.array_h):
            for y in range(self.array_w):
                # For each individual PE
                # Populate the mul first, with no nan
                cycle = 0
                len_filter = len(filter[x][y])
                window = 0
                num_windows = int(((len(self.ifm[0])-len(self.fil[0]))/self.stride) +1)
                for window in range(num_windows):
                    for i in range(len_filter):
                        self.mem_ifm_rd[x][y].append(i+(window*self.stride))
                        self.mem_filter_rd[x][y].append(i)
                        cycle +=1

        # Fine tunning
        for x in range(self.array_h):
            for y in range(self.array_w):
                cycle = 0
                # IFM
                el = self.mem_ifm_rd[x][y][-1]
                numIfm = len(self.mem_ifm_rd[x][y])
                found_cycle = -1
                for g in range(len(self.mem_ifm_wr[x][y])):
                    if el in self.mem_ifm_wr[x][y][g]:
                        found_cycle = g
                        break
                if found_cycle == -1:
                    raise

                # TODO: check this
                add_cycles = found_cycle*(self.hw[c.IFM_BW])-numIfm+1+self.hw[c.FIL_BW]
                if add_cycles > 0:
                    for i in range(add_cycles):
                        self.mem_ifm_rd[x][y].insert(0,s.NAN) # Insert nan at the beggining of the list
                        self.mem_filter_rd[x][y].insert(0,s.NAN) # Insert nan at cycle the list

                # Do the same for filters
                el = self.mem_filter_rd[x][y][-1]
                numFilter = len(self.mem_filter_rd[x][y])
                found_cycle = -1
                for g in range(len(self.mem_filter_wr[x][y])):
                    if el in self.mem_filter_wr[x][y][g]:
                        found_cycle = g
                        break
                if found_cycle == -1:
                    raise
                add_cycles = found_cycle*(self.hw[c.FIL_BW])-numFilter+1+self.hw[c.FIL_BW]
                if add_cycles > 0:
                    for i in range(add_cycles):
                        self.mem_ifm_rd[x][y].insert(0,s.NAN) # Insert nan at the beggining of the list
                        self.mem_filter_rd[x][y].insert(0,s.NAN) # Insert nan at cycle the list

                #FILTER
                # Align bot IFM and Filter to happen in the same cycle
                diff = index_nonan2(self.mem_filter_rd[x][y]) - index_nonan2(self.mem_ifm_rd[x][y])
                if diff < 0:
                    for i in range(abs(diff)):
                        self.mem_filter_rd[x][y].insert(0,s.NAN)
                if diff > 0:
                    for i in range(diff):
                        self.mem_ifm_rd[x][y].insert(0,s.NAN)


        ############################################################
        # Last row of PEs start first
        #   - They do not have any bubbles in the midle
        # Next rows of PEs start +1 cycles later (to account for the latency of receiving psums from the other PEs)
        #   - They have X bubles after
        ############################################################
        for x in range(self.array_h-2,-1,-1): # Start from the bottom
            for y in range(self.array_w):
                 idx_nxt  = index_nonan(self.mem_filter_rd[x+1][y])
                 idx_curr = index_nonan(self.mem_filter_rd[x][y])
                 insert_curr = (idx_nxt - idx_curr) + (self.array_h -2 -x)
                 for i in range(insert_curr):
                    self.mem_filter_rd[x][y].insert(0,s.NAN)
                    self.mem_ifm_rd[x][y].insert(0,s.NAN)

        ############################################################
        # Insert bubles, one after each window
        ############################################################
        num_windows = int(((len(self.ifm[0])-len(self.fil[0]))/self.stride) +1)
        for x in range(self.array_h-1,-1,-1): # Start from the bottom
            for y in range(self.array_w):
                num_bubbles = 0
                for w in range(num_windows):
                    end_window = self.__index_window(w,self.mem_ifm_rd[x][y])
                    self.mem_ifm_rd[x][y].insert(end_window+1, s.NAN)
                    self.mem_filter_rd[x][y].insert(end_window+1, s.NAN)



        # Makes all the arrays the same size
        self.__adjust_array_same_size(self.mem_ifm_rd)
        self.__adjust_array_same_size(self.mem_filter_rd)

        if debug_mode:
            for w in range(len(self.mem_filter_rd[0])):
                for h in range(len(self.mem_filter_rd)):
                    print("mem_filter_rd["+str(h)+"]["+str(w)+"] "+str(self.mem_filter_rd[h][w]))
            for w in range(len(self.mem_ifm_rd[0])):
                for h in range(len(self.mem_ifm_rd)):
                    print("mem_ifm_rd["+str(h)+"]["+str(w)+"]    "+str(self.mem_ifm_rd[h][w]))

    def __gen_mem_psum(self):
        '''
        1) Generate mem_psum_wr
        We write in PSUM all the elements of the window
        with the exception of the last row of elements, which do not save the last element of the window
        2) Generate mem_psum_rd: it is just  mem_psum_wr shifted by one
        '''
        window_size = len(self.fil[0])
        delay = self.hw[c.MUL] + self.hw[c.SUM] # Delay with respect to mem_ifm_rd
        num_windows = int(((len(self.ifm[0])-len(self.fil[0]))/self.stride) +1)
        for x in range(self.array_h):
            for y in range(self.array_w):
                pos_mem_psum_wr = 0 # Start by writting in the possition 0
                for w in range(num_windows):
                    # This is the index of the begining of the window
                    idx = self.__index_window(w, self.mem_ifm_rd[x][y], True)
                    # fill with nan until that index
                    idx_mem_psum_wr = idx + delay
                    for i in range(len(self.mem_psum_wr[x][y]), idx_mem_psum_wr-1,1):
                        self.mem_psum_wr[x][y].insert(i,s.NAN)

                    if x == self.array_h -1:
                        # In PEs at the botton (e.g., do not receive PSUMs from another PEs) num of psum writes is equal to window - 1 (the last time is sent to memory or another PE )
                        for r in range(window_size-1):
                            self.mem_psum_wr[x][y].append(pos_mem_psum_wr)
                        self.mem_psum_wr[x][y].append(s.NAN)
                    else:
                        # Other PEs, num of psum writes is equal to window
                        for r in range(window_size):
                            self.mem_psum_wr[x][y].append(pos_mem_psum_wr)
                    # Update the position where to write
                    pos_mem_psum_wr += 1

        # Adjust all the arrays to the same size
        self.__adjust_array_same_size(self.mem_psum_wr)

        # Generate mem_psum_rd
        for x in range(self.array_h):
            for y in range(self.array_w):
                self.mem_psum_rd[x][y] = self.mem_psum_wr[x][y].copy()
                self.mem_psum_rd[x][y].insert(0,s.NAN) # Shift to the right by 1

        if debug_mode:
            for x in range(self.array_h):
                for y in range(self.array_w):
                    print("mem_psum_wr["+str(x)+"]["+str(y)+"]   "+str(self.mem_psum_wr[x][y]))
            for x in range(self.array_h):
                for y in range(self.array_w):
                    print("mem_psum_rd["+str(x)+"]["+str(y)+"]   "+str(self.mem_psum_rd[x][y]))


    def __gen_out_psum(self):
        '''
        Generate the out_psum signals
        '''
        window_size = len(self.fil[0])
        delay = self.hw[c.MUL] + self.hw[c.SUM] # Delay with respect to mem_ifm_rd
        num_windows = int(((len(self.ifm[0])-len(self.fil[0]))/self.stride) +1)
        for x in range(self.array_h): #  the last row
            for y in range(self.array_w):
                if x == self.array_h -1:
                    # out_psum: calculate the end of the window
                    for w in range(num_windows):
                        idx_end = self.__index_window(w, self.mem_ifm_rd[x][y])
                        idx_end += delay
                        for i in range(len(self.out_psum[x][y]), idx_end -1, +1):
                            self.out_psum[x][y].insert(i,0)
                        self.out_psum[x][y].insert(idx_end,1) # insert a 1
                else:
                    #calculate the end of the window
                    for w in range(num_windows):
                        idx_end = self.__index_window(w, self.mem_ifm_rd[x][y])
                        idx_end += delay + 1 # One extra latency to accumulate from the previous PE
                        for i in range(len(self.out_psum[x][y]), idx_end -1, +1):
                            self.out_psum[x][y].insert(i,0)
                        self.out_psum[x][y].insert(idx_end,1) # insert a 1

    def __gen_mux(self):
        '''
        Generate the mux signals
        The mux is one in the bubles of the mem_ifm_rd + latMUL + latSUM
        In the last row, the mux is always zero (it does not receive PSUM from other PEs)
        '''
        window_size = len(self.fil[0])
        delay = self.hw[c.MUL] + self.hw[c.SUM] # Delay with respect to mem_ifm_rd
        num_windows = int(((len(self.ifm[0])-len(self.fil[0]))/self.stride) +1)
        for x in range(self.array_h): #  the last row
            for y in range(self.array_w):
                if x == self.array_h -1:
                    for i in range(len(self.mem_ifm_rd[x][y])):
                        self.mux_seq[x][y].append(0)
                else:
                    #calculate the end of the window
                    for w in range(num_windows):
                        idx_end = self.__index_window(w, self.mem_ifm_rd[x][y])
                        idx_end += delay +1
                        for i in range(len(self.mux_seq[x][y]), idx_end, +1):
                            self.mux_seq[x][y].insert(i,0)
                        self.mux_seq[x][y].insert(idx_end,1) # insert a 1

        if debug_mode:
            for x in range(self.array_h):
                for y in range(self.array_w):
                    print("mux_seq["+str(x)+"]["+str(y)+"]    "+str(self.mux_seq[x][y]))

    def __gen_ofm_seq(self):
        '''
        Generate the sequence of outputs, cycle by cycle
        Only the first row generate outputs
        This is the same sequence than mux
        '''
        pos    = [[] for y in range(self.array_w)]
        count    = [0 for y in range(self.array_w)]
        # Get the positions
        for y in range(self.array_w):
            pos[y] = self.mux_seq[0][y].copy()

        for x in range(1, self.array_h, 1): # start at
            for y in range(self.array_w):
                for i in range(len(pos[y])):
                    self.ofm_seq[x][y].append("")
                # Aditional last cycle to write into GBUFFER
                self.ofm_seq[x][y].append("")

        # Only write to memory the first row
        for y in range(self.array_w):
            for i in range(len(pos[y])):
                if pos[y][i] == 0:
                    self.ofm_seq[0][y].append("")
                else:
                    _ofm = self.ofm[count[y]][y]
                    self.ofm_seq[0][y].append(_ofm)
                    count[y] +=1
            # Aditional last cycle to write into GBUFFER
            self.ofm_seq[0][y].append("")

        if debug_mode:
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
        else:
            self.__gen_mem_ifm_wr()
        stop = timeit.default_timer()
        self.print_d("time: "+str(stop-start)+"s")
        self.print_d("--> (3/8) Generating mem_filter write sequences")
        if initialize_memories:
            self.__init_mem_filter()
        else:
            self.__gen_mem_filter_wr()
        stop = timeit.default_timer()
        self.print_d("time: "+str(stop-start)+"s")
        self.print_d("--> (4/8) Generating mem_filter and mem_ifm read sequences")

        if initialize_memories:
            self.__gen_mem_ifm_fil_rd2() # When the memories are initialized
        else:
            self.__gen_mem_ifm_fil_rd() # Read and filter

        stop = timeit.default_timer()
        self.print_d("time: "+str(stop-start)+"s")
        self.print_d("--> (5/8) Generating mem_psum sequences")
        self.__gen_mem_psum()

        stop = timeit.default_timer()
        self.print_d("time: "+str(stop-start)+"s")
        self.print_d("--> (6/8) Generating mux sequences")
        self.__gen_mux()
        stop = timeit.default_timer()
        self.print_d("time: "+str(stop-start)+"s")
        self.print_d("--> (7/8) Generating out_psum sequences")
        self.__gen_out_psum()
        stop = timeit.default_timer()
        self.print_d("time: "+str(stop-start)+"s")
        self.print_d("--> (8/8) Generating ofm sequences")
        self.__gen_ofm_seq()
        stop = timeit.default_timer()
        self.print_d("time: "+str(stop-start)+"s")

        # Make all the arrays the same size
        if initialize_memories:
            max_size = all_same_size(self.array_h, self.array_w, self.mem_ifm_rd, s.NAN, self.mem_filter_rd, s.NAN, self.mem_psum_wr, s.NAN, self.mem_psum_rd, s.NAN, self.mux_seq, 0, self.out_psum, 0,  self.ofm_seq, '')
        else:
            max_size = all_same_size(self.array_h, self.array_w, self.mem_ifm_wr, s.NAN, self.mem_ifm_rd, s.NAN, self.mem_filter_wr, s.NAN, self.mem_filter_rd, s.NAN, self.mem_psum_wr, s.NAN, self.mem_psum_rd, s.NAN, self.mux_seq, 0, self.out_psum, 0,  self.ofm_seq, '')


        # Adjust the memory reads and writes to actually fit in the available memory

        self.print_d("--> Adjusting RD/WRs to fit in the hardware <--")
        self.__fit_in_memory(self.mem_ifm_wr, self.hw[c.IFM_MEM])
        self.__fit_in_memory(self.mem_ifm_rd, self.hw[c.IFM_MEM])
        self.__fit_in_memory(self.mem_filter_wr, self.hw[c.FILTER_MEM])
        self.__fit_in_memory(self.mem_filter_rd, self.hw[c.FILTER_MEM])

        stop = timeit.default_timer()
        self.print_d("time: "+str(stop-start)+"s")
        self.print_d("--> Last signals calculations <--")
        signals = {}
        signals[s.MEM_IFM_WR]    = self.mem_ifm_wr
        signals[s.MEM_IFM_RD]    = self.mem_ifm_rd
        signals[s.MEM_FILTER_WR] = self.mem_filter_wr
        signals[s.MEM_FILTER_RD] = self.mem_filter_rd
        signals[s.MEM_PSUM_WR]   = self.mem_psum_wr
        signals[s.MEM_PSUM_RD]   = self.mem_psum_rd
        signals[s.MUX_SEQ]       = self.mux_seq
        signals[s.OUT_PSUM]      = self.out_psum
        signals[s.OFM_SEQ]       = self.ofm_seq

        # Mem init
        signals[s.MEM_IFM_INIT]     = self.mem_ifm_init
        signals[s.MEM_FILTER_INIT]  = self.mem_filter_init

        signals[s.PE_TYPE]       = self.pe_type
        signals[s.HW]            = self.hw
        signals[s.ARRAY_W]       = self.array_w
        signals[s.ARRAY_H]       = self.array_h

        signals[s.NUM_CHANNELS] = self.num_channels
        signals[s.NUM_FILTERS]  = self.num_filters
        signals[s.BATCH]        = self.batch

        # Dataflow mapping
        signals[s.DF_M] = self.info[s.DF_M]
        signals[s.DF_N] = self.info[s.DF_N]
        #signals[s.DF_E] = self.info[s.DF_E]
        signals[s.DF_P] = self.info[s.DF_P]
        signals[s.DF_Q] = self.info[s.DF_Q]
        signals[s.DF_R] = self.info[s.DF_R]
        signals[s.DF_T] = self.info[s.DF_T]

        # Calculate the string of IFM values
        ifm_w = len(self.ifm[0])
        ifm_h = len(self.ifm)
        ifm_1D = ["" for i in range(ifm_w * ifm_h)]
        idx = 0
        for w in range(0, ifm_w, self.hw[c.IFM_BW]):
            for h in range(ifm_h):
                for b in range(self.hw[c.IFM_BW]):
                    if (w+b) < len(self.ifm):
                        # TODO: Why is this working? [w+b] and [h] should be swaped (see multiply.py)
                        ifm_1D[idx] =self.ifm[w+b][h]
                        idx += 1

        signals[s.IFM] = ifm_1D
        self.print_d("ifm_1D: "+str(ifm_1D))

        # Calculate the string of FILTER values
        fil_w = len(self.fil[0])
        fil_h = len(self.fil)
        fil_1D = ["" for i in range(fil_w * fil_h)]
        idx = 0
        for w in range(0, fil_w, self.hw[c.FIL_BW]):
            for h in range(fil_h):
                for b in range(self.hw[c.FIL_BW]):
                    if (w+b) < len(self.fil):
                        fil_1D[idx] = self.fil[w+b][h]
                        idx += 1
        signals[s.FILTER] = fil_1D

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


        if debug_mode:
            for i in range(len(multicast_fil)):
                print("multicast_fil: "+str(multicast_fil[i]))

        fill_1Darray(max_size, s.NAN, self.filter_seq_multicast)
        fill_1Darray(max_size, s.NAN, self.ifm_seq_multicast)

        signals[s.FILTER_SEQ_MULTICAST] = self.filter_seq_multicast
        signals[s.IFM_SEQ_MULTICAST]    = self.ifm_seq_multicast

        return signals



