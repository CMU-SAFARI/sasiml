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
import numpy as np
import modules.constants as s
from modules.sanity_check import check_dimension
from modules.sanity_check import check_conv
from modules.transpose import matrix_trans
from modules.transpose import matrix_rot
from modules.transpose import matrix_trans_conv
from modules.transpose import gen_rot_from_trans
from modules.common import next_num
from modules.common import prev_num
from modules.common import gen_pos
from modules.common import del_nan
from modules.common import last_index_nonan
from modules.common import first_index_noval
from modules.common import get_idx_withnan
from modules.common import merge
from modules.common import all_same_size
from modules.common import fill_1Darray
from modules.common import same_size
from modules.common import val_nonan
from modules.common import val_nonan2
import sys
sys.path.insert(0,'..')
import hw.constants as c

initialize_memories = True
# Display some extra information for debuging
debug_mode = False

class gflow_fgrad(object):
    '''
    New dataflow for calculating the filter gradients
    Supports PE expansion
    '''
    def __init__(self, ifm, errors, fil, stride, hw, pe_type, grouping, num_channels, num_filters, batch):
        self.pe_type = pe_type

        self.num_channels = num_channels
        self.num_filters = num_filters
        self.batch = batch

        # Sanity check
        check_dimension(errors)
        check_dimension(fil)
        check_dimension(ifm)
        self.errors = errors
        self.ifm = ifm

        if grouping < 0:
            self.expand = True
        else:
            self.expand = False

        self.grouping = abs(grouping)

        # These are virtual dimensions
        self.array_w = len(fil[0])
        # This is the h dimension after folding
        if not self.expand:
            self.array_h = int(len(fil)/self.grouping) + len(fil)%self.grouping
        else:
            self.array_h = int(len(fil)*self.grouping)

        self.virtual_array_h = len(fil)

        self.fil = fil
        self.stride = stride
        self.hw = hw

        if (self.virtual_array_h%self.grouping) == 0:
            self.group_offset_newlow = 0
        else:
            self.group_offset_newlow = (self.grouping) - (self.virtual_array_h%self.grouping)

        # Create the data structures to save the signals
        self.mem_errors_wr    = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mem_errors_rd    = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mem_ifm_wr = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mem_ifm_rd = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mem_psum_wr   = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mem_psum_rd   = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mux_seq       = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.out_psum      = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.fil_g_seq       = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.multicast_ifm  = [[0 for w in range(self.array_w)] for h in range(self.array_h)]

        # Initial values on IFM and Filter memories
        self.mem_errors_init     = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mem_ifm_init        = [[[] for w in range(self.array_w)] for h in range(self.array_h)]

        self.pe_ifm_group = [[] for i in range(len(self.ifm)*len(self.ifm[0]))]

        if not self.expand:
            self.pe_errors_group = [[]] # only one group, broadcast. We mantain two dimensions for compatibility
        else:
            self.pe_errors_group = [[] for i in range(self.grouping)] # only one group, broadcast. We mantain two dimensions for compatibility


        self.errors_seq_multicast = []
        self.ifm_seq_multicast = []

        self.fil_seq       = [[[] for w in range(self.array_w)] for h in range(self.array_h)]

        self.mem_errors_rd_nobubbles    = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mem_ifm_rd_nobubbles = [[[] for w in range(self.array_w)] for h in range(self.array_h)]

        # This are necesary to calculate when we have to store locally, transmit to other PE, or store to Memory
        self.__pos_errors_wr = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.__pos_ifm_wr = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.__pos_errors_rd = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.__pos_ifm_rd = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.__remote = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.__toMem = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.__iqacc = [[[] for w in range(self.array_w)] for h in range(self.array_h)]

        self.er_w = len(self.errors[0])
        self.er_h = len(self.errors)
        self.fil_w = len(self.fil[0])
        self.fil_h = len(self.fil)
        self.ifm_w = len(self.ifm[0])
        self.ifm_h = len(self.ifm)
        # String of errors
        self.fil_1D = ["" for i in range(self.fil_w * self.fil_h)]
        # String of errors
        ifm_w = len(self.ifm[0])
        ifm_h = len(self.ifm)
        self.ifm_1D = ["" for i in range(ifm_w * ifm_h)]

        self.track = [[0 for w in range(self.array_w)] for h in range(self.array_h)]


    def __pe_grouping(self, y, x):
        '''
        Return pe grouping
        '''
        if self.expand:
            full_group = self.er_w*self.er_h
            size_first_group = int(full_group/self.grouping) + full_group%self.grouping
            # we need to keep track
            if self.track[y][x] < size_first_group:
                _h = y*self.grouping
            else:
                _h = y*self.grouping + int((self.track[y][x]-size_first_group)/(int(full_group/self.grouping)))+1

            res = [_h,x]
            self.track[y][x] += 1
            return res
        else:
            return [int(y/self.grouping),x]


    def __reset_track(self):
        '''
        '''
        for y in range(len(self.track)):
            for x in range(len(self.track[0])):
                self.track[y][x] = 0

    def __gen_multicast_groups(self):
        '''
        Generate the multicast groups
        '''
        self.__reset_track()
        # Broadcast the Errors
        if not self.expand:
            for w in range(self.array_w):
                for h in range(self.array_h):
                    self.pe_errors_group[0].append([h,w])
        else:
            for w in range(self.array_w):
                for h in range(self.array_h):
                    group =  h%self.grouping
                    self.pe_errors_group[group].append([h,w])

        # Pad errors to create the multicast groups

        self.__reset_track()

        # Create IMF multicast groups
        group = 0
        inner_pad = self.stride - 1
        for i_h in range(self.ifm_h):
            for i_w in range(self.ifm_w):
                for y in range(i_h, i_h - (self.er_h + inner_pad*(self.er_h-1)), -1*self.stride):
                        for x in range(i_w, i_w - (self.er_w + inner_pad*(self.er_w-1)),-1*self.stride):
                                if x >= 0 and y>=0 and x < self.fil_w and y < self.fil_h:
                                    _pe = self.__pe_grouping(y,x)
                                    if _pe not in self.pe_ifm_group[group]:
                                        self.pe_ifm_group[group].append(_pe)
                group += 1

        if debug_mode:
            for i in range(len(self.pe_errors_group)):
                print("error_group["+str(i)+"] "+str(self.pe_errors_group[i]))
            for i in range(len(self.pe_ifm_group)):
                print("pe_ifm_group["+str(i)+"]: "+str(self.pe_ifm_group[i]))


    def __cal_pos(self,h,w,cycle):
        pos = 0
        for g in range(cycle):
            if [h,w] in self.pe_ifm_group[g]:
                pos += 1
        return pos


    def __ifm_pos(self, cycle):
        count = 0
        for h in range(len(self.ifm)):
            for w in range(len(self.ifm[0])):
                if count == cycle:
                    return h,w
                count+=1

    def __init_mem_ifm(self):
        '''
        NEW
        Initialize the memory of the PEs
        TODO: check correctness
        '''
        ifm_plain = [self.ifm[h][w] for h in range(len(self.ifm)) for w in range(len(self.ifm[0]))]
        pos = [[0 for x in range(self.array_w)] for y in range(self.array_h)]
        for c in range(len(self.pe_ifm_group)):
            for g in self.pe_ifm_group[c]:
                self.mem_ifm_init[g[0]][g[1]].append(ifm_plain[c])
        if debug_mode:
            for x in range(len(self.mem_ifm_init)):
                for y in range(len(self.mem_ifm_init[0])):
                    print("mem_ifm_init["+str(x)+"]["+str(y)+"]    "+str(self.mem_ifm_init[x][y]))


    def __gen_mem_ifm_wr(self):
        '''
        Generate mem_errors_wr signals
        This function is (almost) the same than __gen_ifm_ifm_wr in conv.py
        '''
        cycle = 0
        list_sent_groups = [[[] for x in range(self.array_w)] for y in range(self.array_h)]
        saved_cycles = 0
        base = 0
        max_id = -1

        red_pe = [[[] for x in range(self.array_w)] for y in range(self.array_h)]
        # Address generator
        ifm_1D_count = 0
        reduced_cycles = 0
        counted_cycle = -1
        sent = False
        for h in range(self.ifm_h):
            base_id = reduced_cycles
            for w in range(self.ifm_w):

                for y in range(self.array_w):
                    for x in range(self.array_h):
                        self.mem_ifm_wr[x][y].append([s.NAN])

                # update the PEs that receive values
                for g in self.pe_ifm_group[cycle]:
                    bw =  self.hw[c.IFM_BW]
                    # Save values in consequtive positions
                    if cycle not in list_sent_groups[g[0]][g[1]]:
                        list_sent_groups[g[0]][g[1]].append(cycle)
                        bw -= 1
                        _pos = self.__cal_pos(g[0], g[1], cycle)
                        self.mem_ifm_wr[g[0]][g[1]][cycle-base_id] = [_pos]
                        max_id = max(max_id, _pos)
                        found = False
                        for i in range(len(self.ifm_seq_multicast)):
                            if self.ifm_seq_multicast[i][0] == cycle:
                                found = True
                        if not found:
                            self.ifm_seq_multicast.append([cycle])
                        self.__pos_ifm_wr[g[0]][g[1]].append([h,w]) # This will be useful to calculate other signals later

                        if self.ifm[h][w] not in self.ifm_1D:
                            self.ifm_1D[ifm_1D_count] = self.ifm[h][w]
                            ifm_1D_count += 1


                        # Check if we can group more
                        if bw > 0:
                            for m in range(len(self.pe_ifm_group)):
                                if m not in list_sent_groups[g[0]][g[1]]:
                                    if self.pe_ifm_group[m] == self.pe_ifm_group[cycle]:
                                        # We can send it in the same cycle!!
                                        _pos = self.__cal_pos(g[0], g[1], m)
                                        self.mem_ifm_wr[g[0]][g[1]][cycle-base_id].append(_pos) #
                                        # String of errors in the correct order
                                        e_h, e_w = self.__ifm_pos(m)
                                        if self.ifm[e_h][e_w] not in self.ifm_1D:
                                            self.ifm_1D[ifm_1D_count] = self.ifm[e_h][e_w]
                                            ifm_1D_count += 1

                                        if m not in self.ifm_seq_multicast[-1]:
                                            self.ifm_seq_multicast[-1].append(m)

                                        list_sent_groups[g[0]][g[1]].append(m)
                                        _h,_w = self.__ifm_pos(m)
                                        self.__pos_ifm_wr[g[0]][g[1]].append([_h,_w]) # This will be useful to calculate other signals later
                                        bw -= 1

                                        if self.ifm[h][w] not in self.ifm_1D:
                                            self.ifm_1D[ifm_1D_count] = self.ifm[h][w]
                                            ifm_1D_count += 1

                                        if counted_cycle == cycle:
                                            reduced_cycles += 1

                                        # Model limited BW
                                        if bw == 0:
                                            break
                        if reduced_cycles > 0:
                            counted_cycle=cycle
                cycle +=1
        deleted = 0
        for i in range(len(self.mem_ifm_wr[0][0])):
            empty = True
            for h in range(self.array_h):
                for w in range(self.array_w):
                    if self.mem_ifm_wr[h][w][i-deleted] != [s.NAN]:
                        empty = False
                        break
                if not empty:
                    break
            if empty:
                for h in range(self.array_h):
                    for w in range(self.array_w):
                        del self.mem_ifm_wr[h][w][i-deleted]
                deleted += 1
        if debug_mode:
            for x in range(len(self.mem_ifm_wr)):
                for y in range(len(self.mem_ifm_wr[0])):
                    print("mem_ifm_wr["+str(x)+"]["+str(y)+"]    "+str(self.mem_ifm_wr[x][y]))

    def __init_mem_errors(self):
        '''
        New
        Initialize the errors memory
        TODO: check correctness
        '''
        pos = [[0 for x in range(self.array_w)] for y in range(self.array_h)]
        for h in range(len(self.errors)):
            for w in range(len(self.errors[0])):
                for g in self.pe_errors_group[0]:
                    self.mem_errors_init[g[0]][g[1]].append(self.errors[h][w]) #
        if debug_mode:
            for x in range(len(self.mem_errors_init)):
                for y in range(len(self.mem_errors_init[0])):
                    print("mem_errors_init["+str(x)+"]["+str(y)+"]    "+str(self.mem_errors_init[x][y]))

    def __gen_mem_errors_wr(self):
        '''
        Generate mem_filter_wr signals
        This is (almost) the same than in conv.py
        '''
        cycle = 0
        pos = [[0 for x in range(self.array_w)] for y in range(self.array_h)]
        bw =  self.hw[c.FIL_BW]
        for h in range(self.er_h):
            for w in range(self.er_w):
                # init this cycle to nan
                for x in range(self.array_h):
                    for y in range(self.array_w):
                        self.mem_errors_wr[x][y].append([s.NAN])
                # update the PEs that receive values
                idx_egroup = 0
                if self.expand:
                    idx_egroup = cycle%self.grouping

                bw -= 1
                for g in self.pe_errors_group[idx_egroup]:
                    if self.mem_errors_wr[g[0]][g[1]][cycle] == [s.NAN]:
                        self.mem_errors_wr[g[0]][g[1]][cycle] = [pos[g[0]][g[1]]] #
                    else:
                        self.mem_errors_wr[g[0]][g[1]][cycle].append(pos[g[0]][g[1]]) #

                    pos[g[0]][g[1]] +=1 # Write the elements of the filter in consequtive positions
                    self.__pos_errors_wr[g[0]][g[1]].append([h,w]) # This will be useful to calculate other signals later
                    self.__pos_errors_rd[g[0]][g[1]].append([h,w]) # This will be useful to calculate other signals later

                if bw == self.hw[c.FIL_BW] - 1:
                    # First element in the cycle
                    self.errors_seq_multicast.append([idx_egroup]) # This is a broadcast
                else:
                    self.errors_seq_multicast[cycle].append(idx_egroup) # This is a broadcast

                if bw == 0:
                    bw = self.hw[c.FIL_BW]
                    cycle +=1
        if debug_mode:
            for x in range(len(self.mem_errors_wr)):
                for y in range(len(self.mem_errors_wr[0])):
                    print("mem_errors_wr["+str(x)+"]["+str(y)+"] "+str(self.mem_errors_wr[x][y]))

    def __index_mem_ifm(self,i):
        '''
        Return the index to read
        When grouping is 1, consequative addresses
        If not, it is tricky
        '''
        if self.grouping == 1:
            return i

        rd_group = 0
        count_i = 0
        count = 0
        for h in range(len(self.fil)):
            for h2 in range(self.er_w*self.er_h):
                if h2 < self.er_w:
                    _temp = count - (self.er_w*self.er_h)*(self.stride-1) - self.er_w + (h2)
                    if _temp > 0:
                        rd_group = _temp
                    else:
                        rd_group = count
                        count += 1

                else:
                    rd_group = count
                    count += 1

                if count_i == i:
                    return rd_group

                count_i += 1

    def __gen_mem_errors_filter_rd2(self):

        '''
        Generate the actual signals from the abstract representations
        '''
        # All PEs can start operating at the same time
        # We do it in this way for simplicity
        # Calculate the minimum starting cycle for all PEs
        # We are very onservative here
        cycle_init = 1
        # Insert bubles at the begining
        for h in range(self.array_h):
            for w in range(self.array_w):
                if not self.expand:
                    for c in range(cycle_init):
                        self.mem_errors_rd[h][w].append(s.NAN)
                        self.mem_ifm_rd[h][w].append(s.NAN)
                else:
                    for c in range(cycle_init+(self.grouping-h%self.grouping)):
                        self.mem_errors_rd[h][w].append(s.NAN)
                        self.mem_ifm_rd[h][w].append(s.NAN)

        if not self.expand:
            size_last_group =  self.virtual_array_h%self.grouping
            if size_last_group == 0:
                size_last_group = self.grouping

            # Insert bubbles after each accumulation
            for h in range(self.array_h):
                if h == self.array_h-1:
                    factor = size_last_group
                else:
                    factor = self.grouping
                for w in range(self.array_w):
                    for i in range(self.er_w*self.er_h*factor):
                        num_mod = self.er_w*self.er_h
                        self.mem_errors_rd[h][w].append(val_nonan2(self.mem_errors_wr[h][w],i%num_mod))
                        #TODO: Calculate the index for the ifm
                        _i = self.__index_mem_ifm(i)
                        self.mem_ifm_rd[h][w].append(_i)
        else:
            # Expand
            full_group = self.er_w*self.er_h
            size_first_group = int(full_group/self.grouping) + full_group%self.grouping #full_group%self.grouping
            if size_first_group == 0:
                size_first_group = int(full_group/self.grouping)

            # Insert bubbles after each accumulation
            for h in range(self.array_h):

                if h%self.grouping == 0:
                    factor = size_first_group
                else:
                    factor = int(full_group/self.grouping)


                for w in range(self.array_w):
                    for i in range(factor):
                        num_mod = factor#self.er_w*self.er_h
                        self.mem_errors_rd[h][w].append(val_nonan2(self.mem_errors_wr[h][w],i%num_mod))
                        _i = self.__index_mem_ifm(i)
                        self.mem_ifm_rd[h][w].append(_i)


        if debug_mode:
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("mem_errors_rd["+str(h)+"]["+str(w)+"]: "+str(self.mem_errors_rd[h][w]))
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("mem_ifm_rd["+str(h)+"]["+str(w)+"]: "+str(self.mem_ifm_rd[h][w]))

    def __gen_mem_errors_filter_rd(self):
        '''
        Generate the actual signals from the abstract representations
        '''
        # All PEs can start operating at the same time
        # We do it in this way for simplicity
        # Calculate the minimum starting cycle for all PEs
        cycle_init = 0
        for h in range(self.array_h):
            for w in range(self.array_w):
                cycle_init = max(cycle_init, last_index_nonan(self.mem_errors_wr[h][w]))
                cycle_init = max(cycle_init, last_index_nonan(self.mem_ifm_wr[h][w]))
        cycle_init += 2

        # Insert bubles at the begining
        for h in range(self.array_h):
            for w in range(self.array_w):
                if not self.expand:
                    for c in range(cycle_init):
                        self.mem_errors_rd[h][w].append(s.NAN)
                        self.mem_ifm_rd[h][w].append(s.NAN)
                else:
                    for c in range(cycle_init+(self.grouping-h%self.grouping)):
                        self.mem_errors_rd[h][w].append(s.NAN)
                        self.mem_ifm_rd[h][w].append(s.NAN)

        if not self.expand:
            size_last_group =  self.virtual_array_h%self.grouping
            if size_last_group == 0:
                size_last_group = self.grouping

            # Insert bubbles after each accumulation
            for h in range(self.array_h):

                if h == self.array_h-1:
                    factor = size_last_group
                else:
                    factor = self.grouping

                for w in range(self.array_w):
                    for i in range(self.er_w*self.er_h*factor):
                        num_mod = self.er_w*self.er_h
                        self.mem_errors_rd[h][w].append(val_nonan2(self.mem_errors_wr[h][w],i%num_mod))
                        _i = self.__index_mem_ifm(i)
                        self.mem_ifm_rd[h][w].append(_i)
        else:
            # Expand
            full_group = self.er_w*self.er_h
            size_first_group = int(full_group/self.grouping) + full_group%self.grouping
            if size_first_group == 0:
                size_first_group = int(full_group/self.grouping)

            # Insert bubbles after each accumulation
            for h in range(self.array_h):

                if h%self.grouping == 0:
                    factor = size_first_group
                else:
                    factor = int(full_group/self.grouping)


                for w in range(self.array_w):
                    for i in range(factor):
                        num_mod = factor
                        self.mem_errors_rd[h][w].append(val_nonan2(self.mem_errors_wr[h][w],i%num_mod))
                        _i = self.__index_mem_ifm(i)
                        self.mem_ifm_rd[h][w].append(_i)


        if debug_mode:
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("mem_errors_rd["+str(h)+"]["+str(w)+"]: "+str(self.mem_errors_rd[h][w]))
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("mem_ifm_rd["+str(h)+"]["+str(w)+"]: "+str(self.mem_ifm_rd[h][w]))

    def __gen_mux_deprecated(self):
        '''
        Generate the mux signals
        '''
        for h in range(self.array_h):
            for w in range(self.array_w):
                for x in range(0, len(self.mem_errors_rd[h][w]), 1):
                    self.mux_seq[h][w].append(0)

        # No mux, because there is no PE communication
        if debug_mode:
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("mux_seq["+str(h)+"]["+str(w)+"]:     "+str(self.mux_seq[h][w]))

        raise

    def __gen_mux(self):
        '''
        Generate the mux signals
        '''
        delay = self.hw[c.MUL] + self.hw[c.SUM]
        max_size = 0
        for h in range(self.array_h):
            for w in range(self.array_w):
                for i in range(len(self.__iqacc[h][w])):
                    # If 1, access the queue
                    idx = get_idx_withnan(self.mem_ifm_rd[h][w],i)
                    idx += delay
                    end = idx
                    beging = len(self.mux_seq[h][w])

                    for x in range(beging, end, 1):
                        self.mux_seq[h][w].append(0)

                    if self.__iqacc[h][w][i] == 1:
                        self.mux_seq[h][w].append(1)
                    max_size = max(max_size, len(self.mux_seq[h][w]))

        cycle_init = 100000000
        for h in range(self.array_h):
            for w in range(self.array_w):
                cycle_init = min(cycle_init, first_index_noval(self.mux_seq[h][w],0))

        #make all same size
        for h in range(self.array_h):
            for w in range(self.array_w):
                for i in range(len(self.mux_seq[h][w]), max_size,1):
                    self.mux_seq[h][w].append(0)


        if debug_mode:
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("mux_seq["+str(h)+"]["+str(w)+"]:     "+str(self.mux_seq[h][w]))
            print("mux_seq cycle_init: "+str(cycle_init))

    def __calc_remote(self, h, w, i, num_sums):
        '''
        '''
        # If this is the last SUM: remote, MEM
        toMem = None
        if num_sums == 1:
            toMem = 1  # No PSUMS, one element
        # Look if there is more data belonging to this PSUM later on the PE
        if toMem is None:
            if i < (num_sums-1):
                toMem = 0
            else:
                toMem = 1

        return toMem

    def __get_all_sums(self, pos_pair):
        '''

        '''
        for i in range(len(self.list_groups)):
            if pos_pair in self.list_groups[i]:
                res = self.list_groups[i].copy()
                return res

    def __get_pos_rd_from_wr(self,h,w,pos):
        '''

        '''
        count_data = 0
        for i in range(len(self.mem_ifm_wr[h][w])):
            cycle = self.mem_ifm_wr[h][w][i]
            if type(cycle) == type([]):
                for l in range(len(cycle)):
                    if cycle[l] == pos:
                        return self.__pos_ifm_wr[h][w][count_data]
                    if cycle[l] is not s.NAN:
                        count_data += 1
            else:
                if cycle == pos:
                    return self.__pos_ifm_wr[h][w][count_data]
                if cycle is not s.NAN:
                    count_data += 1

    def __gen_mem_ifm_rd_abs(self):
        '''
        Generate the pattern addresses
        Without bubbles yet
        '''

        sum_group = self.er_w*self.er_h
        accu = 0

        for h in range(self.array_h):
            h_a  = int((h)/self.grouping)
            if accu == sum_group:
                accu = 0
            for w in range(self.array_w):
                #I get unique values by converting the list to a set
                ifm_per_pe =  int(sum_group/self.grouping)#len(set([res[y][0] for y in range(len(res))]))
                if h%self.grouping == 0:
                    ifm_per_pe = ifm_per_pe + (sum_group%self.grouping)
                for x in range(ifm_per_pe):
                    # For calculating data dependencies later
                    pos_rd_a = accu%ifm_per_pe
                    pos_wr = self.__get_pos_rd_from_wr(h, w, pos_rd_a)
                    self.__pos_ifm_rd[h][w].append(pos_wr)
                    accu += 1
        if debug_mode:
            for x in range(len(self.__pos_ifm_rd)):
                for y in range(len(self.__pos_ifm_rd[0])):
                    print("pos_ifm_rd["+str(x)+"]["+str(y)+"]    "+str(self.__pos_ifm_rd[x][y]))

    def __calc_remote2(self, h, w, i, num_sums):
        '''
        '''
        # If this is the last SUM: remote, MEM
        self.__gen_mem_ifm_rd_abs()
        remote = None
        toMem  = 0
        toMemGroup = -1
        iqacc  = 0
        if len(all_sums) == 1:
            remote = 1  # No PSUMS, one element
            toMem = 1
            toMemGroup = all_sums_copy
        else:
            if pos_pair in all_sums:
                all_sums.remove([self.__pos_errors_rd[h][w][i], self.__pos_ifm_rd[h][w][i]])
            else:
                raise
        # Look if there is more data belonging to this PSUM later on the PE
        if remote is None:
            for _i in range(i+1,len(self.__pos_errors_rd[h][w]),1):
                pair = [self.__pos_errors_rd[h][w][_i], self.__pos_ifm_rd[h][w][_i]]
                if pair in all_sums:
                    remote = 0
                    break
            if remote is None:
                remote = 1

        if remote == 1:
            # Is this going to Mem or to other PEs?
            # if toMem is 0:
            # It goes to memory if all the other elements were already calculated in this PE
            # Or in previous PEs
            init = i

            PEs = []
            for _h in range(h, self.array_h, 1):
                for _i in range(init,-1,-1):
                    if _i > len(self.__pos_ifm_rd[_h][w]) - 1:
                        continue
                    pair = [self.__pos_errors_rd[_h][w][_i], self.__pos_ifm_rd[_h][w][_i]]
                    pair_copy = pair.copy()
                    if pair in all_sums:
                        if _h not in PEs: # Avoid repetitions
                            PEs.append(_h)
                        all_sums.remove(pair)
                    if len(all_sums) == 0:
                        toMem = 1
                        toMemGroup = all_sums_copy
                init = len(self.__pos_errors_rd[_h][w]) - 1
            # Calculate __iqacc
            # Do we have to accumulate the result with the PSUMS from the previous PE?
            for i in range(len(PEs)):
                if PEs[i] > h:
                    iqacc = 1


        return remote, toMem, iqacc, toMemGroup


    def __gen_local_remote(self):
        '''
        0 -> local
        1 -> remote (PES or MEM)
        '''
        if not self.expand:
            size_last_group =  self.virtual_array_h%self.grouping
            if size_last_group == 0:
                size_last_group = self.grouping

            num_sums = self.er_h*self.er_w
            for h in range(self.array_h-1,-1,-1):
                if h == self.array_h-1:
                    factor = size_last_group
                else:
                    factor = self.grouping
                for w in range(self.array_w):
                    for i in range(num_sums*factor):
                        toMem  = self.__calc_remote(h,w,i%num_sums,num_sums)
                        self.__toMem[h][w].append(toMem)
        else:
            # Expanding
            full_group = self.er_w*self.er_h
            size_first_group = int(full_group/self.grouping) + full_group%self.grouping #full_group%self.grouping
            if size_first_group == 0:
                size_first_group = int(full_group/self.grouping)

            num_sums = self.er_h*self.er_w
            for h in range(self.array_h-1,-1,-1):
                if h%self.grouping == 0:
                    factor = size_first_group
                else:
                    factor = int(full_group/self.grouping)
                for w in range(self.array_w):
                    for i in range(factor):
                        if i == factor-1:
                            remote = 1
                        else:
                            remote = 0
                        if h%self.grouping == 0 and remote:
                            toMem = 1
                        else:
                            toMem = 0
                        if h%self.grouping != self.grouping-1 and remote:
                            iqacc = 1
                        else:
                            iqacc = 0

                        self.__remote[h][w].append(remote)
                        self.__toMem[h][w].append(toMem)
                        self.__iqacc[h][w].append(iqacc)

        if debug_mode:
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("toMem["+str(h)+"]["+str(w)+"]:  "+str(self.__toMem[h][w]))
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("toMem["+str(h)+"]["+str(w)+"]:  "+str(self.__toMem[h][w]))
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("iqacc["+str(h)+"]["+str(w)+"]:  "+str(self.__iqacc[h][w]))

    def __gen_out_psum(self):
        '''
        Determines when the values are going out of the PE (for memory or other PEs)
        Data goes OUT in two different cases:
            1) directly (no read from PSUM)
            2) read from PSUM
        '''
        delay = self.hw[c.MUL] + self.hw[c.SUM]-1
        extra = 0
        for h in range(self.array_h):
            for w in range(self.array_w):
                if not self.expand:
                    for i in range(len(self.__toMem[h][w])):
                        if self.__toMem[h][w][i] == 1:
                            idx = get_idx_withnan(self.mem_errors_rd[h][w],i)
                            idx += delay
                            # Fill with nan
                            for f in range(len(self.out_psum[h][w]), idx, +1):
                                self.out_psum[h][w].append(0)
                            self.out_psum[h][w].append(1)
                else:
                    for i in range(len(self.__remote[h][w])):
                        if self.__remote[h][w][i] == 1:
                            idx = get_idx_withnan(self.mem_errors_rd[h][w],i)
                            idx += delay
                            # Fill with nan
                            for f in range(len(self.out_psum[h][w]), idx, +1):
                                self.out_psum[h][w].append(0)
                            # If accumulated with the previous PE, delay one cycle
                            if len(self.mux_seq[h][w]) > idx+1:
                                if self.mux_seq[h][w][idx+1] == 1:
                                    extra += 1
                                    self.out_psum[h][w].append(0) # and extra nan for accumulating with previous PE
                            self.out_psum[h][w].append(1)


        if debug_mode:
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("out_psum["+str(h)+"]["+str(w)+"]:     "+str(self.out_psum[h][w]))

    def __gen_mem_psum(self):
        '''
        Generate mem psum signals
        rd and wr
        Only one PSUM position is used
        '''
        # Prepare some abstract representations to make life easier

        delay = self.hw[c.MUL] + self.hw[c.SUM] - 1
        for h in range(self.array_h):
            for w in range(self.array_w):

                # PSUM WRITE
                self.mem_psum_wr[h][w] = self.mem_ifm_rd[h][w].copy()
                # Add ALU delay
                for d in range(delay):
                    self.mem_psum_wr[h][w].insert(0,s.NAN)

                # substitute for the actual PSUM memory positions
                # In this case is always the first position in the array
                # PSUM WRITE
                idx = 0
                for i in range(len(self.mem_psum_wr[h][w])):
                    if self.mem_psum_wr[h][w][i] is not s.NAN :
                        psum_pos = 0 # We just use one memory position
                        # UPDATE PSUM WRITE
                        if not self.expand:
                            if self.__toMem[h][w][idx] == 0: # Only write if it is not remote
                                self.mem_psum_wr[h][w][i] = psum_pos
                            else:
                                self.mem_psum_wr[h][w][i] = s.NAN
                        else:
                            #expand
                            if self.__remote[h][w][idx] == 0: # Only write if it is not remote
                                self.mem_psum_wr[h][w][i] = psum_pos
                            elif self.__iqacc[h][w][idx] == 1 and self.__remote[h][w][idx] == 0:
                                self.mem_psum_wr[h][w][i] = psum_pos
                            elif self.__remote[h][w][idx] == 1 and ( (i+1) < len(self.mux_seq[h][w]) and self.mux_seq[h][w][i+1] == 1):
                                self.mem_psum_wr[h][w][i] = psum_pos
                            else:
                                self.mem_psum_wr[h][w][i] = s.NAN
                        idx += 1


                # PSUM READ
                if not self.expand:
                    self.mem_psum_rd[h][w] = self.mem_ifm_rd[h][w].copy()
                    for d in range(delay):
                        self.mem_psum_rd[h][w].insert(0,s.NAN)

                    idx = 0
                    first = []
                    next_mux = s.NAN
                    for i in range(len(self.mem_psum_rd[h][w])):
                        if not self.expand:
                            if self.mem_psum_rd[h][w][i] is not s.NAN:
                                if idx%(self.er_w*self.er_h) == 0:
                                    first = []
                                psum_pos = 0
                                # UPDATE MEM_PSUM_READ
                                if psum_pos not in first:
                                    #include
                                    first.append(psum_pos)
                                    # not writing this
                                    self.mem_psum_rd[h][w][i] = s.NAN
                                else:
                                    self.mem_psum_rd[h][w][i] = psum_pos
                                idx += 1
                else:
                    self.mem_psum_rd[h][w] = self.mem_psum_wr[h][w].copy()
                    self.mem_psum_rd[h][w].insert(0,s.NAN)

        if debug_mode:
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("mem_psum_wr["+str(h)+"]["+str(w)+"]:     "+str(self.mem_psum_wr[h][w]))
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("mem_psum_rd["+str(h)+"]["+str(w)+"]:     "+str(self.mem_psum_rd[h][w]))

    def __gen_out_psum(self):
        '''
        Determines when the values are going out of the PE (for memory or other PEs)
        Exactly the same as toMem
        '''
        delay = self.hw[c.MUL] + self.hw[c.SUM]-1
        extra = 0
        for h in range(self.array_h):
            for w in range(self.array_w):
                if not self.expand:
                    for i in range(len(self.__toMem[h][w])):
                        if self.__toMem[h][w][i] == 1:
                            idx = get_idx_withnan(self.mem_errors_rd[h][w],i)
                            idx += delay
                            # Fill with zeros
                            for f in range(len(self.out_psum[h][w]), idx, +1):
                                self.out_psum[h][w].append(0)
                            self.out_psum[h][w].append(1)
                else:
                    for i in range(len(self.__remote[h][w])):
                        if self.__remote[h][w][i] == 1:
                            idx = get_idx_withnan(self.mem_errors_rd[h][w],i)
                            idx += delay
                            # Fill with zeros
                            for f in range(len(self.out_psum[h][w]), idx, +1):
                                self.out_psum[h][w].append(0)
                            # If accumulated with the previous PE, delay one cycle
                            if len(self.mux_seq[h][w]) > idx+1:
                                if self.mux_seq[h][w][idx+1] == 1:
                                    extra += 1
                                    self.out_psum[h][w].append(0) # and extra nan for accumulating with previous PE
                            self.out_psum[h][w].append(1)



        if debug_mode:
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("out_psum["+str(h)+"]["+str(w)+"]:     "+str(self.out_psum[h][w]))

    def __gen_fil_seq(self):
        '''
        Generate the output gradient sequence
        '''
        # Add delay
        delay = 1 # It is delayed one cycle compared to out_psum and mem_psum_wr
        for h in range(self.array_h):
            for w in range(self.array_w):
                for d in range(delay):
                    self.fil_seq[h][w].append("")

        idx_abs = [[0 for w in range(len(self.__toMem[0]))] for h in range(len(self.__toMem))]
        idx_abs_pair = [[0 for w in range(len(self.__toMem[0]))] for h in range(len(self.__toMem))]
        # Calculate
        for h in range(self.array_h):
            for w in range(self.array_w):
                offset = [0 for i in range(len(self.fil))]
                for i in range(len(self.out_psum[h][w])):
                    if not self.expand:
                        if self.out_psum[h][w][i] == 1:
                            while self.__toMem[h][w][idx_abs[h][w]] != 1:
                                idx_abs[h][w] += 1

                            self.fil_seq[h][w].append(self.fil[offset[h]+h*self.grouping][w])
                            offset[h] +=1

                            idx_abs[h][w] += 1
                        else:
                            self.fil_seq[h][w].append("")
                    else:
                        if self.out_psum[h][w][i] == 1:
                            while self.__remote[h][w][idx_abs[h][w]] != 1:
                                idx_abs[h][w] += 1

                            if self.__remote[h][w][idx_abs[h][w]] == 1 and self.__toMem[h][w][idx_abs[h][w]] == 1:
                                while self.__toMem[h][w][idx_abs[h][w]] != 1:
                                    idx_abs[h][w] += 1
                                self.fil_seq[h][w].append(self.fil[offset[int(h/self.grouping)]+int(h/self.grouping)][w])
                                offset[int(h/self.grouping)] +=1
                                # Check if this goes to memory or t
                            else:
                                self.fil_seq[h][w].append("")
                            idx_abs[h][w] += 1

                        else:
                            self.fil_seq[h][w].append("")

        for h in range(self.array_h):
            for w in range(self.array_w):
                self.fil_seq[h][w].append("")

        if debug_mode:
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("fil_seq["+str(h)+"]["+str(w)+"]:     "+str(self.fil_seq[h][w]))

    def gen_signals(self):
        '''
        Generate the signals
        Row stationary dataflow
        '''
        self.__gen_multicast_groups()
        if initialize_memories:
            self.__init_mem_ifm()
            self.__init_mem_errors()
        self.__gen_mem_ifm_wr()
        self.__gen_mem_errors_wr()

        if initialize_memories:
            self.__gen_mem_errors_filter_rd2() # generate errors and filter reads
        else:
            self.__gen_mem_errors_filter_rd() # generate errors and filter reads
        self.__gen_local_remote()
        self.__gen_mux()
        self.__gen_mem_psum()
        self.__gen_out_psum()
        self.__gen_fil_seq()

        # we increase the mux_seq one cycle
        # This was not done before to facilitate the computations of other signals
        for h in range(self.array_h):
            for w in range(self.array_w):
                self.mux_seq[h][w].insert(0,0)

        # Make all the arrays the same size
        if initialize_memories:
            max_size = all_same_size(self.array_h, self.array_w, self.mem_ifm_rd, s.NAN, self.mem_errors_rd, s.NAN, self.mem_psum_wr, s.NAN, self.mem_psum_rd, s.NAN, self.mux_seq, 0, self.out_psum, 0,  self.fil_seq, '')
        else:
            max_size = all_same_size(self.array_h, self.array_w, self.mem_ifm_wr, s.NAN, self.mem_ifm_rd, s.NAN, self.mem_errors_wr, s.NAN, self.mem_errors_rd, s.NAN, self.mem_psum_wr, s.NAN, self.mem_psum_rd, s.NAN, self.mux_seq, 0, self.out_psum, 0,  self.fil_seq, '')

        if initialize_memories:
            self.mem_ifm_wr       = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
            self.mem_errors_wr    = [[[] for w in range(self.array_w)] for h in range(self.array_h)]

        signals = {}
        signals[s.MEM_IFM_WR]    = self.mem_ifm_wr
        signals[s.MEM_IFM_RD]    = self.mem_ifm_rd
        signals[s.MEM_FILTER_WR] = self.mem_errors_wr
        signals[s.MEM_FILTER_RD] = self.mem_errors_rd
        signals[s.MEM_PSUM_WR]   = self.mem_psum_wr
        signals[s.MEM_PSUM_RD]   = self.mem_psum_rd
        signals[s.MUX_SEQ]       = self.mux_seq
        signals[s.OUT_PSUM]      = self.out_psum
        signals[s.OFM_SEQ]       = self.fil_seq

        # Mem init
        signals[s.MEM_IFM_INIT]     = self.mem_ifm_init
        signals[s.MEM_FILTER_INIT]  = self.mem_errors_init

        signals[s.PE_TYPE]       = self.pe_type
        signals[s.HW]            = self.hw
        signals[s.ARRAY_W]       = self.array_w
        signals[s.ARRAY_H]       = self.array_h

        signals[s.NUM_CHANNELS] = self.num_channels
        signals[s.NUM_FILTERS]  = self.num_filters
        signals[s.BATCH]        = self.batch

        if debug_mode:
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("PE["+str(h)+"]["+str(w)+"]")
                    print("    mem_ifm_wr: "+str(self.mem_ifm_wr[h][w]))
                    print("    mem_ifm_rd: "+str(self.mem_ifm_rd[h][w]))
                    print("    mem_errors_wr: "+str(self.mem_errors_wr[h][w]))
                    print("    mem_errors_rd: "+str(self.mem_errors_rd[h][w]))
                    print("    mem_psum_wr:   "+str(self.mem_psum_wr[h][w]))
                    print("    mem_psum_rd:   "+str(self.mem_psum_rd[h][w]))
                    print("    mux_seq:       "+str(self.mux_seq[h][w]))
                    print("    out_psum:      "+str(self.out_psum[h][w]))
                    print("    fil_seq:  "+str(self.fil_seq[h][w]))
        signals[s.IFM] = self.ifm_1D

        # Calculate the string of FILTER values
        errors_1D = ["" for i in range(self.er_w * self.er_h)]
        idx = 0
        for w in range(self.er_w):
            for h in range(self.er_h):
                errors_1D[idx] = self.errors[w][h]
                idx += 1
        signals[s.FILTER] = errors_1D

        # Multicast error groups
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

        # Multicast error groups
        multicast_errors = [[[] for i in range(self.array_w)] for a in range(self.array_h)]
        for x in range(self.array_h):
            for y in range(self.array_w):
                for i in range(len(self.pe_errors_group)):
                    for a in range(len(self.pe_errors_group[i])):
                        if self.pe_errors_group[i][a] == [x,y] :
                            multicast_errors[x][y].append(i)
        signals[s.MULTICAST_FILTER] = multicast_errors

        if debug_mode:
            for i in range(len(multicast_errors)):
                print("multicast_errors: "+str(multicast_errors[i]))
            print("errors_seq_multicast: "+str(self.errors_seq_multicast))
            print("ifm_seq_multicast: "+str(self.ifm_seq_multicast))
            print("ifm_1D: "+str(self.ifm_1D))

        fill_1Darray(max_size, s.NAN, self.errors_seq_multicast)
        fill_1Darray(max_size, s.NAN, self.ifm_seq_multicast)
        if initialize_memories:
            signals[s.FILTER_SEQ_MULTICAST] = [-1 for x in range(max_size)]
            signals[s.IFM_SEQ_MULTICAST]    = [-1 for x in range(max_size)]
        else:
            signals[s.FILTER_SEQ_MULTICAST] = self.errors_seq_multicast
            signals[s.IFM_SEQ_MULTICAST]    = self.ifm_seq_multicast


        return signals
