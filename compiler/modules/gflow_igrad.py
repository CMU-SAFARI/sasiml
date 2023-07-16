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

import sys
sys.path.insert(0,'..')
import hw.constants as c

initialize_memories = True

# To print the name of the variables instead of the actual signals
debug_mode = False

class gflow_igrad(object):
    '''
    New Dataflow for the backward pass
    Supports PE grouping
    '''
    def __init__(self, errors, fil, gradients, stride, hw, pe_type, grouping, num_channels, num_filters, batch):
        # TODO: implement with the new architecture
        if pe_type == s.NEWARCH:
            raise
        self.pe_type = pe_type

        self.num_channels = num_channels
        self.num_filters = num_filters
        self.batch = batch

        # TRansposed filter
        self.fil_t = matrix_trans(fil)
        # Rotated filter
        self.fil_r = matrix_rot(fil)

        # Sanity check
        check_dimension(errors)
        check_dimension(self.fil_t)
        check_dimension(gradients)
        self.errors = errors


        # These are virtual dimensions
        self.array_w = len(errors[0])
        # This is the h dimension after folding
        self.grouping = grouping
        self.array_h = int(len(errors)/grouping) + len(errors)%grouping
        self.virtual_array_h = len(errors)


        if debug_mode:
            for h in range(len(self.fil_t)):
                print("_fil_t:  "+str(self.fil_t[h]))
            for h in range(len(self.errors)):
                print("_errors: "+str(self.errors[h]))

        self.gradients = gradients
        self.stride = stride
        self.hw = hw


        if (self.virtual_array_h%self.grouping) == 0:
            self.group_offset_newlow = 0
        else:
            self.group_offset_newlow = (self.grouping) - (self.virtual_array_h%self.grouping)

        # Create the data structures to save the signals
        self.mem_errors_wr    = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mem_errors_rd    = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mem_filter_wr = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mem_filter_rd = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mem_psum_wr   = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mem_psum_rd   = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mux_seq       = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.out_psum      = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.gradient_seq       = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.multicast_filter  = [[0 for w in range(self.array_w)] for h in range(self.array_h)]

        # Initial values on IFM and Filter memories
        self.mem_errors_init           = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mem_filter_init        = [[[] for w in range(self.array_w)] for h in range(self.array_h)]

        self.virtual_pe_errors_group = [[] for i in range(self.virtual_array_h*self.array_w)]
        self.pe_errors_group = [[] for i in range(self.virtual_array_h*self.array_w)]

        self.pe_fil_group = [[]] # only one group, broadcast. We mantain two dimensions for compatibility

        self.errors_seq_multicast = []
        self.filter_seq_multicast = []

        self.mem_errors_rd_nobubbles    = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.mem_filter_rd_nobubbles = [[[] for w in range(self.array_w)] for h in range(self.array_h)]

        # This are necesary to calculate when we have to store locally, transmit to other PE, or store to Memory
        self.__pos_errors_wr = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.__pos_filter_wr = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.__pos_errors_rd = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.__pos_filter_rd = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.__num_sums = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.__remote = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.__toMem = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.__toMemGroup = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.__iqacc = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        self.list_groups = []
        self.list_groups_out = []


        # String of errors
        errors_w = len(self.errors[0])
        errors_h = len(self.errors)
        self.errors_1D = ["" for i in range(errors_w * errors_h)]


    def __shift(self, seq, n):
        '''
        Shift a vector to the left
        '''
        n = n % len(seq)
        n_right = len(seq) - n
        return seq[n_right:]+seq[:n_right]

    def __shift_calc(self):
        '''
        Key in our algorithm
        '''
        f_h = len(self.fil_t)
        f_w = len(self.fil_t[0])
        weights = f_h*f_w

        rd_patterns = [[] for w in range(weights)]
        # Initial values
        for w in range(weights):
            for o in range(len(self.errors[0])):
                rd_patterns[w].append(o)

        # Shift
        for h in range(f_h):
            for w in range(f_w):
                weight_n = h*f_h + w
                n = int(weight_n/(f_h*self.stride))
                rd_patterns[weight_n] = self.__shift(rd_patterns[weight_n], n)
        return rd_patterns

    def __gen_multicast_groups(self):
        '''
        Generate the multicast groups
        '''
        # Broadcast the Filters
        for w in range(self.array_w):
            for h in range(self.array_h):
                self.pe_fil_group[0].append([h,w])

        # Create multicast groups
        errors_per_pe = min(len(self.fil_t[0])-(self.stride-1), len(self.errors[0]))

        weights = len(self.fil_t)*len(self.fil_t[0])
        group = 0

        res = self.__shift_calc()

        for h in range(self.virtual_array_h):
            for w in range(self.array_w):
                for a in range(len(res[0])):
                    for i in range(len(res)):
                        if group%self.array_w == res[i][a]:
                            if [h,a] not in self.virtual_pe_errors_group[group]:
                                self.virtual_pe_errors_group[group].append([h,a])
                group += 1
        # calculate
        # Do the ERROR grouping
        # The groups are the same, we just reduce the h dimension
        for g in range(len(self.virtual_pe_errors_group)):
            for i in range(len(self.virtual_pe_errors_group[g])):
                # group vertically
                h = self.virtual_pe_errors_group[g][i][0]
                w = self.virtual_pe_errors_group[g][i][1]
                h = int((h)/self.grouping)
                self.pe_errors_group[g].append([h,w])

        if debug_mode:
            for i in range(len(self.virtual_pe_errors_group)):
                print("virtual_error_group["+str(i)+"] "+str(self.virtual_pe_errors_group[i]))
            for i in range(len(self.pe_errors_group)):
                print("error_group["+str(i)+"] "+str(self.pe_errors_group[i]))
            for i in range(len(self.pe_fil_group)):
                print("pe_fil_group["+str(i)+"]: "+str(self.pe_fil_group[i]))


    def __cal_pos(self,h,w,cycle):
        pos = 0
        for g in range(cycle):
            if [h,w] in self.pe_errors_group[g]:
                pos += 1
        return pos

    def __error_pos(self, cycle):
        count = 0
        for h in range(len(self.errors)):
            for w in range(len(self.errors[0])):
                if count == cycle:
                    return h,w
                count+=1

    def __id_to_coor(self, idx, array):
        '''
        From plain idx to coordinate
        '''
        count = 0
        for h in range(len(array)):
            for w in range(len(array[0])):
                if count == idx:
                    return h,w
                count += 1
        raise

    def __init_mem_errors(self):
        '''
        NEW
        Initialize the memory of the PEs
        '''
        errors_plain = [self.errors[h][w] for h in range(len(self.errors)) for w in range(len(self.errors[0]))]

        pos = [[0 for x in range(self.array_w)] for y in range(self.array_h)]
        for c in range(len(self.pe_errors_group)):
            for g in self.pe_errors_group[c]:
                self.mem_errors_init[g[0]][g[1]].append(errors_plain[c]) #

        if debug_mode:
            for x in range(len(self.mem_errors_init)):
                for y in range(len(self.mem_errors_init[0])):
                    print("mem_errors_init["+str(x)+"]["+str(y)+"]    "+str(self.mem_errors_init[x][y]))

    def __gen_mem_errors_wr(self):
        '''
        Generate mem_errors_wr signals
        This function is (almost) the same than __gen_ifm_ifm_wr in conv.py
        '''
        cycle = 0
        bw =  self.hw[c.IFM_BW]
        list_sent_groups = [[[] for x in range(self.array_w)] for y in range(self.array_h)]
        saved_cycles = 0
        base = 0
        max_id = -1


        # Address generator
        errors_1D_count = 0
        reduced_cycles = 0
        counted_cycle = -1

        for h in range(len(self.errors)):
            base_id = reduced_cycles#max_id+1
            for w in range(len(self.errors[0])):
                # init this cycle to nan
                for y in range(self.array_w):
                    for x in range(self.array_h):
                        self.mem_errors_wr[x][y].append([s.NAN])

                # update the PEs that receive values
                for g in self.pe_errors_group[cycle]:
                    bw =  self.hw[c.IFM_BW]
                    if cycle not in list_sent_groups[g[0]][g[1]]:
                        # Keep track of all the cycles we have send
                        list_sent_groups[g[0]][g[1]].append(cycle)
                        bw -= 1
                        _pos = self.__cal_pos(g[0], g[1], cycle)
                        self.mem_errors_wr[g[0]][g[1]][cycle-base_id] = [_pos]
                        max_id = max(max_id, _pos)

                        found = False
                        for i in range(len(self.errors_seq_multicast)):
                            if self.errors_seq_multicast[i][0] == cycle:
                                found = True

                        if not found:
                            self.errors_seq_multicast.append([cycle])

                        self.__pos_errors_wr[g[0]][g[1]].append([h,w]) # This will be useful to calculate other signals later

                        if self.errors[h][w] not in self.errors_1D:
                            self.errors_1D[errors_1D_count] = self.errors[h][w]
                            errors_1D_count += 1

                        # Check if we can group more
                        if bw > 0:
                            for m in range(len(self.pe_errors_group)):
                                if m not in list_sent_groups[g[0]][g[1]]:
                                    if self.pe_errors_group[m] == self.pe_errors_group[cycle]:
                                        # We can send it in the same cycle!!
                                        _pos = self.__cal_pos(g[0], g[1], m)
                                        self.mem_errors_wr[g[0]][g[1]][cycle-base_id].append(_pos) #
                                        # String of errors in the correct order
                                        e_h, e_w = self.__error_pos(m)
                                        if self.errors[e_h][e_w] not in self.errors_1D:
                                            self.errors_1D[errors_1D_count] = self.errors[e_h][e_w]
                                            errors_1D_count += 1

                                        if m not in self.errors_seq_multicast[-1]:
                                            self.errors_seq_multicast[-1].append(m)

                                        list_sent_groups[g[0]][g[1]].append(m)
                                        _h,_w = self.__error_pos(m)
                                        self.__pos_errors_wr[g[0]][g[1]].append([_h,_w])
                                        bw -= 1

                                        if self.errors[h][w] not in self.errors_1D:
                                            self.errors_1D[errors_1D_count] = self.errors[h][w]
                                            errors_1D_count += 1

                                        if counted_cycle != cycle:
                                            reduced_cycles += 1

                                        # Model limited BW
                                        if bw == 0:
                                            break

                        if reduced_cycles > 0:
                            counted_cycle=cycle
                cycle +=1

        if debug_mode:
            for x in range(len(self.mem_errors_wr)):
                for y in range(len(self.mem_errors_wr[0])):
                    print("mem_errors_wr["+str(x)+"]["+str(y)+"]    "+str(self.mem_errors_wr[x][y]))

            for x in range(len(self.__pos_errors_wr)):
                for y in range(len(self.__pos_errors_wr[0])):
                    print("pos_errors_wr["+str(x)+"]["+str(y)+"]    "+str(self.__pos_errors_wr[x][y]))

            for i in range(len(self.errors_seq_multicast)):
                print("errors_seq_multicast["+str(i)+"]    "+str(self.errors_seq_multicast[i]))

            for i in range(len(self.errors_1D)):
                print("errors_1D["+str(i)+"]    "+str(self.errors_1D[i]))

    def __init_mem_filter(self):
        '''
        New
        Initialize the filter memory
        '''
        pos = [[0 for x in range(self.array_w)] for y in range(self.array_h)]
        # update the PEs that receive values
        for h in range(len(self.fil_t)):
            for w in range(len(self.fil_t[0])):
                for g in self.pe_fil_group[0]:
                    self.mem_filter_init[g[0]][g[1]].append(self.fil_t[h][w]) #
                    # Write the elements of the filter in consequtive positions
        if debug_mode:
            for x in range(len(self.mem_filter_init)):
                for y in range(len(self.mem_filter_init[0])):
                    print("mem_filter_init["+str(x)+"]["+str(y)+"]    "+str(self.mem_filter_init[x][y]))


    def __gen_mem_filter_wr(self):
        '''
        Generate mem_filter_wr signals
        This is (almost) the same than in conv.py
        '''
        cycle = 0
        pos = [[0 for x in range(self.array_w)] for y in range(self.array_h)]
        bw =  self.hw[c.FIL_BW]
        for h in range(len(self.fil_t)):
            for w in range(len(self.fil_t[0])):
                # init this cycle to nan
                for x in range(self.array_h):
                    for y in range(self.array_w):
                        self.mem_filter_wr[x][y].append([s.NAN])
                # update the PEs that receive values
                bw -= 1
                for g in self.pe_fil_group[0]:
                    # Save values in consequtive positions
                    if self.mem_filter_wr[g[0]][g[1]][cycle] == [s.NAN]:
                        self.mem_filter_wr[g[0]][g[1]][cycle] = [pos[g[0]][g[1]]] #
                    else:
                        self.mem_filter_wr[g[0]][g[1]][cycle].append(pos[g[0]][g[1]]) #


                    pos[g[0]][g[1]] +=1 # Write the elements of the filter in consequtive positions
                    self.__pos_filter_wr[g[0]][g[1]].append([h,w]) # This will be useful to calculate other signals later

                if bw == self.hw[c.FIL_BW] - 1:
                    # First element in the cycle
                    self.filter_seq_multicast.append([0]) # This is a broadcast
                else:
                    self.filter_seq_multicast[cycle].append(0) # This is a broadcast

                if bw == 0:
                    bw = self.hw[c.FIL_BW]
                    cycle +=1

        if debug_mode:
            for x in range(len(self.mem_filter_wr)):
                for y in range(len(self.mem_filter_wr[0])):
                    print("mem_filter_wr["+str(x)+"]["+str(y)+"] "+str(self.mem_filter_wr[x][y]))

            for x in range(len(self.__pos_filter_wr)):
                for y in range(len(self.__pos_filter_wr[0])):
                    print("pos_filter_wr["+str(x)+"]["+str(y)+"]    "+str(self.__pos_filter_wr[x][y]))

            print("filter_seq_multicast: "+str(self.filter_seq_multicast))


    def __res_pos(self,res):
        '''
        Calculate positions in memory
        '''
        result = [[0 for w in range(len(res[0]))] for h in range(len(res))]
        for w in range(len(res[0])):
            for h in range(len(res)):
                ss = sorted(list(set([res[y][w] for y in range(len(res))])))
                for i in range(len(ss)):
                    if res[h][w] == ss[i]:
                        result[h][w] = i
                        break
        return result

    def __get_pos_rd_from_wr(self,h,w,pos):
        '''

        '''
        count_data = 0
        for i in range(len(self.mem_errors_wr[h][w])):
            cycle = self.mem_errors_wr[h][w][i]
            if type(cycle) == type([]):
                for l in range(len(cycle)):
                    if cycle[l] == pos:
                        return self.__pos_errors_wr[h][w][count_data]
                    if cycle[l] is not s.NAN:
                        count_data += 1
            else:
                if cycle == pos:
                    return self.__pos_errors_wr[h][w][count_data]
                if cycle is not s.NAN:
                    count_data += 1

    def __gen_mem_errors_rd_abs(self):
        '''
        Generate the pattern addresses
        Without bubbles yet
        '''
        f_h = len(self.fil_t)
        f_w = len(self.fil_t[0])
        # N1CHANGES
        res = self.__shift_calc()

        res_pos = self.__res_pos(res)


        for h in range(self.virtual_array_h):
            for w in range(self.array_w):
                h_a  = int((h)/self.grouping)
                #I get unique values by converting the list to a set
                errors_per_pe =  len(set([res[y][0] for y in range(len(res))]))
                for x in range(len(res)):
                    pos_rd = res_pos[x][w]
                    if h != 0:
                        pos_rd_a = (pos_rd + errors_per_pe*(((h)%self.grouping)))#%(len_errors)
                    else:
                        pos_rd_a = pos_rd
                    self.mem_errors_rd_nobubbles[h_a][w].append(pos_rd_a)
                    # For calculating data dependencies later
                    pos_wr = self.__get_pos_rd_from_wr(h_a, w, pos_rd_a)
                    self.__pos_errors_rd[h_a][w].append(pos_wr)

        if debug_mode:
            for w in range(self.array_w):
                for h in range(self.array_h):
                    print("mem_errors_rd_nobubbles["+str(h)+"]["+str(w)+"] "+str(self.mem_errors_rd_nobubbles[h][w]))
            for x in range(len(self.__pos_errors_rd)):
                for y in range(len(self.__pos_errors_rd[0])):
                    print("pos_errors_rd["+str(x)+"]["+str(y)+"]    "+str(self.__pos_errors_rd[x][y]))
            #assert(0)

    def __gen_mem_filter_rd_abs(self):
        '''
        Generate the pattern addresses
        '''
        pos = 0
        num_weights = len(self.fil_t[0])*len(self.fil_t)
        for h in range(self.virtual_array_h):
            for w in range(self.array_w):
                h_a  = int((h)/self.grouping)
                for a in range(num_weights):
                    self.mem_filter_rd_nobubbles[h_a][w].append(a)
                    # For calculating data dependencies later
                    self.__pos_filter_rd[h_a][w].append(self.__pos_filter_wr[h_a][w][a])

        if debug_mode:
            for w in range(self.array_w):
                for h in range(self.array_h):
                    print("mem_filter_rd_nobubbles["+str(h)+"]["+str(w)+"] "+str(self.mem_filter_rd_nobubbles[h][w]))

            for x in range(len(self.__pos_filter_rd)):
                for y in range(len(self.__pos_filter_rd[0])):
                    print("pos_filter_rd_abs["+str(x)+"]["+str(y)+"]    "+str(self.__pos_filter_rd[x][y]))

    def __flat(self, matrix):
        '''
        Flat a matrix into a vector
        '''
        idx = 0
        vector = [None for i in range(len(matrix)*len(matrix[0]))]
        for h in range(len(matrix)):
            for w in range(len(matrix[0])):
                vector[idx] = matrix[h][w]
                idx += 1
        return vector

    def __mmul(self, vfil, verrors):
        '''
        Perform a matrix multiplication
        Store both fil index and error index
        '''
        matrix = [[None for w in range(len(verrors))] for h in range(len(vfil))]
        for ifil in range(len(vfil)):
            for ierr in range(len(verrors)):
                matrix[ifil][ierr] = [vfil[ifil],verrors[ierr]]
        return matrix

    def __conv_count(self, tconv_errors, fil_t, stride):
        '''
        Count how many accumulations needs each
        '''
        # The stride is always 1!!! The stride was already taken into account before to build the padded error matrix for the transposed convolution
        used_stride = 1
        err_w = len(tconv_errors[0])
        err_h = len(tconv_errors)
        fil_w = len(fil_t[0])
        fil_h = len(fil_t)
        for w_h in range(0, err_h - fil_h +used_stride , used_stride):
            for w_w in range(0, err_w - fil_w + used_stride , used_stride):
                pair_list = []
                ele_count = 0
                for f_h in range(fil_h):
                    for f_w in range(fil_w):
                        f_element = fil_t[f_h][f_w]
                        e_element = tconv_errors[w_h+f_h][w_w+f_w]
                        if f_element is not s.NAN and e_element is not s.NAN:
                            pair_list.append([f_element,e_element])
                            ele_count += 1
                self.list_groups.append(pair_list)
                # Address to write to memory
                self.list_groups_out.append(self.gradients[int(w_h/used_stride)][int(w_w/used_stride)])

        if debug_mode:
            for i in range(len(self.list_groups)):
                print("list_groups_out: "+str(self.list_groups_out[i]))
                print("list_groups: "+str(self.list_groups[i]))



    def __get_num_sums(self, pos_filter, pos_error):
        '''
        '''
        count = 0
        for i in range(len(self.list_groups)):
            if [pos_filter, pos_error] in self.list_groups[i]:
                count = len(self.list_groups[i])
                break
        return count

    def __gen_num_sums(self):
        '''
        For each element, calculate to how many othe elements has to me add to generate the output value
        '''
        for h in range(self.array_h):
            for w in range(self.array_w):
                for e in range(len(self.__pos_errors_rd[h][w])):
                    self.__num_sums[h][w].append(self.__get_num_sums(self.__pos_filter_rd[h][w][e], self.__pos_errors_rd[h][w][e]))

        if debug_mode:
            for h in range(len(self.__num_sums)):
                for w in range(len(self.__num_sums[0])):
                    print("num_sums["+str(h)+"]["+str(w)+"]: "+str(self.__num_sums[h][w]))

    def __get_all_sums(self, pos_pair):
        '''

        '''
        for i in range(len(self.list_groups)):
            if pos_pair in self.list_groups[i]:
                res = self.list_groups[i].copy()
                return res

    def __get_group(self, pair):
        '''
        Return the group to which this pair belongs
        '''
        for i in range(len(self.list_groups)):
            if pair in self.list_groups[i]:
                return self.list_groups[i]

        for i in range(len(self.list_groups)):
            print("list_groups: "+str(self.list_groups[i]))
        raise

    def __calc_remote(self, h, w, i, num_sums):
        '''
        '''
        # If this is the last SUM: remote, MEM
        remote = None
        toMem  = 0
        toMemGroup = -1
        iqacc  = 0
        pos_pair = [self.__pos_filter_rd[h][w][i], self.__pos_errors_rd[h][w][i]]
        all_sums = self.__get_all_sums(pos_pair)
        all_sums_copy = all_sums.copy()
        if len(all_sums) == 1:
            remote = 1  # No PSUMS, one element
            toMem = 1
            toMemGroup = all_sums_copy
        else:
            if pos_pair in all_sums:
                all_sums.remove([self.__pos_filter_rd[h][w][i], self.__pos_errors_rd[h][w][i]])
            else:
                raise
        # Look if there is more data belonging to this PSUM later on the PE
        if remote is None:
            for _i in range(i+1,len(self.__pos_filter_rd[h][w]),1):
                pair = [self.__pos_filter_rd[h][w][_i], self.__pos_errors_rd[h][w][_i]]
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
                    if _i > len(self.__pos_errors_rd[_h][w]) - 1:
                        continue
                    pair = [self.__pos_filter_rd[_h][w][_i], self.__pos_errors_rd[_h][w][_i]]
                    pair_copy = pair.copy()
                    if pair in all_sums:
                        if _h not in PEs: # Avoid repetitions
                            PEs.append(_h)
                        all_sums.remove(pair)
                    if len(all_sums) == 0:
                        toMem = 1
                        toMemGroup = all_sums_copy
                init = len(self.__pos_filter_rd[_h][w]) - 1
            for i in range(len(PEs)):
                if PEs[i] > h:
                    iqacc = 1


        return remote, toMem, iqacc, toMemGroup

    def __gen_local_remote(self):
        '''
        0 -> local
        1 -> remote (PES or MEM)
        '''
        for h in range(self.array_h-1,-1,-1):
            for w in range(self.array_w):
                for i in range(len(self.__num_sums[h][w])):
                    remote, toMem, iqacc, toMemGroup = self.__calc_remote(h,w,i,self.__num_sums[h][w][i])
                    self.__remote[h][w].append(remote)
                    self.__toMem[h][w].append(toMem)
                    self.__toMemGroup[h][w].append(toMemGroup)
                    self.__iqacc[h][w].append(iqacc)
        if debug_mode:
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("remote["+str(h)+"]["+str(w)+"]: "+str(self.__remote[h][w]))
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("toMem["+str(h)+"]["+str(w)+"]:  "+str(self.__toMem[h][w]))
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("toMemGroup["+str(h)+"]["+str(w)+"]:  "+str(self.__toMemGroup[h][w]))
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("iqacc["+str(h)+"]["+str(w)+"]:  "+str(self.__iqacc[h][w]))

    def __gen_data_movements_abs(self):
        '''
        Determine when the results is stored localy
        '''
        # We need to determine which values need to sum together and which dont
        # We now make the transposed convolution of errors and filter
        # with the goal of knowing which elements should be sum together in the matrix

        # This is the matrix padded, prepared for doing the transposed convolution
        pos_errors   = gen_pos(self.errors)
        pos_fil_t    = gen_pos(self.fil_t)
        pos_fil_r    = gen_rot_from_trans(pos_fil_t)
        tconv_errors = matrix_trans_conv(self.errors, self.stride, self.fil_r, self.gradients, s.NAN)
        pos_tconv_errors = matrix_trans_conv(pos_errors, self.stride, pos_fil_r, self.gradients, s.NAN)

        # Make the actual convolution and count how many sums has each individual element
        self.__conv_count(pos_tconv_errors, pos_fil_r, self.stride)

        # Calculate the number of sums
        self.__gen_num_sums()
        self.__gen_local_remote()


    def __gen_mem_errors_filter_rd(self):
        '''
        Generate the actual signals from the abstract representations
        '''
        # All PEs can start operating at the same time
        # We do it in this way for simplicity
        # Calculate the minimum starting cycle for all PEs
        # We are very onservative here
        cycle_init = 0
        for h in range(self.array_h):
            for w in range(self.array_w):
                cycle_init = max(cycle_init, last_index_nonan(self.mem_errors_wr[h][w]))
                cycle_init = max(cycle_init, last_index_nonan(self.mem_filter_wr[h][w]))
        cycle_init += 2

        # Insert bubles at the begining
        for h in range(self.array_h):
            for w in range(self.array_w):
                for c in range(cycle_init):
                    self.mem_errors_rd[h][w].append(s.NAN)
                    self.mem_filter_rd[h][w].append(s.NAN)

        # Insert bubbles after each accumulation
        for h in range(self.array_h):
            for w in range(self.array_w):
                for i in range(len(self.__iqacc[h][w])):
                    self.mem_errors_rd[h][w].append(self.mem_errors_rd_nobubbles[h][w][i])
                    self.mem_filter_rd[h][w].append(self.mem_filter_rd_nobubbles[h][w][i])
                    if self.__iqacc[h][w][i] == 1:
                        self.mem_errors_rd[h][w].append(s.NAN)
                        self.mem_filter_rd[h][w].append(s.NAN)
        if debug_mode:
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("mem_errors_rd: "+str(self.mem_errors_rd[h][w]))
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("mem_filter_rd: "+str(self.mem_filter_rd[h][w]))

    def __gen_mem_errors_filter_rd2(self):
        '''
        New, values are preloaded in memory
        '''
        # All PEs can start operating at the same time
        # We do it in this way for simplicity
        # Calculate the minimum starting cycle for all PEs
        # We are very onservative here
        cycle_init = 1

        # Insert bubles at the begining
        for h in range(self.array_h):
            for w in range(self.array_w):
                for c in range(cycle_init):
                    self.mem_errors_rd[h][w].append(s.NAN)
                    self.mem_filter_rd[h][w].append(s.NAN)

        # Insert bubbles after each accumulation
        for h in range(self.array_h):
            for w in range(self.array_w):
                for i in range(len(self.__iqacc[h][w])):
                    self.mem_errors_rd[h][w].append(self.mem_errors_rd_nobubbles[h][w][i])
                    self.mem_filter_rd[h][w].append(self.mem_filter_rd_nobubbles[h][w][i])
                    if self.__iqacc[h][w][i] == 1:
                        self.mem_errors_rd[h][w].append(s.NAN)
                        self.mem_filter_rd[h][w].append(s.NAN)
        if debug_mode:
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("mem_errors_rd: "+str(self.mem_errors_rd[h][w]))
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("mem_filter_rd: "+str(self.mem_filter_rd[h][w]))

    def __calc_psum_mem_pos(self):
        # Caculate the read/write groups for each pe
        # To know the PSUM memory position to write to
        self.__psum_mem_pos = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
        for h in range(self.array_h):
            for w in range(self.array_w):
                for i in range(len(self.__pos_errors_rd[h][w])):
                    pair = [self.__pos_filter_rd[h][w][i], self.__pos_errors_rd[h][w][i]]
                    found = False
                    for p in range(len(self.__psum_mem_pos[h][w])):
                        if pair in self.__psum_mem_pos[h][w][p]:
                            found = True
                    if found == False:
                        group = self.__get_group(pair)
                        self.__psum_mem_pos[h][w].append(group)

    def __get_psum_pos(self, h, w,  idx):
        pair = [self.__pos_filter_rd[h][w][idx], self.__pos_errors_rd[h][w][idx]]
        psum_pos = None
        for p in range(len(self.__psum_mem_pos[h][w])):
            if pair in self.__psum_mem_pos[h][w][p]:
                psum_pos = p
        return psum_pos

    def __gen_mem_psum(self):
        '''
        Generate mem psum signals
        rd and wr
        Same group in the same position: this makes things easier
        '''

        self.__calc_psum_mem_pos() # calc

        if debug_mode:
            for h in range(self.array_h):
                for w in range(self.array_w):
                    for i in range(len(self.__psum_mem_pos[h][w])):
                        print("psum_mem_pos["+str(h)+"]["+str(w)+"]: "+str(self.__psum_mem_pos[h][w][i]))

        # Copy
        delay = self.hw[c.MUL] + self.hw[c.SUM] - 1
        for h in range(self.array_h):
            for w in range(self.array_w):

                self.mem_psum_wr[h][w] = self.mem_errors_rd[h][w].copy()
                # Add ALU delay
                for d in range(delay):
                    self.mem_psum_wr[h][w].insert(0,s.NAN)


                # substitute for the actual PSUM memory positions
                # PSUM WRITE
                idx = 0
                for i in range(len(self.mem_psum_wr[h][w])):
                    if self.mem_psum_wr[h][w][i] is not s.NAN:
                        if self.mux_seq[h][w][i] == 1:
                            raise  # This can not happen in our approach

                        psum_pos = self.__get_psum_pos(h,w, idx)
                        # UPDATE PSUM WRITE
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
                self.mem_psum_rd[h][w] = self.mem_errors_rd[h][w].copy()
                for d in range(delay):
                    self.mem_psum_rd[h][w].insert(0,s.NAN)

                idx = 0
                first = []
                next_mux = s.NAN
                for i in range(len(self.mem_psum_rd[h][w])):
                    if self.mem_psum_rd[h][w][i] is not s.NAN:
                        if self.mux_seq[h][w][i] == 1:
                            raise
                        psum_pos = self.__get_psum_pos(h,w, idx)
                        # UPDATE MEM_PSUM_READ
                        next_mux = s.NAN
                        if i+1 < len(self.mux_seq[h][w]):
                            if self.mux_seq[h][w][i+1] == 1:
                                next_mux = psum_pos
                        if psum_pos not in first:
                            #include
                            first.append(psum_pos)
                            # not writing this
                            self.mem_psum_rd[h][w][i] = s.NAN
                        else:
                            self.mem_psum_rd[h][w][i] = psum_pos
                        idx += 1
                    else:
                        # Check if mux is one
                        if i < len(self.mux_seq[h][w]):
                            if self.mux_seq[h][w][i] == 1:
                                self.mem_psum_rd[h][w][i] = next_mux
        if debug_mode:
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("mem_psum_wr["+str(h)+"]["+str(w)+"]: "+str(self.mem_psum_wr[h][w]))
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("mem_psum_rd["+str(h)+"]["+str(w)+"]: "+str(self.mem_psum_rd[h][w]))

    def __gen_mux(self):
        '''
        Generate the mux signals
        '''
        delay = self.hw[c.MUL] + self.hw[c.SUM]
        for h in range(self.array_h):
            for w in range(self.array_w):
                for i in range(len(self.__iqacc[h][w])):
                    # If 1, access the queue

                    idx = get_idx_withnan(self.mem_errors_rd[h][w],i)
                    idx += delay
                    end = idx
                    beging = len(self.mux_seq[h][w])

                    for x in range(beging, end, 1):
                        self.mux_seq[h][w].append(0)

                    if self.__iqacc[h][w][i] == 1:
                        self.mux_seq[h][w].append(1)

        cycle_init = 100000000
        for h in range(self.array_h):
            for w in range(self.array_w):
                cycle_init = min(cycle_init, first_index_noval(self.mux_seq[h][w],0))

        if debug_mode:
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("mux_seq["+str(h)+"]["+str(w)+"]:     "+str(self.mux_seq[h][w]))
            print("mux_seq cycle_init: "+str(cycle_init))


    def __gen_out_psum(self):
        '''
        Determines when the values are going out of the PE (for memory or other PEs)
        Data goes OUT in three different cases:
            1) after accumulation with previous PEs
            2) directly (no read from PSUM)
            3) read from PSUM
        '''
        delay = self.hw[c.MUL] + self.hw[c.SUM]-1
        extra = 0
        for h in range(self.array_h):
            for w in range(self.array_w):
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

    def __gen_gradient_seq(self):
        '''
        Generate the output gradient sequence
        '''
        # Add delay
        delay = 1 # It is delayed one cycle compared to out_psum and mem_psum_wr #self.hw[c.MUL] + self.hw[c.SUM]
        for h in range(self.array_h):
            for w in range(self.array_w):
                for d in range(delay):
                    self.gradient_seq[h][w].append("")

        # Calculate
        idx_abs = [[0 for w in range(len(self.__remote[0]))] for h in range(len(self.__remote))]
        idx_abs_pair = [[0 for w in range(len(self.__remote[0]))] for h in range(len(self.__remote))]
        for h in range(self.array_h):
            for w in range(self.array_w):
                for i in range(len(self.out_psum[h][w])):
                    if self.out_psum[h][w][i] == 1:

                        while self.__remote[h][w][idx_abs[h][w]] != 1:
                            idx_abs[h][w] += 1
                        # Check if this goes to memory or t
                        if self.__remote[h][w][idx_abs[h][w]] == 1 and self.__toMem[h][w][idx_abs[h][w]] == 1:

                            group = self.__toMemGroup[h][w][idx_abs[h][w]]
                            for l in range(len(self.list_groups)):
                                if group == self.list_groups[l]:
                                    self.gradient_seq[h][w].append(self.list_groups_out[l])
                        else:
                            self.gradient_seq[h][w].append("")
                        idx_abs[h][w] += 1

                    else:
                        self.gradient_seq[h][w].append("")

        for h in range(self.array_h):
            for w in range(self.array_w):
                self.gradient_seq[h][w].append("")

        if debug_mode:
            for h in range(self.array_h):
                for w in range(self.array_w):
                    print("gradient_seq["+str(h)+"]["+str(w)+"]:     "+str(self.gradient_seq[h][w]))

    def gen_signals(self):
        '''
        Generate the signals
        Row stationary dataflow
        '''
        self.__gen_multicast_groups()
        if initialize_memories:
            self.__init_mem_errors()
            self.__init_mem_filter()

        # In case of initialized memories, we need to calculate the wr signals, because the rd signals are infered dire
        self.__gen_mem_errors_wr()
        self.__gen_mem_filter_wr()

        # Abstract representations, no real signals yet
        # Prepare the terrain to make the generation of signals easier
        self.__gen_mem_errors_rd_abs()      # no bubles yet
        self.__gen_mem_filter_rd_abs()      # no bubles yet
        self.__gen_data_movements_abs()     # no bubles yet

        # generate the actual signals
        if initialize_memories:
            self.__gen_mem_errors_filter_rd2() # generate errors and filter reads
        else:
            self.__gen_mem_errors_filter_rd() # generate errors and filter reads

        self.__gen_mux()
        self.__gen_mem_psum()
        self.__gen_out_psum()
        self.__gen_gradient_seq()


        # we increase the mux_seq one cycle
        # This was not done before to facilitate the computations of other signals
        for h in range(self.array_h):
            for w in range(self.array_w):
                self.mux_seq[h][w].insert(0,0)

        # Make all the arrays the same size
        if initialize_memories:
            max_size = all_same_size(self.array_h, self.array_w, self.mem_errors_rd, s.NAN, self.mem_filter_rd, s.NAN, self.mem_psum_wr, s.NAN, self.mem_psum_rd, s.NAN, self.mux_seq, 0, self.out_psum, 0,  self.gradient_seq, '')
        else:
            max_size = all_same_size(self.array_h, self.array_w, self.mem_errors_wr, s.NAN, self.mem_errors_rd, s.NAN, self.mem_filter_wr, s.NAN, self.mem_filter_rd, s.NAN, self.mem_psum_wr, s.NAN, self.mem_psum_rd, s.NAN, self.mux_seq, 0, self.out_psum, 0,  self.gradient_seq, '')

        signals = {}


        if initialize_memories:
            self.mem_errors_wr    = [[[] for w in range(self.array_w)] for h in range(self.array_h)]
            self.mem_filter_wr    = [[[] for w in range(self.array_w)] for h in range(self.array_h)]

        signals[s.MEM_IFM_WR]    = self.mem_errors_wr
        signals[s.MEM_FILTER_WR] = self.mem_filter_wr

        signals[s.MEM_IFM_RD]    = self.mem_errors_rd
        signals[s.MEM_FILTER_RD] = self.mem_filter_rd
        signals[s.MEM_PSUM_WR]   = self.mem_psum_wr
        signals[s.MEM_PSUM_RD]   = self.mem_psum_rd
        signals[s.MUX_SEQ]       = self.mux_seq
        signals[s.OUT_PSUM]      = self.out_psum
        signals[s.OFM_SEQ]       = self.gradient_seq

        # Mem init
        signals[s.MEM_IFM_INIT]     = self.mem_errors_init
        signals[s.MEM_FILTER_INIT]  = self.mem_filter_init

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
                    print("    mem_errors_wr: "+str(self.mem_errors_wr[h][w]))
                    print("    mem_errors_rd: "+str(self.mem_errors_rd[h][w]))
                    print("    mem_filter_wr: "+str(self.mem_filter_wr[h][w]))
                    print("    mem_filter_rd: "+str(self.mem_filter_rd[h][w]))
                    print("    mem_psum_wr:   "+str(self.mem_psum_wr[h][w]))
                    print("    mem_psum_rd:   "+str(self.mem_psum_rd[h][w]))
                    print("    mux_seq:       "+str(self.mux_seq[h][w]))
                    print("    out_psum:      "+str(self.out_psum[h][w]))
                    print("    gradient_seq:  "+str(self.gradient_seq[h][w]))


        signals[s.IFM] = self.errors_1D

        # Calculate the string of FILTER values
        fil_w = len(self.fil_t[0])
        fil_h = len(self.fil_t)
        fil_1D = ["" for i in range(fil_w * fil_h)]
        idx = 0
        for w in range(fil_w):
            for h in range(fil_h):
                fil_1D[idx] = self.fil_t[w][h]
                idx += 1
        signals[s.FILTER] = fil_1D

        # Multicast error groups
        multicast_error = [[[] for i in range(self.array_w)] for a in range(self.array_h)]
        for x in range(self.array_h):
            for y in range(self.array_w):
                for i in range(len(self.pe_errors_group)):
                    for a in range(len(self.pe_errors_group[i])):
                        if self.pe_errors_group[i][a] == [x,y]:
                            multicast_error[x][y].append(i)
        signals[s.MULTICAST_IFM] = multicast_error

        if debug_mode:
            for i in range(len(multicast_error)):
                print("multicast_errors: "+str(multicast_error[i]))

        # Multicast filter groups
        multicast_fil = [[[] for i in range(self.array_w)] for a in range(self.array_h)]
        for x in range(self.array_h):
            for y in range(self.array_w):
                for i in range(len(self.pe_fil_group)):
                    for a in range(len(self.pe_fil_group[i])):
                        if self.pe_fil_group[i][a] == [x,y] :
                            multicast_fil[x][y].append(i)
        signals[s.MULTICAST_FILTER] = multicast_fil


        fill_1Darray(max_size, s.NAN, self.filter_seq_multicast)
        fill_1Darray(max_size, s.NAN, self.errors_seq_multicast)

        if debug_mode:
            for i in range(len(multicast_fil)):
                print("multicast_fil: "+str(multicast_fil[i]))
            print("filter_seq_multicast: "+str(self.filter_seq_multicast))
            print("errors_seq_multicast: "+str(self.errors_seq_multicast))

        if initialize_memories:
            signals[s.FILTER_SEQ_MULTICAST] = [-1 for x in range(max_size)]
            signals[s.IFM_SEQ_MULTICAST]    = [-1 for x in range(max_size)]
        else:
            signals[s.FILTER_SEQ_MULTICAST] = self.filter_seq_multicast
            signals[s.IFM_SEQ_MULTICAST]    = self.errors_seq_multicast

        return signals
