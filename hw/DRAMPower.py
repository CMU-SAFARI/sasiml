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
import math
import os
import logging, sys

DEBUG = True
memory_model = "MICRON_4Gb_DDR4-1866_8bit_A.xml"
sasimpath = os.environ['SASIMPATH']
results_path = sasimpath+"/drampower_results"
memory_clock = 900 # MHz

class DRAMPower(object):
    def __init__(self, dram_rd_ifm, dram_rd_fil, dram_wr_ofm, cycles, clock):
        if DEBUG:
            logging.basicConfig(stream=sys.stderr, level=logging.DEBUG) # Managing the prints
        else:
            logging.basicConfig(stream=sys.stderr, level=logging.INFO) # Managing the prints
        # File for traces
        self.ftrace = "ifm_"+str(dram_rd_ifm)+"_fil_"+str(dram_rd_fil)+"_ofm_"+str(dram_wr_ofm)+".trace"
        # File for DRAMPower results
        self.fout = "ifm_"+str(dram_rd_ifm)+"_fil_"+str(dram_rd_fil)+"_ofm_"+str(dram_wr_ofm)+".out"

        # Parameters
        self.block_size = 8
        self.dram_columns = 8192/self.block_size
        self.banks = 16

        # Each element is 2 bytes
        self.I = math.ceil(dram_rd_ifm /self.block_size )
        self.F = math.ceil(dram_rd_fil/self.block_size)
        self.O = math.ceil(dram_wr_ofm /self.block_size)


        print("I: "+str(self.I))
        print("F: "+str(self.F))
        print("O: "+str(self.O))

        total = self.block_size * (self.I + self.F + self.O)
        print("Total data: "+str(total/(1024*1024))+" MB")
        print("Total cycles accelerator: "+str(cycles))

        dram_cycles = cycles * memory_clock / clock
        time_interval = int(dram_cycles/(self.I + self.F + self.O))
        if time_interval < 1:
            time_interval = 1
        # Timing parameters
        T_ACT = 15
        T_PRE = 15
        T_RD = time_interval
        T_WR = time_interval

        self.LAT = {
            'ACT': T_ACT,
            'PRE': T_PRE,
            'RD': T_RD,
            'WR': T_WR,
        }

    # Always write to bank 0
    def write(self,file, time, cmd, bank = 0):
        file.write(','.join([str(time), cmd, str(bank)]))
        file.write('\n')
        return self.LAT[cmd]

    def generate_traces(self):
        '''
        Generate traces for DRAM Power
        '''
        logging.debug("Generating traces")
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        if os.path.exists(results_path+"/"+self.ftrace) and not DEBUG:
            return

        time = 0
        fout = open(results_path+"/"+self.ftrace, 'w')

        banks_per_datat = 5
        offset_ifm = 0
        offset_fil = 5
        offset_ofm = 10
        time_ifm = 0
        time_fil = 0
        time_ofm = 0
        bank_ifm = 0
        bank_fil = 0
        bank_ofm = 0

        bank_count = []
        for b in range(self.banks):
            bank_count.append(0)
            self.write(fout, 0, 'ACT', b)

        bank_time = []
        for b in range(self.banks):
            bank_time.append(0)

        while True:
            if self.I > 0:
                bank_time[bank_ifm + offset_ifm] += self.write(fout, bank_time[bank_ifm + offset_ifm], 'RD', bank_ifm + offset_ifm)
                if (bank_count[bank_ifm + offset_ifm] % self.dram_columns) == 0:
                    bank_time[bank_ifm + offset_ifm] += self.write(fout, bank_time[bank_ifm + offset_ifm], 'PRE', bank_ifm + offset_ifm)
                    bank_time[bank_ifm + offset_ifm] += self.write(fout, bank_time[bank_ifm + offset_ifm], 'ACT', bank_ifm + offset_ifm)
                # Decrease the number of required accesses
                self.I -= 1
                # Next bank
                bank_ifm += 1
                bank_count[bank_ifm + offset_ifm] += 1
                if bank_ifm == banks_per_datat:
                    bank_ifm = 0
            if self.F > 0:
                bank_time[bank_fil + offset_fil] += self.write(fout, bank_time[bank_fil + offset_fil], 'RD', bank_fil + offset_fil)
                if (bank_count[bank_fil + offset_fil] % self.dram_columns) == 0:
                    bank_time[bank_fil + offset_fil] += self.write(fout, bank_time[bank_fil + offset_fil], 'PRE', bank_fil + offset_fil)
                    bank_time[bank_fil + offset_fil] += self.write(fout, bank_time[bank_fil + offset_fil], 'ACT', bank_fil + offset_fil)
                # Decrease the number of required accesses
                self.F -= 1
                # Next bank
                bank_fil += 1
                bank_count[bank_fil + offset_fil] += 1
                if bank_fil == banks_per_datat:
                    bank_fil = 0
            if self.O > 0:
                bank_time[bank_ofm + offset_ofm] += self.write(fout, bank_time[bank_ofm + offset_ofm], 'WR', bank_ofm + offset_ofm)
                if (bank_count[bank_ofm + offset_ofm] % self.dram_columns) == 0:
                    bank_time[bank_ofm + offset_ofm] += self.write(fout, bank_time[bank_ofm + offset_ofm], 'PRE', bank_ofm + offset_ofm)
                    bank_time[bank_ofm + offset_ofm] += self.write(fout, bank_time[bank_ofm + offset_ofm], 'ACT', bank_ofm + offset_ofm)
                # Decrease the number of required accesses
                self.O -= 1
                # Next bank
                bank_ofm += 1
                bank_count[bank_ofm + offset_ofm] += 1
                if bank_ofm == banks_per_datat:
                    bank_ofm = 0

            if self.I == 0 and self.F == 0 and self.O == 0:
                break

        fout.close()

    def simulate_energy(self):
        '''
        Simulate with DRAM Power
        '''
        if not os.path.exists(results_path+"/"+self.fout) or DEBUG:
            # Run DRAM Power
            sasimpath = os.environ['SASIMPATH']
            path = sasimpath+"/DRAMPower/"
            cmd = path+"drampower -m "+path+"memspecs/"+memory_model+" -c "+results_path+"/"+self.ftrace+" > "+ results_path + "/" + self.fout
            logging.debug(cmd)
            os.system(cmd)

        # Get the energy
        energy = 0
        with open(results_path+"/"+self.fout, 'r') as fin:
            line = fin.readline()
            while line:
                if "Total Trace Energy:" in line:
                    energy = float(line.split(" ")[3])
                    logging.debug("energy: "+str(energy)+" pJ")
                    break
                line = fin.readline()
        return energy

