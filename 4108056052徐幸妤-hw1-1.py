from inspect import Parameter
from logging import raiseExceptions
from re import L
import tarfile
import random
import copy
import time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import Operator
from scipy import rand
from qiskit.algorithms.linear_solvers.hhl import HHL

# Parameter for KNQTS
population = 1
loop = 5000
DELTA = 0.00133
mu = 1.01
global last_HD
global Q  # prob.matrix
global gb, gw

# Parameter for exp
rand_seed = 114
test = 5

# Parameter for circuit
depth = 15
n = 3  # num of qubit → top qubit = rightmost
global target  # target gate
GATE_SET = {0: 'cx', 1: 'control', 2: 't', 3: 'tdg', 4: 'h', 5: 'copy_bit'}
TARGET = 'fredkin'
pi = 3.14


def target_gate():
    # draw ans circuit
    ans = QuantumCircuit(n)
    if TARGET == 'toffoli':
        ans.ccx(2, 1, 0)
    elif TARGET == 'peres':
        ans.ccx(2, 1, 0)
        ans.cx(2, 1)
    elif TARGET == 'fredkin':
        ans.ccx(2, 1, 0)
        ans.ccx(2, 0, 1)
        ans.ccx(2, 1, 0)
    elif TARGET == 'or':
        ans.cx(2, 0)
        ans.x(2)
        ans.ccx(2, 1, 0)
        ans.x(2)
    elif TARGET == 'toffoli with a 0-control':
        ans.x(2)
        ans.ccx(2, 1, 0)
        ans.x(2)
    elif TARGET == 'hhl':
        # FIXME
        hhl = [[0, 0, 0, 0 - 1j, 0, 0, 0, 0],
               [0, 0, -1, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0],
               [-1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, -1, 0],
               [0, 0, 0, 0, 0, 0, 0, 0 - 1j]]

    op = Operator(ans)
    print(ans)
    return op


class Circuit:
    def __init__(self):
        self.n = n
        self.deep = depth
        self.gate_set = GATE_SET
        self.gate_count = 0
        self.op = ''
        self.fit = 0
        self.correct = False
        self.diff = list()
        self.circuit = QuantumCircuit(n)
        self.measure = list()
        self.exp = ''
        self.i = ''
        for j in range(n):
            self.measure.append(list())

    # generate the circuit and cnt gate count
    def genCircuit(self):
        cnt_gate = 0
        for k in range(self.deep):
            isCx = False
            isCtr = False
            cx_index = 0
            ctr_index = 0
            for j in range(self.n):
                temp = self.measure[j][k]
                if temp == 'cx':
                    isCx = True
                    cx_index = j
                elif temp == 'control':
                    isCtr = True
                    ctr_index = j
                elif temp == 't':
                    self.circuit.t(j)
                    cnt_gate += 1
                elif temp == 'tdg':
                    self.circuit.tdg(j)
                    cnt_gate += 1
                elif temp == 'h':
                    self.circuit.h(j)
                    cnt_gate += 1
                elif temp == 'copy_bit':
                    pass
            if isCx and isCtr:
                self.circuit.cx(ctr_index, cx_index)
                cnt_gate += 1
        self.op = Operator(self.circuit)
        self.gate_count = cnt_gate

    # copy best ans
    def copy(self, a):
        keys = list(self.__dict__.keys())
        for k in keys:
            try:
                setattr(self, k, getattr(getattr(a, k), 'copy')())
            except AttributeError:
                setattr(self, k, getattr(a, k))

    def print(self):
        print("Fit: {}".format(self.fit))
        print("Correct: {}".format(self.correct))
        print("Wrong Bits: {}".format(self.diff))
        print("Gate count: {}".format(self.gate_count))
        print("Circuit graph: \n{}".format(self.circuit))

    def __repr__(self):
        tmp = ""
        for j in range(n):
            for k in range(depth):
                tmp += self.measure[j][k]
                for h in range(10 - len(self.measure[j][k])):
                    tmp += " "
                tmp += ","
            tmp += "\n"
        return tmp

# measure and build the circuit


def measure(x):
    for p in x:
        for a in range(n):  # through n bit
            for b in range(depth):  # through each line
                r = random.random()  # 0~1
                # ↓ measure ↓ #
                if r <= Q[a][b][0]:
                    p.measure[a].append(GATE_SET[0])
                elif Q[a][b][0] < r <= (Q[a][b][0] + Q[a][b][1]):
                    p.measure[a].append(GATE_SET[1])
                elif (Q[a][b][0] + Q[a][b][1]) < r <= (Q[a][b][0] + Q[a][b][1] + Q[a][b][2]):
                    p.measure[a].append(GATE_SET[2])
                elif (Q[a][b][0] + Q[a][b][1] + Q[a][b][2]) < r <= (Q[a][b][0] + Q[a][b][1] + Q[a][b][2] + Q[a][b][3]):
                    p.measure[a].append(GATE_SET[3])
                elif (Q[a][b][0] + Q[a][b][1] + Q[a][b][2] + Q[a][b][3]) < r <= (Q[a][b][0] + Q[a][b][1] + Q[a][b][2] +
                                                                                 Q[a][b][3] + Q[a][b][4]):
                    p.measure[a].append(GATE_SET[4])
                else:
                    p.measure[a].append(GATE_SET[5])


def cntCOP(target_op, circuit_op):
    COP = 0
    possible_input = list()
    diff = list()

    # generate 000 ~ 111 input
    for a in range(pow(2, n)):
        possible_input.append(str(decToBin(a)).zfill(3))
        for b in range(pow(2, n)):
            circuit_op.data[a][b] = np.round(
                circuit_op.data[a][b], 2)  # 四捨五入目前矩陣
        # 比較每一列看是否相同
        if (target_op.data[a] == circuit_op.data[a]).all():
            COP += 1
        else:
            diff.append(possible_input[a])
    return COP, diff


def circuit_list():
    circuit_num = list()
    for p in range(population):
        circuit_num.append(list())  # build circuit list (null)
    return circuit_num


def initx(x, expTime, ecx):
    for p in range(population):
        x[p] = Circuit()
        x[p].exp = expTime
        x[p].i = ecx


def decToBin(d):
    b = format(d, 'b')
    return b


def binToDec(b):
    d = int(str(b), 2)
    return d


def presetAns(p):
    if TARGET == 'toffoli':
        p.measure[0][0] = 'copy_bit'
        p.measure[0][1] = 'h'
        p.measure[0][2] = 't'
        p.measure[0][3] = 'control'
        p.measure[0][4] = 'cx'
        p.measure[0][5] = 'tdg'
        p.measure[0][6] = 'copy_bit'
        p.measure[0][7] = 'copy_bit'
        p.measure[0][8] = 'cx'
        p.measure[0][9] = 'control'
        p.measure[0][10] = 'h'
        p.measure[0][11] = 'copy_bit'
        p.measure[0][12] = 'copy_bit'
        p.measure[0][13] = 'copy_bit'
        p.measure[0][14] = 'copy_bit'

        p.measure[1][0] = 'copy_bit'
        p.measure[1][1] = 'cx'
        p.measure[1][2] = 'tdg'
        p.measure[1][3] = 'cx'
        p.measure[1][4] = 'copy_bit'
        p.measure[1][5] = 't'
        p.measure[1][6] = 'cx'
        p.measure[1][7] = 'tdg'
        p.measure[1][8] = 'copy_bit'
        p.measure[1][9] = 'cx'
        p.measure[1][10] = 't'
        p.measure[1][11] = 'copy_bit'
        p.measure[1][12] = 'copy_bit'
        p.measure[1][13] = 'copy_bit'
        p.measure[1][14] = 'copy_bit'

        p.measure[2][0] = 'copy_bit'
        p.measure[2][1] = 'control'
        p.measure[2][2] = 't'
        p.measure[2][3] = 'copy_bit'
        p.measure[2][4] = 'control'
        p.measure[2][5] = 'copy_bit'
        p.measure[2][6] = 'control'
        p.measure[2][7] = 'copy_bit'
        p.measure[2][8] = 'control'
        p.measure[2][9] = 'copy_bit'
        p.measure[2][10] = 'copy_bit'
        p.measure[2][11] = 'copy_bit'
        p.measure[2][12] = 'copy_bit'
        p.measure[2][13] = 'copy_bit'
        p.measure[2][14] = 'copy_bit'

    elif TARGET == 'peres':
        p.measure[0][0] = 'h'
        p.measure[0][1] = 'cx'
        p.measure[0][2] = 'tdg'
        p.measure[0][3] = 'cx'
        p.measure[0][4] = 't'
        p.measure[0][5] = 'cx'
        p.measure[0][6] = 'tdg'
        p.measure[0][7] = 'cx'
        p.measure[0][8] = 't'
        p.measure[0][9] = 'h'
        p.measure[0][10] = 'copy_bit'
        p.measure[0][11] = 'copy_bit'
        p.measure[0][12] = 'copy_bit'
        p.measure[0][13] = 'copy_bit'
        p.measure[0][14] = 'copy_bit'

        p.measure[1][0] = 'copy_bit'
        p.measure[1][1] = 'control'
        p.measure[1][2] = 'copy_bit'
        p.measure[1][3] = 'copy_bit'
        p.measure[1][4] = 'copy_bit'
        p.measure[1][5] = 'control'
        p.measure[1][6] = 't'
        p.measure[1][7] = 'copy_bit'
        p.measure[1][8] = 'cx'
        p.measure[1][9] = 'tdg'
        p.measure[1][10] = 'copy_bit'
        p.measure[1][11] = 'copy_bit'
        p.measure[1][12] = 'copy_bit'
        p.measure[1][13] = 'copy_bit'
        p.measure[1][14] = 'copy_bit'

        p.measure[2][0] = 'copy_bit'
        p.measure[2][1] = 'copy_bit'
        p.measure[2][2] = 'copy_bit'
        p.measure[2][3] = 'control'
        p.measure[2][4] = 't'
        p.measure[2][5] = 'copy_bit'
        p.measure[2][6] = 'copy_bit'
        p.measure[2][7] = 'control'
        p.measure[2][8] = 'control'
        p.measure[2][9] = 'copy_bit'
        p.measure[2][10] = 'copy_bit'
        p.measure[2][11] = 'copy_bit'
        p.measure[2][12] = 'copy_bit'
        p.measure[2][13] = 'copy_bit'
        p.measure[2][14] = 'copy_bit'
    elif TARGET == 'fredkin':
        # p.measure[0][0] = 'control'
        # p.measure[0][1] = 'h'
        # p.measure[0][2] = 'cx'
        # p.measure[0][3] = 'tdg'
        # p.measure[0][4] = 'cx'
        # p.measure[0][5] = 't'
        # p.measure[0][6] = 'cx'
        # p.measure[0][7] = 'tdg'
        # p.measure[0][8] = 'cx'
        # p.measure[0][9] = 't'
        # p.measure[0][10] = 'h'
        # p.measure[0][11] = 'copy_bit'
        # p.measure[0][12] = 'control'  # final
        # p.measure[0][13] = 'copy_bit'
        # p.measure[0][14] = 'copy_bit'

        # p.measure[1][0] = 'cx'
        # p.measure[1][1] = 'copy_bit'
        # p.measure[1][2] = 'control'
        # p.measure[1][3] = 'copy_bit'
        # p.measure[1][4] = 'copy_bit'
        # p.measure[1][5] = 'copy_bit'
        # p.measure[1][6] = 'control'
        # p.measure[1][7] = 't'
        # p.measure[1][8] = 'copy_bit'
        # p.measure[1][9] = 'cx'
        # p.measure[1][10] = 'tdg'
        # p.measure[1][11] = 'cx'
        # p.measure[1][12] = 'cx'  # final
        # p.measure[1][13] = 'copy_bit'
        # p.measure[1][14] = 'copy_bit'

        # p.measure[2][0] = 'copy_bit'
        # p.measure[2][1] = 'copy_bit'
        # p.measure[2][2] = 'copy_bit'
        # p.measure[2][3] = 'copy_bit'
        # p.measure[2][4] = 'control'
        # p.measure[2][5] = 't'
        # p.measure[2][6] = 'copy_bit'
        # p.measure[2][7] = 'copy_bit'
        # p.measure[2][8] = 'control'
        # p.measure[2][9] = 'control'
        # p.measure[2][10] = 'copy_bit'
        # p.measure[2][11] = 'control'
        # p.measure[2][12] = 'copy_bit'  # final
        # p.measure[2][13] = 'copy_bit'
        # p.measure[2][14] = 'copy_bit'

        p.measure[0][0] = 'h'
        p.measure[0][1] = 'control'
        p.measure[0][2] = 'cx'
        p.measure[0][3] = 'tdg'
        p.measure[0][4] = 'copy_bit'
        p.measure[0][5] = 'cx'
        p.measure[0][6] = 't'
        p.measure[0][7] = 'control'
        p.measure[0][8] = 'h'
        p.measure[0][9] = 't'
        p.measure[0][10] = 'tdg'
        p.measure[0][11] = 'copy_bit'
        p.measure[0][12] = 'copy_bit'  # final
        p.measure[0][13] = 'copy_bit'
        p.measure[0][14] = 'copy_bit'

        p.measure[1][0] = 't'
        p.measure[1][1] = 'cx'
        p.measure[1][2] = 'copy_bit'
        p.measure[1][3] = 'tdg'
        p.measure[1][4] = 'cx'
        p.measure[1][5] = 'copy_bit'
        p.measure[1][6] = 't'
        p.measure[1][7] = 'cx'
        p.measure[1][8] = 'tdg'
        p.measure[1][9] = 'copy_bit'
        p.measure[1][10] = 'copy_bit'
        p.measure[1][11] = 'copy_bit'
        p.measure[1][12] = 'copy_bit'  # final
        p.measure[1][13] = 'copy_bit'
        p.measure[1][14] = 'copy_bit'

        p.measure[2][0] = 't'
        p.measure[2][1] = 'copy_bit'
        p.measure[2][2] = 'control'
        p.measure[2][3] = 't'
        p.measure[2][4] = 'control'
        p.measure[2][5] = 'control'
        p.measure[2][6] = 'tdg'
        p.measure[2][7] = 'copy_bit'
        p.measure[2][8] = 'copy_bit'
        p.measure[2][9] = 'copy_bit'
        p.measure[2][10] = 'copy_bit'
        p.measure[2][11] = 'copy_bit'
        p.measure[2][12] = 'copy_bit'  # final
        p.measure[2][13] = 'copy_bit'
        p.measure[2][14] = 'copy_bit'
    elif TARGET == 'or':
        # p.measure[0][0] = 'h'
        # p.measure[0][1] = 'control'
        # p.measure[0][2] = 'tdg'
        # p.measure[0][3] = 'copy_bit'
        # p.measure[0][4] = 'control'
        # p.measure[0][5] = 'tdg'
        # p.measure[0][6] = 'control'
        # p.measure[0][7] = 'tdg'
        # p.measure[0][8] = 'control'
        # p.measure[0][9] = 'h'
        # p.measure[0][10] = 'copy_bit'
        # p.measure[0][11] = 'copy_bit'
        # p.measure[0][12] = 'copy_bit'
        # p.measure[0][13] = 'copy_bit'
        # p.measure[0][14] = 'copy_bit'

        # p.measure[1][0] = 'tdg'
        # p.measure[1][1] = 'cx'
        # p.measure[1][2] = 't'
        # p.measure[1][3] = 'cx'
        # p.measure[1][4] = 'copy_bit'
        # p.measure[1][5] = 't'
        # p.measure[1][6] = 'cx'
        # p.measure[1][7] = 'tdg'
        # p.measure[1][8] = 'copy_bit'
        # p.measure[1][9] = 'cx'
        # p.measure[1][10] = 'copy_bit'
        # p.measure[1][11] = 'copy_bit'
        # p.measure[1][12] = 'copy_bit'
        # p.measure[1][13] = 'copy_bit'
        # p.measure[1][14] = 'copy_bit'

        # p.measure[2][0] = 'copy_bit'
        # p.measure[2][1] = 'copy_bit'
        # p.measure[2][2] = 'tdg'
        # p.measure[2][3] = 'control'
        # p.measure[2][4] = 'cx'
        # p.measure[2][5] = 'copy_bit'
        # p.measure[2][6] = 'copy_bit'
        # p.measure[2][7] = 't'
        # p.measure[2][8] = 'cx'
        # p.measure[2][9] = 'control'
        # p.measure[2][10] = 'copy_bit'
        # p.measure[2][11] = 'copy_bit'
        # p.measure[2][12] = 'copy_bit'
        # p.measure[2][13] = 'copy_bit'
        # p.measure[2][14] = 'copy_bit'

        p.measure[0][0] = 'h'
        p.measure[0][1] = 'control'
        p.measure[0][2] = 'tdg'
        p.measure[0][3] = 'copy_bit'
        p.measure[0][4] = 'control'
        p.measure[0][5] = 'tdg'
        p.measure[0][6] = 'control'
        p.measure[0][7] = 'tdg'
        p.measure[0][8] = 'control'
        p.measure[0][9] = 'h'
        p.measure[0][10] = 'copy_bit'
        p.measure[0][11] = 'copy_bit'
        p.measure[0][12] = 'copy_bit'
        p.measure[0][13] = 'copy_bit'
        p.measure[0][14] = 'copy_bit'

        p.measure[1][0] = 'tdg'
        p.measure[1][1] = 'cx'
        p.measure[1][2] = 't'
        p.measure[1][3] = 'cx'
        p.measure[1][4] = 'copy_bit'
        p.measure[1][5] = 't'
        p.measure[1][6] = 'cx'
        p.measure[1][7] = 'tdg'
        p.measure[1][8] = 'copy_bit'
        p.measure[1][9] = 'cx'
        p.measure[1][10] = 'copy_bit'
        p.measure[1][11] = 'copy_bit'
        p.measure[1][12] = 'copy_bit'
        p.measure[1][13] = 'copy_bit'
        p.measure[1][14] = 'copy_bit'

        p.measure[2][0] = 'copy_bit'
        p.measure[2][1] = 'copy_bit'
        p.measure[2][2] = 'copy_bit'
        p.measure[2][3] = 'control'
        p.measure[2][4] = 'cx'
        p.measure[2][5] = 't'
        p.measure[2][6] = 'copy_bit'
        p.measure[2][7] = 'copy_bit'
        p.measure[2][8] = 'cx'
        p.measure[2][9] = 'control'
        p.measure[2][10] = 'tdg'
        p.measure[2][11] = 'copy_bit'
        p.measure[2][12] = 'copy_bit'
        p.measure[2][13] = 'copy_bit'
        p.measure[2][14] = 'copy_bit'

    elif TARGET == 'toffoli with a 0-control':
        # p.measure[0][0] = 'h'
        # p.measure[0][1] = 'control'
        # p.measure[0][2] = 'tdg'
        # p.measure[0][3] = 'copy_bit'
        # p.measure[0][4] = 'control'
        # p.measure[0][5] = 'copy_bit'
        # p.measure[0][6] = 'control'
        # p.measure[0][7] = 'control'
        # p.measure[0][8] = 'h'
        # p.measure[0][9] = 'copy_bit'
        # p.measure[0][10] = 'copy_bit'
        # p.measure[0][11] = 'copy_bit'
        # p.measure[0][12] = 'copy_bit'
        # p.measure[0][13] = 'copy_bit'
        # p.measure[0][14] = 'copy_bit'

        # p.measure[1][0] = 'copy_bit'
        # p.measure[1][1] = 'copy_bit'
        # p.measure[1][2] = 'tdg'
        # p.measure[1][3] = 'control'
        # p.measure[1][4] = 'cx'
        # p.measure[1][5] = 't'
        # p.measure[1][6] = 'copy_bit'
        # p.measure[1][7] = 'cx'
        # p.measure[1][8] = 'control'
        # p.measure[1][9] = 'copy_bit'
        # p.measure[1][10] = 'copy_bit'
        # p.measure[1][11] = 'copy_bit'
        # p.measure[1][12] = 'copy_bit'
        # p.measure[1][13] = 'copy_bit'
        # p.measure[1][14] = 'copy_bit'

        # p.measure[2][0] = 't'
        # p.measure[2][1] = 'cx'
        # p.measure[2][2] = 'tdg'
        # p.measure[2][3] = 'cx'
        # p.measure[2][4] = 'copy_bit'
        # p.measure[2][5] = 't'
        # p.measure[2][6] = 'cx'
        # p.measure[2][7] = 'tdg'
        # p.measure[2][8] = 'cx'
        # p.measure[2][9] = 'copy_bit'
        # p.measure[2][10] = 'copy_bit'
        # p.measure[2][11] = 'copy_bit'
        # p.measure[2][12] = 'copy_bit'
        # p.measure[2][13] = 'copy_bit'
        # p.measure[2][14] = 'copy_bit'

        p.measure[0][0] = 'h'
        p.measure[0][1] = 'control'
        p.measure[0][2] = 'tdg'
        p.measure[0][3] = 'copy_bit'
        p.measure[0][4] = 'control'
        p.measure[0][5] = 'copy_bit'
        p.measure[0][6] = 'control'
        p.measure[0][7] = 'control'
        p.measure[0][8] = 'h'
        p.measure[0][9] = 'copy_bit'
        p.measure[0][10] = 'copy_bit'
        p.measure[0][11] = 'copy_bit'
        p.measure[0][12] = 'copy_bit'
        p.measure[0][13] = 'copy_bit'
        p.measure[0][14] = 'copy_bit'

        p.measure[1][0] = 'copy_bit'
        p.measure[1][1] = 'copy_bit'
        p.measure[1][2] = 'copy_bit'
        p.measure[1][3] = 'control'
        p.measure[1][4] = 'cx'
        p.measure[1][5] = 't'
        p.measure[1][6] = 'copy_bit'
        p.measure[1][7] = 'cx'
        p.measure[1][8] = 'control'
        p.measure[1][9] = 'tdg'
        p.measure[1][10] = 'copy_bit'
        p.measure[1][11] = 'copy_bit'
        p.measure[1][12] = 'copy_bit'
        p.measure[1][13] = 'copy_bit'
        p.measure[1][14] = 'copy_bit'

        p.measure[2][0] = 't'
        p.measure[2][1] = 'cx'
        p.measure[2][2] = 'tdg'
        p.measure[2][3] = 'cx'
        p.measure[2][4] = 't'
        p.measure[2][5] = 'copy_bit'
        p.measure[2][6] = 'cx'
        p.measure[2][7] = 'tdg'
        p.measure[2][8] = 'cx'
        p.measure[2][9] = 'copy_bit'
        p.measure[2][10] = 'copy_bit'
        p.measure[2][11] = 'copy_bit'
        p.measure[2][12] = 'copy_bit'
        p.measure[2][13] = 'copy_bit'
        p.measure[2][14] = 'copy_bit'

    p.genCircuit()
    print(p.circuit)
    p.circuit.draw('mpl', filename="peres.png")
    print(p.op)


if __name__ == '__main__':
    random.seed(rand_seed)
    Q = np.full((n, depth, len(GATE_SET)), 1 / 6)
    target_op = target_gate()  # get target operator(matrix)
    expBest = Circuit()
    expBest.fit = float('-inf')
    x = circuit_list()
    initx(x, 1, 1)
    measure(x)  # built circuit
    presetAns(x[0])
    # x[0].genCircuit()
    print(x[0].circuit)
    COP, a = cntCOP(target_op, x[0].op)
    print(target_op)
    print(COP)
