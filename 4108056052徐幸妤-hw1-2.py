import random

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

# Parameter for KNQTS
population = 200
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
        pass

    op = Operator(ans)
    print(ans)
    return op


class Circuit:
    def __init__(self):
        self.n = n
        self.deep = depth
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


def maxProbGate(num, k):
    prob = float('-inf')
    gate = ''
    for h in range(2, len(GATE_SET)):
        if Q[num][k][h] > prob:
            prob = Q[num][k][h]
            gate = GATE_SET[h]
    return gate


# case 1 : 有 control 也有 not  → 兩個 CONTROL → 砍一個
#                               → 兩個 not → 砍一個
# case 2 : 有 control 沒有 not  → 沒有 NOT 最高 → 換掉 CONTROL → 換不掉CONTROL → 隨便換
# case 3 : 有 not 沒有 control  → 沒有 CONTROL 最高 → 換掉 NOT → 換不掉NOT → 隨便換
def repair(x):
    for p in x:
        for k in range(depth):
            isCx = False
            isControl = False
            cx_index = 0
            ctr_index = 0
            control_list = np.full(n, -1)
            not_list = np.full(n, -1)
            not_index = 0
            control_index = 0
            max_prob_gate = np.full(n, '')
            max_prob = np.full(n, -1, dtype=float)

            for j in range(n):  # n bits
                # 找 每一個 bit 最有可能的gate和其最大機率
                max_prob[j] = max(Q[j][k])
                max_prob_gate[j] = GATE_SET[Q[j]
                                            [k].tolist().index(max_prob[j])]
                # 紀錄目前選中的gate
                if p.measure[j][k] == 'cx':
                    not_list[not_index] = j
                    not_index += 1
                elif p.measure[j][k] == 'control':
                    control_list[control_index] = j
                    control_index += 1

                if control_list[0] != -1:
                    isControl = True
                if not_list[0] != -1:
                    isCx = True

            if control_list[0] != -1 and not_list[0] != -1:  # case 1
                if control_list[1] != -1:  # 兩個 CONTROL → 砍一個
                    prob0 = Q[control_list[0]][k][1]
                    prob1 = Q[control_list[1]][k][1]
                    if prob0 > prob1:  # change 1
                        p.measure[control_list[1]][k] = maxProbGate(
                            control_list[1], k)
                    else:  # change 0
                        p.measure[control_list[0]][k] = maxProbGate(
                            control_list[0], k)
                elif not_list[1] != -1:  # 兩個 CONTROL → 砍一個
                    prob0 = Q[not_list[0]][k][0]
                    prob1 = Q[not_list[1]][k][0]
                    if prob0 > prob1:  # change 1
                        p.measure[not_list[1]][k] = maxProbGate(
                            not_list[1], k)
                    else:  # change 0
                        p.measure[not_list[0]][k] = maxProbGate(
                            not_list[0], k)

            # 確認完 n bit 後 -> 有 not 沒有 control
            if isCx and not isControl:
                prob = 0
                gate_index = -1
                for j in range(n):  # 找是 CONTROL 機率最高的
                    if Q[j][k][1] > prob:
                        prob = Q[j][k][1]
                        gate_index = j
                if prob != 0:
                    p.measure[gate_index][k] = 'control'
                    isControl = True
                else:  # 不該選中 control
                    prob = 0
                    gate = ''
                    for h in range(2, len(GATE_SET)):  # 找機率max換掉 not
                        if Q[cx_index][k][h] > prob:
                            prob = Q[cx_index][k][h]
                            gate = GATE_SET[h]
                    p.measure[cx_index][k] = gate
                    isCx = False

                    if gate == '':  # 找不到比not還大的
                        p.measure[cx_index][k] = 'cx'
                        r = random.randrange(n - 1)  # 隨便選一個換control
                        if r >= cx_index:
                            r += 1
                        p.measure[r][k] = 'control'
                        isControl = True
                        isCx = True

            if not isCx and isControl:  # 有control 沒not
                prob = 0
                gate_index = -1
                for j in range(n):  # 找 control給他
                    if Q[j][k][0] > prob:
                        prob = Q[j][k][0]
                        gate_index = j

                if prob != 0:
                    p.measure[gate_index][k] = 'cx'
                else:  # 找不到control給他
                    prob = 0
                    gate = ''
                    for h in range(2, len(GATE_SET)):  # 換掉control
                        if Q[ctr_index][k][h] > prob:
                            prob = Q[ctr_index][k][h]
                            gate = GATE_SET[h]
                    p.measure[ctr_index][k] = gate

                    if gate == '':  # 換不掉control
                        p.measure[ctr_index][k] = 'control'
                        r = random.randrange(n - 1)  # 隨便換掉not
                        if r >= ctr_index:
                            r += 1
                        p.measure[r][k] = 'cx'


def decToBin(d):
    b = format(d, 'b')
    return b


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


def fitness(x, target_op):
    for idx, p in enumerate(x):
        w = 0  # use fit 2 or not

        p.genCircuit()

        COP, p.diff = cntCOP(target_op, p.op)
        WG = (n * depth) - p.gate_count
        fit1 = COP / pow(2, n)
        fit2 = WG / (n * depth)

        if fit1 == 1:
            w = 1
            p.correct = True

        p.fit = fit1 + w * fit2


def recordAndUpdate(gb, gw, lb, lw, ecx, expTime):
    if lb.fit > gb.fit:
        if ecx == 0 or lb.measure != target.measure:
            gb.copy(lb)
        if lb.measure == target.measure:
            gb.fit = 0.99

        if gb.fit >= 1:
            gb.circuit.draw('mpl', filename=str(expTime) + "_" + str(ecx) + '_' + str(gb.fit)
                            + "_myCircuit.png")
    if lw.fit <= gw.fit:
        gw.copy(lw)


def update(x, gb, gw, lb, lw, ecx, expTime, last_HD, delta):
    # find local best and local worst
    lb.copy(x[0])
    lw.copy(x[-1])
    for p in x:
        if p.fit >= lb.fit:
            lb.copy(p)
        if p.fit <= lw.fit:
            lw.copy(p)

    recordAndUpdate(gb, gw, lb, lw, ecx, expTime)

    # ↓ adjust delta ↓ #
    # cnt ham
    HD = 0
    for a in range(n):
        for b in range(depth):
            if lb.measure[a][b] != lw.measure[a][b]:
                HD += 1
    last_HD = HD if last_HD == float('inf') else last_HD

    if HD > last_HD:  # 差異變大
        delta *= mu
    elif HD < last_HD:  # 差異變小
        delta *= (2 - mu)
    else:
        pass

    # ↓ update matrix ↓ #
    for a in range(n):
        for b in range(depth):
            if gb.measure[a][b] != gw.measure[a][b]:
                try:
                    gb_index = list(GATE_SET.values()).index(gb.measure[a][b])
                    gw_index = list(GATE_SET.values()).index(gw.measure[a][b])
                    Q[a][b][gb_index] += delta
                    Q[a][b][gw_index] -= delta

                    # repair prob
                    if Q[a][b][gw_index] <= 0:
                        Q[a][b][gw_index] = 0

                        Q[a][b][gb_index] = 1
                        for c in range(len(GATE_SET)):
                            if c != gb_index:
                                Q[a][b][gb_index] -= Q[a][b][c]

                    # Quantum NOT
                    if Q[a][b][gb_index] < Q[a][b][gw_index]:
                        Q[a][b][gb_index], Q[a][b][gw_index] = Q[a][b][gw_index], Q[a][b][gb_index]

                        # max_prob = max(Q[a][b])
                        # maxIndex = Q[a][b].tolist().index(max_prob)
                        # min_prob = min(Q[a][b])
                        # minIndex = Q[a][b].tolist().index(min_prob)

                        # # swap prob.
                        # Q[a][b][maxIndex] = Q[a][b][gb_index]  # max 設成gb
                        # Q[a][b][gb_index] = max_prob  # gb 設成 max prob
                        # Q[a][b][minIndex] = Q[a][b][gw_index]  # min 設成 gw
                        # Q[a][b][gw_index] = min_prob  # gw 設成min
                except ValueError:
                    print(gb)
                    print(gw)
                    print(gb.measure[a][b])
                    print(gw.measure[a][b])
                    print(a)
                    print(b)
                    raise
    return HD


def presetAns(p):
    if TARGET == 'toffoli':
        # p.measure[0][0] = 'h'
        # p.measure[0][1] = 'cx'
        # p.measure[0][2] = 'tdg'
        # p.measure[0][3] = 'cx'
        # p.measure[0][4] = 't'
        # p.measure[0][5] = 'cx'
        # p.measure[0][6] = 'tdg'
        # p.measure[0][7] = 'cx'
        # p.measure[0][8] = 't'
        # p.measure[0][9] = 'h'
        # p.measure[0][10] = 'copy_bit'
        # p.measure[0][11] = 'copy_bit'
        # p.measure[0][12] = 'copy_bit'
        # p.measure[0][13] = 'copy_bit'
        # p.measure[0][14] = 'copy_bit'

        # p.measure[1][0] = 'copy_bit'
        # p.measure[1][1] = 'control'
        # p.measure[1][2] = 'copy_bit'
        # p.measure[1][3] = 'copy_bit'
        # p.measure[1][4] = 'copy_bit'
        # p.measure[1][5] = 'control'
        # p.measure[1][6] = 'copy_bit'
        # p.measure[1][7] = 'copy_bit'
        # p.measure[1][8] = 't'
        # p.measure[1][9] = 'cx'
        # p.measure[1][10] = 'tdg'
        # p.measure[1][11] = 'cx'
        # p.measure[1][12] = 'copy_bit'
        # p.measure[1][13] = 'copy_bit'
        # p.measure[1][14] = 'copy_bit'

        # p.measure[2][0] = 'copy_bit'
        # p.measure[2][1] = 'copy_bit'
        # p.measure[2][2] = 'copy_bit'
        # p.measure[2][3] = 'control'
        # p.measure[2][4] = 'copy_bit'
        # p.measure[2][5] = 'copy_bit'
        # p.measure[2][6] = 'copy_bit'
        # p.measure[2][7] = 'control'
        # p.measure[2][8] = 'copy_bit'
        # p.measure[2][9] = 'control'
        # p.measure[2][10] = 't'
        # p.measure[2][11] = 'control'
        # p.measure[2][12] = 'copy_bit'
        # p.measure[2][13] = 'copy_bit'
        # p.measure[2][14] = 'copy_bit'

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
        p.measure[0][1] = 'control'
        p.measure[0][2] = 'cx'
        p.measure[0][3] = 'tdg'
        p.measure[0][4] = 'copy_bit'
        p.measure[0][5] = 'cx'
        p.measure[0][6] = 't'
        p.measure[0][7] = 'control'
        p.measure[0][8] = 'h'
        p.measure[0][9] = 'copy_bit'
        p.measure[0][10] = 'copy_bit'
        p.measure[0][11] = 'copy_bit'
        p.measure[0][12] = 'copy_bit'
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
        p.measure[1][12] = 'copy_bit'
        p.measure[1][13] = 'copy_bit'
        p.measure[1][14] = 'copy_bit'

        p.measure[2][0] = 't'
        p.measure[2][1] = 'copy_bit'
        p.measure[2][2] = 'control'
        p.measure[2][3] = 'copy_bit'
        p.measure[2][4] = 'control'
        p.measure[2][5] = 'control'
        p.measure[2][6] = 'copy_bit'
        p.measure[2][7] = 'copy_bit'
        p.measure[2][8] = 'copy_bit'
        p.measure[2][9] = 'copy_bit'
        p.measure[2][10] = 'copy_bit'
        p.measure[2][11] = 'copy_bit'
        p.measure[2][12] = 'copy_bit'
        p.measure[2][13] = 'copy_bit'
        p.measure[2][14] = 'copy_bit'
    elif TARGET == 'fredkin':
        p.measure[0][0] = 'control'
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
        p.measure[0][11] = 'control'
        p.measure[0][12] = 'copy_bit'
        p.measure[0][13] = 'copy_bit'
        p.measure[0][14] = 'copy_bit'

        p.measure[1][0] = 'cx'
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
        p.measure[1][11] = 'cx'
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
    elif TARGET == 'or':
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
        p.measure[2][2] = 'tdg'
        p.measure[2][3] = 'control'
        p.measure[2][4] = 'cx'
        p.measure[2][5] = 'copy_bit'
        p.measure[2][6] = 'copy_bit'
        p.measure[2][7] = 't'
        p.measure[2][8] = 'cx'
        p.measure[2][9] = 'control'
        p.measure[2][10] = 'copy_bit'
        p.measure[2][11] = 'copy_bit'
        p.measure[2][12] = 'copy_bit'
        p.measure[2][13] = 'copy_bit'
        p.measure[2][14] = 'copy_bit'
    elif TARGET == 'toffoli with a 0-control':
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
        # p.measure[0][13] = 'copy_bit'
        # p.measure[0][14] = 'copy_bit'

        p.measure[1][0] = 'copy_bit'
        p.measure[1][1] = 'copy_bit'
        p.measure[1][2] = 'tdg'
        p.measure[1][3] = 'control'
        p.measure[1][4] = 'cx'
        p.measure[1][5] = 't'
        p.measure[1][6] = 'copy_bit'
        p.measure[1][7] = 'cx'
        p.measure[1][8] = 'control'
        p.measure[1][9] = 'copy_bit'
        p.measure[1][10] = 'copy_bit'
        p.measure[1][11] = 'copy_bit'
        p.measure[1][12] = 'copy_bit'
        # p.measure[1][13] = 'copy_bit'
        # p.measure[1][14] = 'copy_bit'

        p.measure[2][0] = 't'
        p.measure[2][1] = 'cx'
        p.measure[2][2] = 'tdg'
        p.measure[2][3] = 'cx'
        p.measure[2][4] = 'copy_bit'
        p.measure[2][5] = 't'
        p.measure[2][6] = 'cx'
        p.measure[2][7] = 'tdg'
        p.measure[2][8] = 'cx'
        p.measure[2][9] = 'copy_bit'
        p.measure[2][10] = 'copy_bit'
        p.measure[2][11] = 'copy_bit'
        p.measure[2][12] = 'copy_bit'
        # p.measure[2][13] = 'copy_bit'
        # p.measure[2][14] = 'copy_bit'


if __name__ == '__main__':
    random.seed(rand_seed)
    target_op = target_gate()  # get target operator(matrix)
    target = Circuit()
    expBest = Circuit()
    expBest.fit = float('-inf')
    x = circuit_list()  # generate circuit (null)

    for expTime in range(test):

        # ↓ initialize ↓ #
        delta = DELTA  # init delta value
        Q = np.full((n, depth, len(GATE_SET)), 1 / 6)  # init prob. matrix Q
        last_HD = float('inf')
        gb = Circuit()
        gw = Circuit()
        lb = Circuit()
        lw = Circuit()
        gb.fit = float('-inf')
        gw.fit = float('inf')
        # ↑ initialize ↑ #

        print("__________{} Exp {} __________".format("KNQTS", expTime))

        for ecx in range(loop):
            if ecx % 100 == 0:
                print("__________{} Exp {} iteration {} __________".format(
                    "KNQTS", expTime, ecx))
                print("global best")
                gb.print()

            initx(x, expTime, ecx)
            measure(x)  # built circuit
            repair(x)

            if ecx == 0:
                presetAns(x[0])
                target.copy(x[0])

            fitness(x, target_op)
            last_HD = update(x, gb, gw, lb, lw, ecx, expTime, last_HD, delta)

        if gb.fit > expBest.fit:
            expBest.copy(gb)

    print("__________Finish__________")
    expBest.print()
    expBest.circuit.draw('mpl', filename="KNQTS" +
                         str(expBest.fit) + "_FinalCircuit.png")
    print(expBest.op)
