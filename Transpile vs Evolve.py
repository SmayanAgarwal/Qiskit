import numpy as np
import wfdb
import pandas as pd
import matplotlib.pyplot as plt
from qiskit.circuit.library.standard_gates import MCXGate
from qiskit.circuit.library import QFT
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator
import qiskit.quantum_info as qi

#This function simply converts a binary number to decimal
def bintodec(n):
    n = str(n)
    if (n == ''):
        return 0
    i = 0
    sum = int(0)
    while (i<len(n)):
        sum = sum + pow(2, len(n) - 1 - i)*int(n[i:i+1])
        i+=1
    return (sum)


n = 5 # N defines the number of data points we consider, we consider 2^N data points
l = 5 # L defines the precision with which we encode each data point. The higher L is, the higher will be the precision

#Here we create a data set which is a sin wave with some randomness added to it

time = np.arange(0, pow(2,n)) 
workingdata = (np.sin(time) + 1)/2 + (np.random.random(pow(2,n)))/2
#Here we normalise the data to fit between 0 and 1
datamax =  np.max(workingdata)
workingdata =  (workingdata/datamax)

#Binaried stores the data in binary form
# We scale each data point up by (2^L - 1) and represent by the closest whole number in binary
binaried = []
for i in workingdata:
    binaried.append(np.binary_repr(int(np.round(i*(pow(2,l)-1))), width = l))

q1=QuantumRegister(n,'time') # depends on data length
q2=QuantumRegister(l,'data') # depends on the bit representation of data..
#Q4 is meant to store the result of splitting the frequency spectrum
#In order to demonstrate the problem, we have kept the function without using its functionality
q4=QuantumRegister(1, 'comp')
circuit=QuantumCircuit(q2,q1,q4)
circuit.id(range(l))
circuit.h(range(l,n+l))

# Lis just tells the MCXGate what qubits to work on
lis = []
for t in range(n):
    lis.append(l+t)
lis.append('0')

# We only apply the MCXGate to the qubit line corresponding to a '1' in the binary representation of the data

for j in range(pow(2,n)):
    for i in range(l):
        if (binaried[j][i]=='1'):
            lis[n] = l-i-1           
            circuit.append(MCXGate(n,ctrl_state=np.binary_repr(j, width = n)), lis)
    circuit.barrier()
print ("Data Processed")
series = []

#Series tells the QFTs which qubits to work on
for i in range (n):
    series.append(l+i)
circuit.append(QFT(num_qubits=n, approximation_degree=0), qargs =series)

#The following MCX Gate is what causes the problem. Excluding it gives the expected result
circuit.append(MCXGate(2),[n+l-2,n+l-1,n+l])
circuit.append(QFT(num_qubits=n, approximation_degree=0, inverse=True), series)
circuit.barrier()
print ("Circuit Created")
circuit.draw('mpl', filename = 'Circuit')

#Zeroes simply defines the starting state for the evole circuit
zeroes = ""
for i in range(n+l+1):
    zeroes = zeroes + "0"

start = qi.Statevector.from_label(zeroes).evolve(circuit)
print ("Evolved")
afteriqft = start.probabilities_dict()
retrieved=np.zeros(pow(2,n))


for index in afteriqft.keys():  
    if (afteriqft[index] > 0.005):
        retrieved[bintodec(int(index[1:n+1]))] = bintodec(index[n+1:])

#The graph shows that the retreived result here is the same as the original
plt.figure(figsize=(40,6))
plt.plot(np.array(retrieved)/((2**l)-1))
plt.plot(workingdata[:pow(2,n)])
plt.legend('Retrieved','Original')
plt.savefig('Evolve')


circuit.measure_all()
simulator= AerSimulator()
compiled_circuit = transpile(circuit,simulator)
sim_result = simulator.run(compiled_circuit, shots = 1000)
counts = sim_result.result().get_counts(compiled_circuit)
retrieved=np.zeros(pow(2,n))


for index in counts.keys():
    retrieved[bintodec(int(index[1:n+1]))] = bintodec(index[n+1:])
plt.figure(figsize=(40,6))
plt.plot(np.array(retrieved)/((2**l)- 1))
plt.plot(workingdata[:pow(2,n)])
plt.legend('Retrieved','Original')
plt.savefig('Transpile')