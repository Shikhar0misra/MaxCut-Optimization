import time

import numpy as np

from qiskit import QuantumCircuit
from qiskit.algorithms.optimizers import COBYLA
from qiskit.quantum_info import Statevector

from algorithms import compute_cut
from quantum_vqe import improve_partition
from quantum_runtime import estimate_quantum_execution_time


def build_qaoa_circuit(G, gammas, betas):

    n = len(G.nodes())
    circuit = QuantumCircuit(n)

    for qubit in range(n):
        circuit.h(qubit)

    for gamma, beta in zip(gammas, betas):
        for u, v in G.edges():
            circuit.rzz(-gamma, u, v)

        for qubit in range(n):
            circuit.rx(2 * beta, qubit)

    return circuit


def expected_cut(G, params, reps):

    gammas = params[:reps]
    betas = params[reps:]
    circuit = build_qaoa_circuit(G, gammas, betas)
    state = Statevector.from_instruction(circuit)

    expectation = 0

    for index, probability in enumerate(state.probabilities()):
        bitstring = format(index, f"0{len(G.nodes())}b")
        partition = [int(bit) for bit in reversed(bitstring)]
        expectation += probability * compute_cut(G, partition)

    return expectation


def best_partition_from_qaoa(G, params, reps):

    gammas = params[:reps]
    betas = params[reps:]
    circuit = build_qaoa_circuit(G, gammas, betas)
    state = Statevector.from_instruction(circuit)

    best_index = np.argmax(state.probabilities())
    bitstring = format(best_index, f"0{len(G.nodes())}b")

    return [int(bit) for bit in reversed(bitstring)]


def run_qaoa(G, reps=1, maxiter=75, restarts=3):

    start = time.time()
    best_result = None
    optimizer = COBYLA(maxiter=maxiter)
    evaluations = 0

    for _ in range(restarts):
        initial_params = np.concatenate([
            np.random.uniform(0, np.pi, reps),
            np.random.uniform(0, np.pi / 2, reps),
        ])

        def objective(params):
            nonlocal evaluations
            evaluations += 1
            return -expected_cut(G, params, reps)

        result = optimizer.minimize(
            fun=objective,
            x0=initial_params,
        )

        if best_result is None or result.fun < best_result.fun:
            best_result = result

    raw_partition = best_partition_from_qaoa(G, best_result.x, reps)
    raw_cut = compute_cut(G, raw_partition)
    qaoa_cut, qaoa_partition = improve_partition(G, raw_partition)

    end = time.time()
    circuit = build_qaoa_circuit(G, best_result.x[:reps], best_result.x[reps:])
    quantum_time = estimate_quantum_execution_time(circuit, evaluations + 1)

    return qaoa_cut, qaoa_partition, end-start, quantum_time, raw_cut
