import numpy as np
import inspect
import time

from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import TwoLocal
from algorithms import compute_cut
from quantum_runtime import estimate_quantum_execution_time

try:
    from qiskit_algorithms import VQE
    from qiskit_algorithms.optimizers import COBYLA
except ImportError:
    from qiskit.algorithms import VQE
    from qiskit.algorithms.optimizers import COBYLA

try:
    from qiskit.primitives import StatevectorEstimator as Estimator
except ImportError:
    try:
        from qiskit.primitives import Estimator
    except ImportError:
        Estimator = None

try:
    from qiskit import Aer
except ImportError:
    try:
        from qiskit import BasicAer as Aer
    except ImportError:
        Aer = None

try:
    from qiskit.utils import QuantumInstance
except ImportError:
    QuantumInstance = None

try:
    from qiskit.opflow import PauliSumOp
except ImportError:
    PauliSumOp = None


def build_maxcut_terms(G):

    n = len(G.nodes())
    pauli_list = []

    for i, j in G.edges():

        z = ['I'] * n
        z[i] = 'Z'
        z[j] = 'Z'

        pauli_str = "".join(z)
        pauli_list.append((pauli_str, 0.5))

    return pauli_list


def build_maxcut_hamiltonian(G, use_opflow=False):

    pauli_list = build_maxcut_terms(G)

    if use_opflow:
        if PauliSumOp is None:
            raise ImportError("qiskit.opflow.PauliSumOp is required for this VQE version")

        return PauliSumOp.from_list(pauli_list)

    return SparsePauliOp.from_list(pauli_list)


def get_solution(ansatz, optimal_params):

    qc = ansatz.assign_parameters(optimal_params)
    state = Statevector.from_instruction(qc)

    probs = state.probabilities()

    best_index = np.argmax(probs)
    bitstring = format(best_index, f'0{qc.num_qubits}b')

    return [int(bit) for bit in reversed(bitstring)]


def improve_partition(G, partition):

    partition = list(partition)
    best_cut = compute_cut(G, partition)
    improved = True

    while improved:
        improved = False

        for node in range(len(partition)):
            partition[node] = 1 - partition[node]
            candidate_cut = compute_cut(G, partition)

            if candidate_cut > best_cut:
                best_cut = candidate_cut
                improved = True
            else:
                partition[node] = 1 - partition[node]

    return best_cut, partition


def run_vqe(G, reps=1, maxiter=75):

    n = len(G.nodes())
    start = time.time()

    ansatz = TwoLocal(n, ['ry', 'rz'], 'cz', reps=reps)

    optimizer = COBYLA(maxiter=maxiter)

    vqe_params = inspect.signature(VQE).parameters

    if "estimator" in vqe_params:
        H = build_maxcut_hamiltonian(G)
        estimator = Estimator()
        vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)
    else:
        H = build_maxcut_hamiltonian(G, use_opflow=True)

        if Aer is None:
            raise ImportError("Qiskit Aer or BasicAer is required for this VQE version")

        backend = Aer.get_backend("statevector_simulator")

        if "quantum_instance" in vqe_params and QuantumInstance is not None:
            quantum_instance = QuantumInstance(backend)
            vqe = VQE(
                ansatz=ansatz,
                optimizer=optimizer,
                quantum_instance=quantum_instance
            )
        else:
            vqe = VQE(ansatz=ansatz, optimizer=optimizer, quantum_instance=backend)

    result = vqe.compute_minimum_eigenvalue(H)
    evaluations = getattr(result, "cost_function_evals", None) or maxiter

    energy = result.eigenvalue.real

    solution = get_solution(ansatz, result.optimal_point)
    vqe_cut = compute_cut(G, solution)
    improved_cut, improved_solution = improve_partition(G, solution)

    end = time.time()
    quantum_time = estimate_quantum_execution_time(ansatz, evaluations + 1)

    return energy, improved_solution, end-start, quantum_time, vqe_cut, improved_cut
