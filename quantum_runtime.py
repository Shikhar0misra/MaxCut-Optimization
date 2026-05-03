DEFAULT_SHOTS = 1024
DEFAULT_GATE_LAYER_TIME = 300e-9
DEFAULT_MEASUREMENT_TIME = 1e-6


def estimate_quantum_execution_time(
    circuit,
    evaluations,
    shots=DEFAULT_SHOTS,
    gate_layer_time=DEFAULT_GATE_LAYER_TIME,
    measurement_time=DEFAULT_MEASUREMENT_TIME,
):

    depth = max(circuit.decompose().depth(), 1)
    single_shot_time = depth * gate_layer_time + measurement_time

    return evaluations * shots * single_shot_time
