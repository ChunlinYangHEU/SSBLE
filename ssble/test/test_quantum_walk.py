import numpy as np
from ssble import tools, qgates, anglecompute, blockencoding
from pyqpanda import *


def test_quantum_walk(order, dim, probability):
    """
    Tests the construction of a quantum walk Hamiltonian using block encoding.

    Args:
        order (int): The order of the quantum walk (number of steps).
        dim (int): The dimension of the Hilbert space for each step (must be a power of 2).
        probability (list or numpy.ndarray): The probability distribution for the quantum walk.

    Returns:
        None: The function prints the constructed quantum circuit and the encoded Hamiltonian matrix.
    """
    dim_log2 = int(np.log2(dim))
    num_ancillary_qubits = int(np.ceil(np.log2(len(probability))))
    num_working_qubits = order * dim_log2

    data_value = np.pad(probability, (0, 2 ** num_ancillary_qubits - len(probability)), mode='constant')
    subnormalization = np.sum(np.abs(data_value))
    data_value_PREP = np.sqrt(data_value) / np.sqrt(subnormalization)
    data_values_PREP_norm = np.abs(data_value_PREP)
    data_values_PREP_phase = np.angle(data_value_PREP)
    norm_angles = anglecompute.binarytree_vector(data_values_PREP_norm, 'norm')
    phase_angles = anglecompute.binarytree_vector(data_values_PREP_phase, 'phase')

    num_qubits = num_ancillary_qubits + num_working_qubits
    init(QMachineType.CPU)
    qubits = qAlloc_many(num_qubits)
    cbits = cAlloc_many(num_qubits)

    circuit = QCircuit()

    circuit_PREP = blockencoding.oracle_PREP(target_qubits=qubits[: num_ancillary_qubits],
                                             norm_angles=norm_angles,
                                             phase_angles=phase_angles)
    circuit << circuit_PREP

    for layer in range(order):
        circuit << qgates.right_shift(
            target_qubits=qubits[num_ancillary_qubits + layer * dim_log2: num_ancillary_qubits + (layer + 1) * dim_log2],
            control_qubits=qubits[:num_ancillary_qubits],
            control_states=tools.binary_list(2 * layer, num_ancillary_qubits))
        circuit << qgates.left_shift(
            target_qubits=qubits[num_ancillary_qubits + layer * dim_log2: num_ancillary_qubits + (layer + 1) * dim_log2],
            control_qubits=qubits[:num_ancillary_qubits],
            control_states=tools.binary_list(2 * layer + 1, num_ancillary_qubits))

    circuit_UNPREP = blockencoding.oracle_PREP(target_qubits=qubits[: num_ancillary_qubits],
                                               norm_angles=norm_angles,
                                               phase_angles=-phase_angles)
    circuit_UNPREP = circuit_UNPREP.dagger()
    circuit << circuit_UNPREP

    print(circuit)

    unitary = get_unitary(circuit)
    unitary = np.array(unitary).reshape(2 ** num_qubits, 2 ** num_qubits)
    res_matrix = subnormalization * blockencoding.get_encoded_matrix(unitary, num_working_qubits)
    print('The encoded Hamiltonian:')
    print(res_matrix)


if __name__ == '__main__':
    order = 2
    dim = 4
    probability = np.random.uniform(-1, 1, size=order) + np.random.uniform(-1, 1, size=order) * 1j
    probability = np.ravel(np.column_stack((probability, np.conj(probability))))
    test_quantum_walk(order, dim, probability)
