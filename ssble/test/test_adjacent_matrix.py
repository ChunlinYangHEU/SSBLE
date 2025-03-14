import numpy as np
from ssble import blockencoding
from pyqpanda import *


def test_adjacent_matrix(alpha, beta, gamma, dim):
    """
    Tests the construction of an encoded adjacency matrix using block encoding.

    Args:
        alpha (float): The first element of the adjacency matrix.
        beta (float): The second element of the adjacency matrix.
        gamma (float): The third element of the adjacency matrix.
        dim (int): The dimension of the adjacency matrix (must be a power of 2).

    Returns:
        None: The function prints the constructed quantum circuit and the encoded adjacency matrix.x.
    """
    data_value = np.array([alpha, beta, gamma])
    data_function = [0, 1, -1]
    defining_domain = [None, None, None]
    data_item = {'data_value': data_value, 'data_function': data_function, 'defining_domain': defining_domain}
    subnormalization = np.sum(np.abs(data_value))

    num_working_qubits = int(np.log2(dim))
    num_ancillary_qubits = int(np.ceil(np.log2(len(data_value)))) + 1
    num_qubits = num_working_qubits + num_ancillary_qubits
    init(QMachineType.CPU)
    qubits = qAlloc_many(num_qubits)
    cbits = cAlloc_many(num_qubits)

    circuit = blockencoding.qcircuit(data_item=data_item,
                                     ancillary_qubits=qubits[:num_ancillary_qubits],
                                     working_qubits=qubits[num_ancillary_qubits:],
                                     is_real=True)
    print(circuit)

    unitary = get_unitary(circuit)
    unitary = np.array(unitary).reshape(2 ** num_qubits, 2 ** num_qubits)
    res_matrix = subnormalization * blockencoding.get_encoded_matrix(unitary, num_working_qubits)
    print('The encoded adjacency matrix:')
    print(res_matrix)


if __name__ == '__main__':
    alpha = 1
    beta = 2
    gamma = 3
    dim = 8

    test_adjacent_matrix(alpha, beta, gamma, dim)