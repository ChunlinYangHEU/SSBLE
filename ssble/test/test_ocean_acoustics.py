import numpy as np
import pandas as pd
from pyqpanda import *
from ssble import tools, blockencoding
import time


def get_data_item(matrix, parameters):
    """
    Obtains the data item for block encoding of matrices A or B based on the given parameters.

    Args:
        matrix (str): The type of matrix to generate data for. Must be 'A' or 'B'.
        parameters (dict): A dictionary containing the physical parameters of the problem.
            - 'depth_ice': The total depth of the ice layer.
            - 'num_ice': The number of grid points in the ice layer.
            - 'depth_seawater': The total depth of the seawater layer.
            - 'num_seawater': The number of grid points in the seawater layer.
            - 'density': A function representing the density at depth z.
            - 'sound_velocity': A function representing the sound velocity at depth z.
            - 'frequency': The frequency of the acoustic wave.
            - 'mu_ice': A function representing the Lamé coefficient μ of ice at depth z.
            - 'lambda_ice': A function representing the Lamé coefficient λ of ice at depth z.

    Returns:
        dict: A dictionary containing the data item for block encoding.
            - 'data_value': The array of data values.
            - 'data_function': The array of data functions.
            - 'defining_domain': The list of defining domains.

    Raises:
        ValueError: If the matrix type is not 'A' or 'B'.
    """
    # Get parameters
    depth_ice = parameters['depth_ice']  # The depth of ice
    num_ice = parameters['num_ice']  # The number of grids
    delta_ice = depth_ice / num_ice  # The depth of each grid
    depth_seawater = parameters['depth_seawater']  # The depth of seawater
    num_seawater = parameters['num_seawater']  # The number of grids
    delta_seawater = depth_seawater / num_seawater  # The depth of each grid
    density = parameters['density']  # Density at depth z
    sound_velocity = parameters['sound_velocity']  # Sound velocity at depth z
    frequency = parameters['frequency']  # Frequency of sound wave
    angle_frequency = 2 * np.pi * frequency  # Angular frequency of sound wave
    mu_ice = parameters['mu_ice']  # Lamé coefficients of ice
    lambda_ice = parameters['lambda_ice']  # Lamé coefficients of ice

    # Related parameters
    x1 = lambda z: 1 / mu_ice(z)
    x2 = lambda z: 1 / (lambda_ice(z) + 2 * mu_ice(z))
    x3 = lambda z: lambda_ice(z) / (lambda_ice(z) + 2 * mu_ice(z))
    x4 = lambda z: (4 * mu_ice(z) * (lambda_ice(z) + mu_ice(z))) / (lambda_ice(z) + 2 * mu_ice(z))
    x5 = lambda z: - density(z) * (angle_frequency ** 2)

    # Dimension of the matrix
    dim = 4 * num_ice + num_seawater + 5
    n = int(np.ceil(np.log2(dim)))

    if matrix == 'A':
        data_value = np.array(
            [x2(0), -x3(0), -1, -1, x1(0), x1(0), 1, 1, x5(0), x5(0), 2 / delta_ice, -2 / delta_ice,
             -2 + delta_seawater ** 2 * angle_frequency ** 2 / sound_velocity(delta_seawater) ** 2,
             delta_seawater ** 2 * angle_frequency ** 2 / (2 * sound_velocity(depth_ice) ** 2) - 1,
             1 / delta_seawater, 1, x2(0), -x3(0), 1, -delta_ice * density(depth_ice) * angle_frequency ** 2,
             -1 / delta_seawater + delta_seawater * angle_frequency ** 2 / (
                     2 * sound_velocity(depth_ice + depth_seawater) ** 2)],
            dtype=complex)
        data_function = [-4, -3, 1, -3, 0, -4, 1, -1, 4, 0, 2, -2, 0, 0, 1, -2, 0, 1, 0, 3, 0]
        defining_domain = [tools.binary_range(7, 4 * num_ice + 3, n, True, True, 3),
                           tools.binary_range(7, 4 * num_ice + 3, n, True, True, 3),
                           tools.binary_range0(4 * num_ice - 3, n, True, 1),
                           tools.binary_range(5, 4 * num_ice + 1, n, True, True, 1),
                           tools.binary_range0(4 * num_ice - 2, n, True, 2),
                           tools.binary_range(6, 4 * num_ice + 2, n, True,  True, 2),
                           tools.binary_range(4 * num_ice + 4, dim - 3, n, True, True),
                           tools.binary_range(4 * num_ice + 4, dim - 1, n, True, True),
                           tools.binary_range0(4 * num_ice - 3, n, True, 0)
                           + tools.binary_range0(4 * num_ice - 3, n, True, 1),
                           tools.binary_range(4, 4 * num_ice + 1, n, True, True, 0)
                           + tools.binary_range(4, 4 * num_ice + 1, n, True, True, 1),
                           tools.binary_range0(4 * num_ice - 1, n, True),
                           tools.binary_range(4, 4 * num_ice + 3, n, True, True),
                           tools.binary_range(4 * num_ice + 5, dim - 2, n, True, True),
                           [tools.binary_list(4 * num_ice + 4, n)],
                           [tools.binary_list(dim - 2, n)],
                           [tools.binary_list(2, n), tools.binary_list(3, n)],
                           tools.binary_range0(4 * num_ice - 1, n, True, 3),
                           tools.binary_range0(4 * num_ice - 1, n, True, 3),
                           [tools.binary_list(4 * num_ice + 2, n), tools.binary_list(4 * num_ice + 3, n)],
                           [tools.binary_list(4 * num_ice + 1, n)],
                           [tools.binary_list(dim - 1, n)]]
    elif matrix == 'B':
        data_value = np.array(
            [-x3(0), -x3(0), -1, -1, -x4(0), -x4(0), delta_seawater ** 2 / 2, delta_seawater ** 2, delta_seawater / 2],
            dtype=complex)
        data_function = [3, -1, 3, -1, 4, 0, 0, 0, 0]
        defining_domain = [tools.binary_range0(4 * (num_ice - 1), n, True, 0),
                           tools.binary_range(4, 4 * num_ice, n, True, True, 0),
                           tools.binary_range(2, 4 * num_ice - 2, n, True, True, 2),
                           tools.binary_range(6, 4 * num_ice + 2, n, True, True, 2),
                           tools.binary_range0(4 * (num_ice - 1), n, True, 0),
                           tools.binary_range(4, 4 * num_ice, n, True, True, 0),
                           [tools.binary_list(4 * num_ice + 4, n)],
                           tools.binary_range(4 * num_ice + 5, 4 * num_ice + num_seawater + 3, n, True, True),
                           [tools.binary_list(4 * num_ice + num_seawater + 4, n)]]
    else:
        raise ValueError('Matrix must be A or B')

    data_item = {'data_value': data_value, 'defining_domain': defining_domain, 'data_function': data_function}

    return data_item


def test_ocean_acoustics(matrix, paremeters):
    """
    Tests the block encoding of matrices related to generalized eigenvalue problems in ocean acoustics.

    Args:
        matrix (str): The type of matrix to encode. Must be 'A' or 'B'.
        parameters (dict): A dictionary containing the parameters for the ocean acoustics problem.
            - 'num_ice': The number of grid points in the ice layer.
            - 'num_seawater': The number of grid points in the seawater layer.
            - Other physical parameters (e.g., density, sound velocity) required by `get_data_item`.

    Returns:
        None: The function prints the constructed circuit, encoded matrix, and
              saves the matrices to Excel files.

    Raises:
        ValueError: If the matrix type is not 'A' or 'B'.
    """
    dim = 4 * paremeters['num_ice'] + paremeters['num_seawater'] + 5
    data_item = get_data_item(matrix, parameters)

    num_ancillary_qubits = int(np.ceil(np.log2(len(data_item['data_value'])))) + 1
    num_working_qubits = int(np.log2(dim))
    num_qubits = num_ancillary_qubits + num_working_qubits
    init(QMachineType.CPU)
    qubits = qAlloc_many(num_qubits)
    cbits = cAlloc_many(num_qubits)

    circuit = blockencoding.qcircuit(data_item=data_item,
                                     ancillary_qubits=qubits[:num_ancillary_qubits],
                                     working_qubits=qubits[num_ancillary_qubits:])
    print(circuit)
    # Get the encoded matrix
    print('--------Waiting for getting the unitary matrix of circuit--------')
    unitary = get_matrix(circuit)
    print('--------Get the unitary of circuit--------')
    unitary = np.array(unitary).reshape(2 ** num_qubits, 2 ** num_qubits)
    print('--------Successfully get the unitary of circuit--------')
    print('--------Waiting for getting the encoded matrix--------')
    subnormalization = np.sum(np.abs(data_item['data_value']))
    # res_matrix = subnormalization * blockencoding.get_encoded_matrix(unitary, num_working_qubits)
    res_matrix = subnormalization * unitary[:2 ** num_working_qubits, :2 ** num_working_qubits]
    print('--------Successfully get the encoded matrix--------')
    print(res_matrix)
    print('--------Waiting for writing the unitary matrix and the encoded matrix into excel--------')
    unitary_pd = pd.DataFrame(unitary)
    matrix_pd = pd.DataFrame(res_matrix)
    unitary_pd.to_excel(matrix + '_unitary.xlsx')
    matrix_pd.to_excel(matrix + '_matrix.xlsx')
    print('--------Successful writing --------')


if __name__ == '__main__':
    # The value of '4 * num_ice + num_seawater + 5' should be a power of two.
    depth_ice = 5                       # The depth of ice
    num_ice = 5                         # The number of grids
    depth_seawater = 7                  # The depth of seawater
    num_seawater = 7                    # The number of grids
    density = lambda z: 1.1             # Density at depth z
    sound_velocity = lambda z: 1900     # Sound velocity at depth z
    frequency = 1000                    # Frequency of sound wave
    lambda_ice = lambda z: 10           # Lamé coefficients of ice
    mu_ice = lambda z: 10               # Lamé coefficients of ice
    matrix = 'B'                        # The matrix to be encoded

    parameters = {'depth_ice': depth_ice, 'num_ice': num_ice,
                  'depth_seawater': depth_seawater, 'num_seawater': num_seawater,
                  'density': density, 'sound_velocity': sound_velocity, 'frequency': frequency,
                  'lambda_ice': lambda_ice, 'mu_ice': mu_ice}

    start_time = time.time()

    test_ocean_acoustics(matrix, parameters)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Running time：{elapsed_time} seconds")
