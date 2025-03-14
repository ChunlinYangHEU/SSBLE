"""
Sparse Structured BLock Encoding:
      \ket{0}_{del} ——————————————⨁———————————————————
                                  |
\ket{0}^{\otimes m} —\——|PREP|———/ \————⊘————|UNPREP|——
                                |Org|   |
            \ket{j} —\———————————\ /———|Oc|————————————

PREP \ket{0}^{\otimes m} = (1/\sqrt{\sum_{l}|A_{l}|}) (\sum_{l=0}^{s_{0}-1}\sqrt{A_{l}}\ket{l} + \sum_{s_{0}}^{s-1}0\ket{l})
UNPREP \ket{0}^{\otimes m} = (1/\sqrt{\sum_{l}|A_{l}|}) (\sum_{l=0}^{s_{0}-1}\sqrt{A_{l}^{*}}\ket{l} + \sum_{s_{0}}^{s-1}0\ket{l})
"""
import numpy as np
from pyqpanda import *
from ssble import tools, qgates, anglecompute


def qcircuit(data_item, ancillary_qubits, working_qubits, control_qubits=None, control_states=None, is_real=False):
    """
    Constructs the sparse structured quantum circuit for a sparse structured matrix based on the provided data item.

    Args:
        data_item (dict): A dictionary containing the data information.
            - 'data_value': The data values to be processed.
            - 'data_function': The function associated with the data.
            - 'defining_domain': The domain of the data function.
        ancillary_qubits (list[Qubit]): A list of ancillary qubits used in the circuit.
        working_qubits (list[Qubit]): A list of working qubits used in the circuit.
        control_qubits (list[Qubit] or Qubit, optional): A list of control qubits. Defaults to None.
        control_states (list[int] or int, optional): A list of states for the control qubits.
            Defaults to None.
        is_real (bool, optional): Indicates whether the data is real. Defaults to False.

    Returns:
        QCircuit: The constructed quantum circuit.
    """
    circuit = QCircuit()
    num_ancillary_qubits = len(ancillary_qubits)

    data_value = data_item['data_value']
    data_function = data_item['data_function']
    defining_domain = data_item['defining_domain']

    subnormalization = np.sum(np.abs(data_value))
    data_value = np.pad(data_value, (0, 2 ** (num_ancillary_qubits - 1) - len(data_value)), mode='constant')
    data_value_PREP = np.sqrt(data_value) / np.sqrt(subnormalization)
    if is_real:
        norm_angles = anglecompute.binarytree_vector(data_value_PREP, 'norm', True)
    else:
        data_values_PREP_norm = np.abs(data_value_PREP)
        data_values_PREP_phase = np.angle(data_value_PREP)
        norm_angles = anglecompute.binarytree_vector(data_values_PREP_norm, 'norm')
        phase_angles = anglecompute.binarytree_vector(data_values_PREP_phase, 'phase')

    # The oracle PREP
    if is_real:
        circuit_PREP = oracle_PREP(target_qubits=ancillary_qubits[1:],
                                   norm_angles=norm_angles)
    else:
        circuit_PREP = oracle_PREP(target_qubits=ancillary_qubits[1:],
                                   norm_angles=norm_angles,
                                   phase_angles=phase_angles)
    circuit << circuit_PREP

    # The oracle Org
    circuit_Org = oracle_Org(control_qubits=ancillary_qubits[1:] + working_qubits,
                             target_qubit=ancillary_qubits[0],
                             defining_domain=defining_domain)
    circuit << circuit_Org

    # The oracle Oc
    circuit_Oc = oracle_Oc(control_qubits=ancillary_qubits[1:],
                           target_qubits=working_qubits,
                           data_function=data_function)
    circuit << circuit_Oc

    # The oracle UNPREP
    if is_real:
        circuit_UNPREP = circuit_PREP.dagger()
    else:
        circuit_UNPREP = oracle_PREP(target_qubits=ancillary_qubits[1:],
                                     norm_angles=norm_angles,
                                     phase_angles=-phase_angles)
        circuit_UNPREP = circuit_UNPREP.dagger()
    circuit << circuit_UNPREP

    # Delete two X gates if they are adjacent
    query_cir = QCircuit()
    query_cir << X(ancillary_qubits[0]) << X(ancillary_qubits[0])
    replace_cir = QCircuit()
    replace_cir << I(ancillary_qubits[0])
    circuit = circuit_optimizer(circuit, [[query_cir, replace_cir]])

    if control_qubits is not None:
        if not isinstance(control_qubits, list):
            control_qubits = [control_qubits]
        if not isinstance(control_states, list):
            control_states = [control_states]
        gate_ctrl_qubits_state0 = [control_qubits[index] for index, state in enumerate(control_states) if state == 0]
        circ = QCircuit()
        for qubit in gate_ctrl_qubits_state0:
            circ << X(qubit)
        circ << circuit.control(control_qubits)
        for qubit in gate_ctrl_qubits_state0:
            circ << X(qubit)
        return circ
    else:
        return circuit


def oracle_PREP(target_qubits, norm_angles, phase_angles=None):
    """
    Constructs the PREP oracle in the quantum circuit of sparse structured block encoding.

    Args:
        target_qubits (list[Qubit] or Qubit): A list of target qubits for state preparation.
        norm_angles (numpy.array): A 1D array of norm angles for state preparation.
        phase_angles (numpy.array, optional): A 1D array of phase angles for state preparation.
            Defaults to None.

    Returns:
        QCircuit: The constructed PREP oracle circuit.
    """
    circuit_PREP = QCircuit()

    num_qubits = len(target_qubits)
    norm_angles = norm_angles.flatten()
    if phase_angles is not None:
        phase_angles = phase_angles.flatten()

    if phase_angles is not None:
        qgates.qgate('RZ',
                     circuit_PREP,
                     target_qubit=target_qubits[0],
                     rotation_angle=phase_angles[0])

    qgates.qgate('RY',
                 circuit_PREP,
                 target_qubit=target_qubits[0],
                 rotation_angle=norm_angles[0])
    angle_index = 1
    for layer in range(1, num_qubits):
        ur_angles = anglecompute.uniformly_rotation_angles(norm_angles[angle_index: angle_index + 2 ** layer])
        circuit_ur = qgates.uniformly_rotation('RY',
                                               target_qubit=target_qubits[layer],
                                               control_qubits=target_qubits[:layer],
                                               rotation_angles=ur_angles)
        circuit_PREP << circuit_ur
        angle_index += 2 ** layer

    if phase_angles is not None:
        qgates.qgate('RZ',
                     circuit_PREP,
                     target_qubit=target_qubits[0],
                     rotation_angle=phase_angles[1])
        angle_index = 2
        for layer in range(1, num_qubits):
            ur_angles = anglecompute.uniformly_rotation_angles(phase_angles[angle_index: angle_index + 2 ** layer])
            circuit_ur = qgates.uniformly_rotation('RZ',
                                                   target_qubit=target_qubits[layer],
                                                   control_qubits=target_qubits[:layer],
                                                   rotation_angles=ur_angles)
            circuit_PREP << circuit_ur
            angle_index += 2 ** layer

    return circuit_PREP


def oracle_Org(control_qubits, target_qubit, defining_domain):
    """
    Constructs the Org oracle in the quantum circuit of sparse structured block encoding.

    Args:
        control_qubits (list[Qubit]): A list of control qubits used in the Org oracle.
        target_qubit (Qubit): The target qubit to encode the defining domain.
        defining_domain (list): The defining domain in data item.

    Returns:
        QCircuit: The constructed Org oracle circuit.
    """
    circuit_Org = QCircuit()

    if not all(x is None for x in defining_domain):
        num_index_qubits = int(np.ceil(np.log2(len(defining_domain))))

        # Create two lists to save control qubits and control states of MC-NOT gates
        ctrl_qubits_list = []
        ctrl_states_list = []

        # Get MC-NOT gates according to "defining_domain"
        for data_index, domain in enumerate(defining_domain):
            if domain is not None:
                for qbits_states in domain:
                    qbits = list(range(num_index_qubits))
                    states = tools.binary_list(data_index, num_index_qubits)
                    for qbit, state in enumerate(qbits_states):
                        if state is not None:
                            qbits.append(num_index_qubits + qbit)
                            states.append(state)
                    ctrl_qubits_list.append(qbits)
                    ctrl_states_list.append(states)

        # Simplify MC-NOT gates
        ctrl_qubits_list, ctrl_states_list = qgates.mcnots_simplify(ctrl_qubits_list, ctrl_states_list)
        print(ctrl_qubits_list)
        print(ctrl_states_list)

        # Perform MC-NOT gates
        for (ctrl_qubits, ctrl_states) in zip(ctrl_qubits_list, ctrl_states_list):
            qgates.qgate('X',
                         circuit_Org,
                         target_qubit=target_qubit,
                         control_qubits=[control_qubits[i] for i in ctrl_qubits],
                         control_states=ctrl_states)

        # Perform X gate to flip the defining domains to out-of-range domains

        circuit_Org << X(target_qubit)

    return circuit_Org


def oracle_Oc(control_qubits, target_qubits, data_function):
    """
   Constructs the Oc oracle in the quantum circuit of sparse structured block encoding.

   Args:
       control_qubits (list[Qubit] or Qubit): A list of control qubits used in the Oc oracle.
       target_qubits (list[Qubit] or Qubit): A list of target qubits to encode the data function.
       data_function (list[int]): A list of integers representing the data function.

   Returns:
       QCircuit: The constructed Oc oracle circuit.
   """
    circuit_Oc = QCircuit()

    if not all(x == 0 for x in data_function):
        num_control_qubits = len(control_qubits)
        num_target_qubits = len(target_qubits)

        # Create empty lists to save the left and right shift gates in the Oc oracle
        lk_target_qubits_list = []
        lk_control_qubits_list = []
        lk_control_states_list = []
        rk_target_qubits_list = []
        rk_control_qubits_list = []
        rk_control_states_list = []

        for data_index, func in enumerate(data_function):
            if func != 0:
                func_binary = bin(np.abs(func))[2:].zfill(num_target_qubits)
                for bit_index, bit in enumerate(func_binary):
                    if bit == '1':
                        trgt_qubits = list(range(bit_index + 1))
                        ctrl_qubits = list(range(num_control_qubits))
                        ctrl_states = tools.binary_list(data_index, num_control_qubits)
                        if func > 0:
                            lk_target_qubits_list.append(trgt_qubits)
                            lk_control_qubits_list.append(ctrl_qubits)
                            lk_control_states_list.append(ctrl_states)
                        elif func < 0:
                            rk_target_qubits_list.append(trgt_qubits)
                            rk_control_qubits_list.append(ctrl_qubits)
                            rk_control_states_list.append(ctrl_states)

        # # Simplify left and right shift gates
        (lk_target_qubits_list, lk_control_qubits_list, lk_control_states_list,
         rk_target_qubits_list, rk_control_qubits_list, rk_control_states_list) \
            = qgates.shift_gates_simplify1(
            lk_target_qubits_list, lk_control_qubits_list, lk_control_states_list,
            rk_target_qubits_list, rk_control_qubits_list, rk_control_states_list)

        # Perform left and right shift gates
        for lk_index, lk_trgt_qubits in enumerate(lk_target_qubits_list):
            circuit_Oc << qgates.left_shift(target_qubits=[target_qubits[i] for i in lk_trgt_qubits],
                                            control_qubits=[control_qubits[i] for i in lk_control_qubits_list[lk_index]],
                                            control_states=lk_control_states_list[lk_index])
        for rk_index, rk_trgt_qubits in enumerate(rk_target_qubits_list):
            circuit_Oc << qgates.right_shift(target_qubits=[target_qubits[i] for i in rk_trgt_qubits],
                                             control_qubits=[control_qubits[i] for i in rk_control_qubits_list[rk_index]],
                                             control_states=rk_control_states_list[rk_index])

    return circuit_Oc


def get_encoded_matrix(unitary, num_working_qubits):
    """
    Extracts the block-encoded matrix from a given unitary matrix.

    Args:
        unitary (numpy.array): The unitary matrix representing the quantum circuit.
        num_working_qubits (int): The number of working qubits used in the encoding.

    Returns:
        numpy.array: The block-encoded matrix.
    """
    unitary = tools.reverse_index_bits(unitary)
    matrix = unitary[: 2 ** num_working_qubits, : 2 ** num_working_qubits]

    return matrix
