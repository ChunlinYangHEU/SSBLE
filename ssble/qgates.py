import numpy as np
from pyqpanda import *
from ssble import tools


def qgate(gate, circuit, target_qubit, control_qubits=None, control_states=None, rotation_angle=None):
    """
    Simulate a quantum gate on a quantum circuit.

    Applies a quantum gate to a target qubit, optionally controlled by one or more control qubits.
    The gate is applied only when the control qubits are in the specified states.

    Args:
        gate (str): The type of quantum gate to apply ('X', 'Y', 'Z', 'H', 'SWAP', 'RX', 'RY', 'RZ').
        circuit (QCircuit): The quantum circuit to which the gate will be added.
        target_qubit (Qubit or list[Qubit, Qubit]): The target qubit(s) for the gate.
        control_qubits (Qubit or list[Qubit], optional): A single qubit or a list of control qubit(s) for the gate.
            Defaults to None.
        control_states (int or list[int], optional): A single state or a list of state(s) of the control qubit(s) that trigger the gate.
            Defaults to None.
        rotation_angle (float, optional): The angle of rotation for RX, RY, RZ gates. Defaults to None.

    Returns:
        None

    Raises:
        ValueError: If the specified gate is not supported.

    Note:
        The function assumes that the quantum circuit and qubits are properly initialized.
        The target_qubit and control_qubits must be of type pyqpanda.pyQPanda.Qubit or a list of such types.

    See Also:
        pyqpanda.Qubit: The class representing a quantum bit in pyqpanda.
    """
    # Perform gate if it has no control qubits
    if control_qubits is None or control_qubits == []:
        if gate == 'X':
            circuit << X(target_qubit)
        elif gate == 'Y':
            circuit << Y(target_qubit)
        elif gate == 'Z':
            circuit << Z(target_qubit)
        elif gate == 'H':
            circuit << H(target_qubit)
        elif gate == 'SWAP':
            circuit << SWAP(target_qubit[0], target_qubit[1])
        elif gate == 'RX':
            circuit << RX(target_qubit, rotation_angle)
        elif gate == 'RY':
            circuit << RY(target_qubit, rotation_angle)
        elif gate == 'RZ':
            circuit << RZ(target_qubit, rotation_angle)
        else:
            raise ValueError(gate + ' is not supported.')
    # Perform gate if it has control qubits
    else:
        # Ensure "control_qubits" and "control_states" be lists
        if not isinstance(control_qubits, list):
            control_qubits = [control_qubits]
        if not isinstance(control_states, list):
            control_states = [control_states]

        # Select control qubits with control states being 0
        qubits_state0 = [control_qubits[index] for index, state in enumerate(control_states) if state == 0]

        # Perform X gates on the control qubits with control states being 0
        for qubit in qubits_state0:
            circuit << X(qubit)

        # Perform controlled gates
        if gate == 'X':
            circuit << X(target_qubit).control(control_qubits)
        elif gate == 'Y':
            circuit << Y(target_qubit).control(control_qubits)
        elif gate == 'Z':
            circuit << Z(target_qubit).control(control_qubits)
        elif gate == 'H':
            circuit << H(target_qubit).control(control_qubits)
        elif gate == 'SWAP':
            circuit << SWAP(target_qubit[0], target_qubit[1]).control(control_qubits)
        elif gate == 'RX':
            circuit << RX(target_qubit, rotation_angle).control(control_qubits)
        elif gate == 'RY':
            circuit << RY(target_qubit, rotation_angle).control(control_qubits)
        elif gate == 'RZ':
            circuit << RZ(target_qubit, rotation_angle).control(control_qubits)
        else:
            raise ValueError(gate + ' is not supported')

        # Perform X gates on the control qubits with control states being 0
        for qubit in qubits_state0:
            circuit << X(qubit)


def left_shift(target_qubits, control_qubits=None, control_states=None):
    """
    Simulate a (controlled) left shift gate L^k on a quantum circuit.

    The left shift gate applies a phase shift to the target qubits, effectively moving the state of the qubits to the left.
    The operation is defined as (L^k ⊗ I^{⊗k}) |j⟩ = |(j+2^k) mod 2^n⟩, where L^k is the left shift gate, I is the identity gate,
    k is the number of qubits to shift, and n is the total number of qubits.

    Args:
        target_qubits (list[Qubit]): The target qubits to apply the right shift gate to.
        control_qubits (Qubit or list[Qubit], optional): A single qubit or a list of control qubit(s) for the gate. Defaults to None.
        control_states (int or list[int], optional): A single state or a list of state(s) of the control qubit(s) that trigger the gate. Defaults to None.

    Returns:
        QCircuit: The quantum circuit with the left shift gate applied.

    Raises:
        ValueError: If the control qubits and control states are not of the same length.

    See Also:
        pyqpanda.QCircuit: The class representing a quantum circuit in pyqpanda.
    """
    circuit = QCircuit()

    # Ensure "control_qubits" and "control_states" be lists
    if control_qubits is None:
        control_qubits = []
        control_states = []
    if not isinstance(control_qubits, list):
        control_qubits = [control_qubits]
    if not isinstance(control_states, list):
        control_states = [control_states]

    # Select control qubits with control states being 0
    ctrl_qubits_state0 = [control_qubits[index] for index, state in enumerate(control_states) if state == 0]

    length = len(target_qubits)

    # Perform X gates on the control qubits of left_shift gate with control state being 0
    for qubit in ctrl_qubits_state0:
        circuit << X(qubit)

    # Perform the MC-NOT and C-NOT gates of left_shift gate
    target_index = 0
    while target_index < length - 1:
        circuit << X(target_qubits[target_index]).control(target_qubits[target_index + 1: length] + control_qubits)
        target_index += 1

    # Perform the last X gate of left_shift gate
    if control_qubits is not None:
        circuit << X(target_qubits[-1]).control(control_qubits)

        # Perform X gates on the control qubits of left_shift gate with control state being 0
        for qubit in ctrl_qubits_state0:
            circuit << X(qubit)
    else:
        circuit << X(target_qubits[-1])

    return circuit


def right_shift(target_qubits, control_qubits=None, control_states=None):
    """
    Simulate a (controlled) right shift gate R^k on a quantum circuit.

    The right shift gate applies a phase shift to the target qubits, effectively moving the state of the qubits to the right.
    The operation is defined as (R^k ⊗ I^{⊗k}) |j⟩ = |(j-2^k) mod 2^n⟩, where R^k is the right shift gate, I is the identity gate,
    k is the number of qubits to shift, and n is the total number of qubits.

    Args:
        target_qubits (list[Qubit]): The target qubits to apply the right shift gate to.
        control_qubits (Qubit or list[Qubit], optional): A single qubit or a list of control qubit(s) for the gate. Defaults to None.
        control_states (int or list[int], optional): A single state or a list of state(s) of the control qubit(s) that trigger the gate. Defaults to None.

    Returns:
        QCircuit: The quantum circuit with the right shift gate applied.

    Raises:
        ValueError: If the control qubits and control states are not of the same length.
    """
    circuit = QCircuit()

    # Ensure "control_qubits" and "control_states" be lists
    if control_qubits is None:
        control_qubits = []
        control_states = []
    if not isinstance(control_qubits, list):
        control_qubits = [control_qubits]
    if not isinstance(control_states, list):
        control_states = [control_states]

    # Select control qubits with control states being 0
    ctrl_qubits_state0 = [control_qubits[index] for index, state in enumerate(control_states) if state == 0]

    length = len(target_qubits)

    # Perform X gates for control qubits in the 0-controlled MC-NOT gate of right_shift gate
    if length > 1:
        circuit << X(target_qubits[1: length])

    # Perform X gates on the control qubits of left_shift gate with control state being 0
    for qubit in ctrl_qubits_state0:
        circuit << X(qubit)

    # Perform the MC-NOT and C-NOT gates of left_shift gate
    target_index = 0
    while target_index < length - 1:
        circuit << X(target_qubits[target_index]).control(target_qubits[target_index + 1: length] + control_qubits)
        circuit << X(target_qubits[target_index + 1])
        target_index += 1

    # Perform the last X gate of left_shift gate
    if control_qubits is not None:
        circuit << X(target_qubits[-1]).control(control_qubits)

        # Perform X gates on the control qubits of left_shift gate with control state being 0
        for qubit in ctrl_qubits_state0:
            circuit << X(qubit)
    else:
        circuit << X(target_qubits[-1])

    return circuit


def uniformly_rotation(gate, target_qubit, control_qubits, rotation_angles, gate_ctrl_qubits=None,
                       gate_ctrl_states=None):
    """
    Generate a quantum circuit for the decomposition of a uniformly controlled rotation (RX, RY, or RZ).

    This function generate a quantum circuit that decomposes a uniformly controlled rotation
        into a sequence of single-qubit controlled rotation gates and C-NOT gates.
    The lengths of control_qubits and rotation_angles must match.

    Args:
        gate (str): The type of rotation gate, such as "Rx", "Ry", or "Rz".
        target_qubit (Qubit): The target qubit.
        control_qubits (Qubit or list[Qubit]): A single qubit or a list of qubit(s) for the control qubits.
        rotation_angles (float or list[float] or np.array): A single rotation angle or a list of rotation angles for rotation gate(s).
        gate_ctrl_qubits (Qubit or list[Qubit], optional): Additional control qubit(s) for the entire gate sequence. Defaults to None.
        gate_ctrl_states (int or list[int], optional): The state(s) of the additional control qubit(s) that trigger the gate sequence. Defaults to None.

    Returns:
        QCircuit: The constructed quantum circuit object.

    Raises:
        ValueError: If the gate type is not supported or if the lengths of control_qubits and rotation_angles do not match.
    """
    # Create a QCircuit object
    circuit = QCircuit()

    if not isinstance(control_qubits, list):
        control_qubits = [control_qubits]

    rotation_angles = rotation_angles.flatten()

    # Perform single qubit controlled rotation gates and CNOT gates
    for i in range(len(rotation_angles)):
        qgate(gate,
              circuit,
              target_qubit=target_qubit,
              rotation_angle=rotation_angles[i])
        # The control qubit of CNOT gate is up to the index of different bits between the gray codes of i and i+1
        # The control qubit of the last CNOT gate is the top one of "control_qubits"
        if i != len(rotation_angles) - 1:
            circuit << X(target_qubit).control(
                control_qubits[tools.different_gray_codes_index(i, i + 1, len(control_qubits))])
        else:
            circuit << X(target_qubit).control(control_qubits[0])

    if gate_ctrl_qubits is not None:
        if not isinstance(gate_ctrl_qubits, list):
            gate_ctrl_qubits = [gate_ctrl_qubits]
        if not isinstance(gate_ctrl_states, list):
            gate_ctrl_states = [gate_ctrl_states]
        gate_ctrl_qubits_state0 = [gate_ctrl_qubits[index] for index, state in enumerate(gate_ctrl_states) if
                                   state == 0]
        for qubit in gate_ctrl_qubits_state0:
            circuit << X(qubit)
        circuit = circuit.control(gate_ctrl_qubits)
        for qubit in gate_ctrl_qubits_state0:
            circuit << X(qubit)

    return circuit


def mcnots_simplify(control_qubits_list, control_states_list):
    """
    Simplify a sequence of MC-NOT gates with the same target qubit.

    This function simplifies a list of MC-NOT gates by removing redundant gates or modifying gates with differing control states.
    If two MC-NOT gates have the same control qubits, it either deletes the gates if their control states are the same,
        or deletes one gate and adjusts the other if they differ by only one control state.

    Args:
        control_qubits_list (list[list[int]]): A list of lists containing control qubits indices for each MC-NOT gate.
        control_states_list (list[list[int]]): A list of lists containing control states for each MC-NOT gate.

    Returns:
        tuple: A tuple containing two lists, the simplified control qubits list and the simplified control states list.

    Raises:
        ValueError: If the input lists are not of the same length.
    """
    # Create two empty lists to save the control qubits and control states of simplified MC-NOT gates
    control_qubits_list_simplify, control_states_list_simplify = [], []

    # A flag to check if input data are changed
    change = False

    for gate_index, gate_ctrl_qubits in enumerate(control_qubits_list):
        # Compare the control qubits of MC-NOT gates in "control_qubits_list" with those in "control_qubits_list_simplify"
        # to check if there exist two MC-NOT gates have the same control qubits
        if gate_ctrl_qubits in control_qubits_list_simplify:
            # Get all indices of the control qubits in "control_qubits_list_simplify" which are the same as "gate_ctrl_qubits"
            indices = [index for index, control_qubits in enumerate(control_qubits_list_simplify) if
                       control_qubits == gate_ctrl_qubits]

            # Create a dic to save the way to deal with the gate
            mode_deal = {}

            for index in indices:
                count_different_state = 0
                index_different_state = None

                # Compare the control states of the two MC-NOT gates
                for index_state, (state_i, state_j) in enumerate(
                        zip(control_states_list[gate_index], control_states_list_simplify[index])):
                    if state_i != state_j:
                        index_different_state = index_state
                        count_different_state += 1
                        if count_different_state > 1:
                            break

                # Save the way to deal with the gate according to "count_different_state"
                if count_different_state == 0:
                    mode_deal[0] = index
                    break
                elif count_different_state == 1:
                    mode_deal[1] = [index, index_different_state]
                elif count_different_state == 2:
                    mode_deal[2] = True

            # Deal with the gate according to "mode_deal"
            # 1) Delete the control qubits and control states of the MC-NOT gate in "control_qubits_list_simplify"
            # if the gate is exactly the same with the MC-NOT gate in "control_qubits_list"
            if 0 in mode_deal:
                control_qubits_list_simplify.pop(mode_deal[0])
                control_states_list_simplify.pop(mode_deal[0])

            # 2) Delete the control qubit and control state of the MC-NOT gate in "control_qubits_list_simplify"
            # if the gate has the same control qubits but differs only one control state with the MC-NOT gate in "control_qubits_list"
            elif 1 in mode_deal:
                del control_qubits_list_simplify[mode_deal[1][0]][mode_deal[1][1]]
                del control_states_list_simplify[mode_deal[1][0]][mode_deal[1][1]]
                change = True

            # 3) Add the control qubits and control states of the MC-NOT gate in "control_qubits_list"
            # if the gate has the same control qubits but differs more that one control state with the MC-NOT gate in "control_qubits_list"
            elif 2 in mode_deal:
                control_qubits_list_simplify.append(gate_ctrl_qubits)
                control_states_list_simplify.append(control_states_list[gate_index])

        # Add the control qubits and control states of the MC-NOT gae of the MC-NOT gate in "control_qubits_list"
        # if the control qubits of the gate are not exactly the same with those of MC-NOT gate in "control_qubits_list_simplify"
        else:
            control_qubits_list_simplify.append(gate_ctrl_qubits)
            control_states_list_simplify.append(control_states_list[gate_index])

    # Iterate if change is True
    if change:
        control_qubits_list_simplify, control_states_list_simplify = mcnots_simplify(control_qubits_list_simplify,
                                                                                     control_states_list_simplify)

    return control_qubits_list_simplify, control_states_list_simplify


def shift_gates_simplify1(lk_target_qubits_list, lk_control_qubits_list, lk_control_states_list,
                          rk_target_qubits_list, rk_control_qubits_list, rk_control_states_list):
    """
    Simplify left and right shift gates by removing redundant gates and adjusting control states.

    This function simplifies the sequences of left and right shift gates by checking for gates with the same control qubits and states.
    If two gates have the same control qubits and states, they are merged or removed.
        If they differ by only one control state, the gate with the differing state is adjusted.

    Args:
        lk_target_qubits_list (list[list[int]]): A list of lists of target qubit indices for left shift gates.
        lk_control_qubits_list (list[list[int]]): A list of lists of control qubit indices for left shift gates.
        lk_control_states_list (list[list[int]]): A list of lists of control states for left shift gates.
        rk_target_qubits_list (list[list[int]]): A list of lists of target qubit indices for right shift gates.
        rk_control_qubits_list (list[list[int]]): A list of lists of control qubit indices for right shift gates.
        rk_control_states_list (list[list[int]]): A list of lists of control states for right shift gates.

    Returns:
        tuple: A tuple containing simplified lists of target qubits, control qubits, and control states for both left and right shift gates.

    Raises:
        ValueError: If the input lists are not properly formatted or if the lengths of corresponding lists do not match.
    """
    # A flag to check if input data are changed
    change = False

    lk_target_qubits_list, lk_control_qubits_list, lk_control_states_list = shift_gate_simplify2(
        lk_target_qubits_list, lk_control_qubits_list, lk_control_states_list)
    rk_target_qubits_list, rk_control_qubits_list, rk_control_states_list = shift_gate_simplify2(
        rk_target_qubits_list, rk_control_qubits_list, rk_control_states_list)

    for lk_index, (lk_trgt_qubits, lk_ctrl_qubits) in enumerate(zip(lk_target_qubits_list, lk_control_qubits_list)):
        if (lk_trgt_qubits, lk_ctrl_qubits) in zip(rk_target_qubits_list, rk_control_qubits_list):
            rk_indices = [index for index, value in enumerate(zip(rk_target_qubits_list, rk_control_qubits_list))
                          if value == (lk_trgt_qubits, lk_ctrl_qubits)]

            for rk_index in rk_indices:
                index_different_state = []
                for index_state, (state_lk, state_rk) in enumerate(zip(lk_control_states_list[lk_index],
                                                                       rk_control_states_list[rk_index])):
                    if state_lk != state_rk:
                        index_different_state.insert(0, index_state)

                    if len(index_different_state) > 3:
                        break

                if len(index_different_state) == 0:
                    del lk_target_qubits_list[lk_index]
                    del lk_control_qubits_list[rk_index]
                    del lk_control_states_list[rk_index]
                    del rk_target_qubits_list[rk_index]
                    del rk_control_qubits_list[rk_index]
                    del rk_control_states_list[rk_index]
                    break
                elif len(index_different_state) == 1:
                    del lk_control_qubits_list[lk_index][index_different_state[0]]
                    del lk_control_states_list[lk_index][index_different_state[0]]
                    if len(rk_target_qubits_list[rk_index]) > 1:
                        rk_target_qubits_list[rk_index].pop()
                    else:
                        del rk_target_qubits_list[rk_index]
                        del rk_control_qubits_list[rk_index]
                        del rk_control_states_list[rk_index]
                    change = True
                    break
                elif len(index_different_state) == 2:
                    if np.sum(np.array(lk_control_states_list[lk_index]) - np.array(rk_control_states_list[rk_index])) == 0:
                        for index in index_different_state:
                            if lk_control_states_list[lk_index][index] == 0:
                                del lk_control_qubits_list[lk_index][index]
                                del lk_control_states_list[lk_index][index]
                            if rk_control_states_list[rk_index][index] == 0:
                                del rk_control_qubits_list[rk_index][index]
                                del rk_control_states_list[rk_index][index]
                        change = True
                        break

    # Iterate if change is True
    if change:
        return shift_gates_simplify1(lk_target_qubits_list, lk_control_qubits_list, lk_control_states_list,
                                     rk_target_qubits_list, rk_control_qubits_list, rk_control_states_list)
    else:
        return (lk_target_qubits_list, lk_control_qubits_list, lk_control_states_list,
                rk_target_qubits_list, rk_control_qubits_list, rk_control_states_list)


def shift_gate_simplify2(target_qubits_list, control_qubits_list, control_states_list):
    """
    Simplify a single shift gate by removing redundant gates and adjusting control states.

    This function simplifies a list of shift gates by checking for gates with the same control qubits and states.
    If two gates have the same control qubits and states, they are merged or removed.
        If they differ by only one control state, the gate with the differing state is adjusted.

    Args:
        target_qubits_list (list[list[int]]): A list of lists of target qubit indices for the shift gates.
        control_qubits_list (list[list[int]]): A list of lists of control qubit indices for the shift gates.
        control_states_list (list[list[int]]): A list of lists of control states for the shift gates.

    Returns:
        tuple: A tuple containing simplified lists of target qubits, control qubits, and control states.

    Raises:
        ValueError: If the input lists are not properly formatted or if the lengths of corresponding lists do not match.
    """
    target_qubits_list_simplify = []
    control_qubits_list_simplify = []
    control_states_list_simplify = []

    # A flag to check if input data are changed
    change = False

    for gate_index, (trgt_qubits, ctrl_qubits) in enumerate(zip(target_qubits_list, control_qubits_list)):
        if (trgt_qubits, ctrl_qubits) in zip(target_qubits_list_simplify, control_qubits_list_simplify):

            indices = [index for index, (trgt_qubits_simplify, ctrl_qubits_simplify) in
                       enumerate(zip(target_qubits_list_simplify, control_qubits_list_simplify))
                       if (trgt_qubits_simplify, ctrl_qubits_simplify) == (trgt_qubits, ctrl_qubits)]

            mode_deal = {}

            for index in indices:
                index_different_state = []

                for index_state, (state_i, state_j) in enumerate(zip(control_states_list[gate_index],
                                                                     control_states_list_simplify[index])):
                    if state_i != state_j:
                        index_different_state.append(index_state)
                        if len(index_different_state) > 1:
                            break

                if len(index_different_state) == 0:
                    mode_deal[0] = index
                    break
                elif len(index_different_state) == 1:
                    mode_deal[1] = [index, index_different_state[0]]
                elif len(index_different_state) == 2:
                    mode_deal[2] = True

            if 0 in mode_deal:
                if len(target_qubits_list_simplify[mode_deal[0]]) > 1:
                    target_qubits_list_simplify[mode_deal[0]].pop()
                    change = True
                else:
                    del target_qubits_list_simplify[mode_deal[0]]
                    del control_qubits_list_simplify[mode_deal[0]]
                    del control_states_list_simplify[mode_deal[0]]
            elif 1 in mode_deal:
                del control_qubits_list_simplify[mode_deal[1][0]][mode_deal[1][1]]
                del control_states_list_simplify[mode_deal[1][0]][mode_deal[1][1]]
                change = True
            elif 2 in mode_deal:
                target_qubits_list_simplify.append(trgt_qubits)
                control_qubits_list_simplify.append(ctrl_qubits)
                control_states_list_simplify.append(control_states_list[gate_index])

        else:
            target_qubits_list_simplify.append(trgt_qubits)
            control_qubits_list_simplify.append(ctrl_qubits)
            control_states_list_simplify.append(control_states_list[gate_index])

    if change:
        return shift_gate_simplify2(target_qubits_list_simplify, control_qubits_list_simplify, control_states_list_simplify)
    else:
        return target_qubits_list_simplify, control_qubits_list_simplify, control_states_list_simplify