import numpy as np
import math


def check_hermitian(op: dict[tuple, int|float|complex]):
    """
    op : dict
        Operator of the format
        tuple : amplitude,

        where each tuple describes a scattering process. 

        Examples of possible tuples (and their meanings) are:

        ((i, 'c'),)  <-> c_i^dagger

        ((i, 'a'),)  <-> c_i

        ((i, 'c'), (j, 'a'))  <-> c_i^dagger c_j

        ((i, 'c'), (j, 'c'), (k, 'a'), (l, 'a')) <-> c_i^dagger c_j^dagger c_k c_l
    """
    # op^\dagger = sum_p (value_p * p)^\dagger = \sum_p value_p^* * p^\dagger = \sum_q value_q * q
    # where 
    # q = p^\dagger
    # and 
    # value_q = value_p^*
    #
    # If Hermitian operator: op^\dagger = op,
    # so for every process p there should also be a process q = p^\dagger
    # with amplitude value_q = value_p^*
    for process, value in op.items():
        assert isinstance(process, tuple)
        assert isinstance(value, (int, float, complex))
        # Process q is the same as process p but opposite event order and changed "a" with "c" and vice versa
        process_q = []
        for event in process[::-1]:
            assert isinstance(event, tuple)
            assert len(event) == 2
            
            # First element should either be a superindex or a tuple of values describing a spin-orbital, e.g. (l,s,m)
            index = event[0]
            if isinstance(index, int):
                # Integer is interpreted as a superindex
                assert index >= 0
            else:
                assert isinstance(index, tuple)
                assert len(index) > 0
            
            if event[1] == "c":
                event_in_q = (index, "a")
            elif event[1] == "a":
                event_in_q = (index, "c")
            else:
                raise ValueError(event)
            process_q.append(event_in_q)
        process_q = tuple(process_q)
        
        # Check Hermitian properties 
        assert process_q in op
        np.testing.assert_allclose(op[process_q], np.conjugate(value))
