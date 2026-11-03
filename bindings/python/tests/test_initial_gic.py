"""Explicit ARMv7-A inherited CPU-interface inputs, independent of guest code."""
import pytest

from fujimod._native.emulator import NativeError
from fujimod.emulator.board import make_board
from fujimod.emulator.definition import Xp4Definition


def board_options(**extra):
    return dict(ram_size=0x10000, cores=2, discovery=True, cbar=0xffff0000, **extra)


@pytest.mark.parametrize('extra', [{}, dict(initial_gic_cpu_interfaces=[])])
def test_omitting_initial_interfaces_retains_cold_controller_state(extra):
    with make_board(Xp4Definition(options=board_options(**extra)), {}) as board:
        assert [state['cpu_control'] for state in board.native.xp4.interrupt_state()] == [
            [0, 0, 0], [0, 0, 0]]


def test_initial_interfaces_are_per_core_and_do_not_enable_distributor():
    initial = [dict(enabled=True, priority_mask=255, binary_point=3),
               dict(enabled=False, priority_mask=64, binary_point=0)]
    with make_board(Xp4Definition(options=board_options(initial_gic_cpu_interfaces=initial)), {}) as board:
        states = board.native.xp4.interrupt_state()
        assert [state['cpu_control'] for state in states] == [[1, 255, 3], [0, 64, 0]]
        assert all(state['distributor_control'] == [0] for state in states)
        assert all(not any(state['pending']) and not any(state['active']) for state in states)
        assert board.instructions == 0


@pytest.mark.parametrize('initial,error,message', [
    ([dict(enabled=True, priority_mask=255, binary_point=0)], NativeError, 'one valid state per core'),
    ([dict(enabled=True, priority_mask=255, binary_point=8)] * 2, NativeError, 'one valid state per core'),
    ([dict(enabled=True, priority_mask=256, binary_point=0)] * 2, ValueError, 'u8'),
    ([dict(enabled=1, priority_mask=255, binary_point=0)] * 2, ValueError, 'boolean'),
    ([dict(enabled=True, priority_mask=255, binary_point=0, unsupported=1)] * 2, ValueError, 'unknown field.*unsupported'),
])
def test_invalid_initial_interfaces_refuse_construction(initial, error, message):
    with pytest.raises(error, match=message):
        make_board(Xp4Definition(options=board_options(initial_gic_cpu_interfaces=initial)), {})


def test_initial_interfaces_require_declared_configuration_base():
    options = board_options(initial_gic_cpu_interfaces=[
        dict(enabled=True, priority_mask=255, binary_point=0)] * 2)
    options.pop('cbar')
    with pytest.raises(NativeError, match='CBAR'):
        make_board(Xp4Definition(options=options), {})
