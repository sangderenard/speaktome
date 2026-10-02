"""Trace a focused test's loop/call operand identities at compiler seams."""
import argparse
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', required=True, type=Path)
parser.add_argument('--function', required=True)
parser.add_argument('test')
args = parser.parse_args()
sys.path.insert(0, str(args.repo.resolve()))
from src.compiler import fortran_c_shell as shell

def trace(functions, stage):
    function = functions.get(args.function)
    if function is None:
        return
    print('STAGE', stage, flush=True)
    for block in function.blocks.values():
        for item in block.instrs:
            if item.op not in {'Phi', 'Call'}:
                continue
            def value(v):
                return (v.id, v.dtype, v.shape, {k: v.accounting[k] for k in (
                    'ssa_loop_carried_feed', 'ssa_storage_alias', 'source_value_id') if k in v.accounting})
            print(block.name, item.op, [value(v) for v in item.args],
                  value(item.res) if item.res else None,
                  {k: item.attributes[k] for k in ('callee', 'plan_callsite_id', 'plan_callsite_marker',
                   'initial_value_id', 'updated_value_id') if k in item.attributes}, flush=True)

shell._debug_loop_carried_operands = trace
import pytest
raise SystemExit(pytest.main(['-q', '-s', args.test]))
