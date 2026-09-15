"""Print compact call/projection contracts from a trusted repository checkpoint."""
import argparse
import pickle
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('checkpoint', type=Path)
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--function', required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.repo.resolve()))
module, _, _ = pickle.loads(args.checkpoint.read_bytes())
for name, function in module.functions.items():
    if args.function != name:
        continue
    for block in function.blocks.values():
        for item in block.instrs:
            if item.op not in {'Call', 'GetElementPtr', 'Load'}:
                continue
            attrs = {key: item.attributes[key] for key in (
                'callee', 'output_ids', 'output_positions', 'callee_output_ids',
                'aggregate_index', 'source_output_id') if key in item.attributes}
            value = lambda v: (v.id, v.dtype, v.shape)
            print(block.name, item.op, [value(v) for v in item.args],
                  value(item.res) if item.res else None, attrs)
