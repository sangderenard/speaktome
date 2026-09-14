"""Print source positions beside a captured Turing control tree."""

import argparse
import pickle
import re
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--stage', default='final-control')
    args = parser.parse_args()
    sys.path.insert(0, str(Path.cwd()))
    from src.compiler.fortran_c_shell import _authored_node_position
    from src.compiler.control_source import SequenceBlock, StatementBlock

    with (args.directory / '_install_lexical_sequence_queries.pkl').open('rb') as stream:
        _before, _result, graph, positional = pickle.load(stream)
    regions = positional[0]
    with (args.directory / (args.stage + '.pkl')).open('rb') as stream:
        held = pickle.load(stream)
    control = held[0] if args.stage == 'final-control' else held[1]
    if isinstance(control, tuple):
        control = control[0]

    def position(node):
        return _authored_node_position(graph.G, node)[:2] if node is not None else None

    def walk(block, depth=0):
        if isinstance(block, SequenceBlock):
            for child in block.blocks:
                walk(child, depth)
            return
        node = next((getattr(block, field, None) for field in (
            'source_node_id', 'source_loop_node_id', 'source_call_node_id', 'site_node_id'
        ) if getattr(block, field, None) is not None), None)
        mutation = getattr(block, 'mutation', None)
        if mutation is not None:
            node = mutation.effect_node_id
        label = type(block).__name__
        if isinstance(block, StatementBlock):
            label = ' '.join(block.lines)
            match = re.fullmatch(r'__scheduled_region_(\d+)__', label)
            if match:
                nodes = regions[int(match[1])].G.graph.get('deployment_nodes', ())
                positions = [position(n) for n in nodes]
                label += f' span={min(positions, default=None)}..{max(positions, default=None)}'
            call = re.fullmatch(r'__plan_callsite_(\d+)__', ' '.join(block.lines))
            if call:
                node = int(call[1])
        print('  ' * depth + f'{label} node={node} source={position(node)}')
        for field in ('condition', 'body', 'orelse', 'callee'):
            child = getattr(block, field, None)
            if child is not None:
                print('  ' * (depth + 1) + field)
                walk(child, depth + 2)

    walk(control.root)


if __name__ == '__main__':
    main()
