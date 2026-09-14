"""Capture lexical control stages during an existing Turing lowering driver."""

import argparse
import pickle
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', required=True)
    parser.add_argument('--match', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path.cwd()))
    from src.compiler import fortran_c_shell as shell, precompile_to_ssa as precompile
    from src.compiler.project_compilation_product import _dump_resolved_process_graph

    args.output.mkdir(parents=True, exist_ok=True)
    stages = ('_place_plan_callsites_lexically', '_install_lexical_sequence_mutations',
              '_install_lexical_sequence_queries', '_attach_graph_control_expressions')

    def wrap(name, original):
        def capture(control, graph, *positional, **keywords):
            result = original(control, graph, *positional, **keywords)
            graph_obj = getattr(graph, 'G', graph)
            if args.match in str(graph_obj.graph.get('function_name', '')):
                with (args.output / (name + '.pkl')).open('wb') as stream:
                    _dump_resolved_process_graph((control, result, graph, positional), stream)
            return result
        return capture

    for name in stages:
        setattr(shell, name, wrap(name, getattr(shell, name)))
    original = precompile.lower_control_sections_to_ssa

    def final(control, **keywords):
        name = str(keywords.get('control_name', ''))
        if args.match in name:
            with (args.output / 'final-control.pkl').open('wb') as stream:
                _dump_resolved_process_graph((control, keywords), stream)
        return original(control, **keywords)

    precompile.lower_control_sections_to_ssa = final
    sys.argv = [args.module]
    runpy.run_module(args.module, run_name='__main__')


if __name__ == '__main__':
    main()
