"""Replay a trusted resolved graph, retaining pre-frame and audit checkpoints."""
import argparse
from collections import Counter
import pickle
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('checkpoint', type=Path)
parser.add_argument('--repo', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--entrypoint', default='validator_simulation_window')
parser.add_argument('--name', default='validator_simulation')
args = parser.parse_args()
sys.path.insert(0, str(args.repo.resolve()))
from src.compiler import fortran_c_shell as shell
from joblib.externals import cloudpickle
from src.compiler.ssa_self_check import run_all
from src.compiler.work_contract import set_active_contract
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference

set_active_contract('develop')
args.output.mkdir(parents=True, exist_ok=True)
graph = pickle.loads(args.checkpoint.read_bytes())
original = shell._class_surface_ssa_program

def checkpoint(*positional, **keywords):
    saved_keywords = {key: value for key, value in keywords.items() if key != 'progress'}
    with (args.output / 'pre-frame-link.pkl').open('wb') as stream:
        cloudpickle.dump((positional, saved_keywords), stream, protocol=5)
    print('Saved pre-frame checkpoint', flush=True)
    return original(*positional, **keywords)

shell._class_surface_ssa_program = checkpoint
result = shell._lower_resolved_process_graph_deployment(
    graph, args.entrypoint, name=args.name, runtime_closure_only=True,
    tensor_ssa_reference=c_backend_repository_ssa_reference(),
    progress=lambda message: print(message, flush=True),
)
with (args.output / 'repository-ssa.pkl').open('wb') as stream:
    pickle.dump(result, stream, protocol=5)
print('Saved repository SSA', flush=True)
findings = run_all(result[0])
(args.output / 'audit.txt').write_text('\n'.join(map(str, findings)), encoding='utf-8')
print('Audit:', len(findings), Counter(getattr(item, 'check', type(item).__name__) for item in findings), flush=True)
