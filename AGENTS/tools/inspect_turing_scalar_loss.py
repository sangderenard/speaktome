"""Retain the exact lowering of a scalar-loss adjoint for compiler diagnosis."""
import argparse
from pathlib import Path
import pickle
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.repo.resolve()))

from src.common.tensors.accelerator_backends.ssa_backend import SSATensorOperations, SSATensorProgram
from src.compiler.process_graph_autograd import obtain_graph_reverse, lower_training_motion_to_repository_ssa
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm
from src.compiler import fortran_c_shell

original_link = fortran_c_shell._class_surface_ssa_program
def retain_link(*positional, **keywords):
    from joblib.externals import cloudpickle
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'pre-frame-link.pkl').open('wb') as stream:
        cloudpickle.dump((positional, keywords), stream)
    return original_link(*positional, **keywords)
fortran_c_shell._class_surface_ssa_program = retain_link

program = SSATensorProgram('scalar_loss_join')
left = SSATensorOperations.input(program, (2, 3))
right = SSATensorOperations.input(program, (2, 3))
loss = (left * left).sum() + (right * right).sum()
product = obtain_graph_reverse(loss, bindings={'left': left, 'right': right},
    wrt=[left.data.value.id, right.data.value.id], packaging='combined', unit_output_seed=True)
lowering = lower_training_motion_to_repository_ssa(product.motion, function_name='scalar_loss_join_reverse')
args.output.mkdir(parents=True, exist_ok=True)
with (args.output / 'lowering.pkl').open('wb') as stream:
    pickle.dump(lowering, stream, protocol=5)
emitted = emit_ssa_function_to_llvm(lowering.module, lowering.function_name, entry_name=lowering.function_name)
(args.output / 'module.ll').write_text(emitted.llvm_ir, encoding='utf-8')
print('shortfalls', emitted.shortfalls)
for name, function in lowering.module.functions.items():
    if 'bw_add' not in name:
        continue
    print('FUNCTION', name, 'FORMALS', function.args)
    for label, block in function.blocks.items():
        print('BLOCK', label)
        for instruction in block.instrs:
            print(instruction)
