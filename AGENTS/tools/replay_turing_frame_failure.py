"""Replay a trusted frame checkpoint and save the SSA at an exception seam."""
import argparse
import pickle
from pathlib import Path
import sys
import traceback

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('checkpoint', type=Path)
parser.add_argument('--repo', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
sys.path.insert(0, str(args.repo.resolve()))
from src.compiler.fortran_c_shell import _class_surface_ssa_program
from src.transmogrifier.ssa import IRModule

positional, keywords = pickle.loads(args.checkpoint.read_bytes())
args.output.mkdir(parents=True, exist_ok=True)
try:
    result = _class_surface_ssa_program(*positional, **{**keywords, 'progress': lambda s: print(s, flush=True)})
except Exception as error:
    frame = error.__traceback__
    while frame:
        local = frame.tb_frame.f_locals
        if frame.tb_frame.f_code.co_name == '_class_surface_ssa_program':
            with (args.output / 'failed-functions.pkl').open('wb') as stream:
                pickle.dump(local.get('all_functions'), stream, protocol=5)
        frame = frame.tb_next
    (args.output / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
    raise
else:
    with (args.output / 'repository-ssa.pkl').open('wb') as stream:
        pickle.dump(result, stream, protocol=5)
    print('SSA saved', flush=True)
