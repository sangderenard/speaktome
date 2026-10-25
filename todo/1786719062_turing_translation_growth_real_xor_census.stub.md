# Capture the real abstract_nn translation growth census

Run the visible evolution launcher on
`turing/examples/xor_project/train_xor.py --entrypoint train` long enough for
class-attributed expansion to cross ProcessGraph, precompile, and SSA. Record
the top-K size/rate/depth/height census and adjust the default emergency clamp
only from that measured evidence. Do not replace the program with a smaller
synthetic capture.
