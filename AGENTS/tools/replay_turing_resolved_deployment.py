"""Replay a trusted source-graph checkpoint and retain exact compiler failures."""
import argparse
import ast
import inspect
import json
from pathlib import Path
import pickle
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path, nargs="?")
    parser.add_argument("--fresh-validator-build", action="store_true")
    parser.add_argument("--lanes", type=int, default=8)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--entrypoint")
    parser.add_argument("--name")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.fresh_validator_build and not (args.graph and args.entrypoint and args.name):
        parser.error("Replay requires graph, --entrypoint and --name")
    sys.path.insert(0, str(args.repo.resolve()))
    from joblib.externals import cloudpickle
    from src.compiler import fortran_c_shell as shell
    from src.compiler import glsl_deployment_strategy as strategy
    from src.compiler.project_compilation_product import _dump_resolved_process_graph
    from src.compiler.work_contract import set_active_contract
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
    set_active_contract("develop")
    args.output.mkdir(parents=True, exist_ok=True)
    graph = None if args.fresh_validator_build else pickle.loads(args.graph.read_bytes())
    original_overlay = strategy._overlay_control_or_require_subdivision
    original_link = shell._class_surface_ssa_program

    def overlay(*positional, **keywords):
        try:
            return original_overlay(*positional, **keywords)
        except Exception:
            held = inspect.signature(original_overlay).bind(*positional, **keywords)
            held.apply_defaults()
            snapshot = dict(held.arguments)
            caller = inspect.currentframe().f_back
            target = caller.f_locals.get("target")
            if target is not None:
                snapshot["dispatch_subgraphs"] = target.dispatch_subgraphs
                snapshot["loop_plans"] = target.loop_plans
                snapshot["hierarchy_plan"] = target.hierarchy_plan
            failed = snapshot["graph"]
            report = dict(function=failed.G.graph.get("function_name"),
                dependencies=snapshot["region_dependencies"],
                nesting=snapshot["nesting"],
                regions=[dict(index=i, nodes=list(sub.G.graph.get("deployment_nodes", ())))
                         for i, sub in enumerate(snapshot.get("dispatch_subgraphs", ()))],
                loops=[dict(node=plan.loop.node_id, body=list(plan.loop.body_nodes),
                            condition=list(getattr(plan.loop, "condition_nodes", ())))
                       for plan in snapshot.get("loop_plans", ())],
                controls=[repr(item) for item in (*snapshot["loop_controls"], *snapshot["conditional_controls"])],
                nodes={str(node): dict(op=data.get("op"), parents=data.get("parents"),
                    attributes=data.get("attributes"),
                    expression=ast.unparse(data["expr_obj"]) if isinstance(data.get("expr_obj"), ast.AST) else None)
                    for node, data in failed.G.nodes(data=True)})
            (args.output / "control-cycle.json").write_text(json.dumps(report, indent=2, default=str))
            with (args.output / "control-snapshot.pkl").open("wb") as stream:
                _dump_resolved_process_graph(snapshot, stream)
            print("Captured exact failing control snapshot", flush=True)
            raise

    def link(*positional, **keywords):
        with (args.output / "pre-frame-link.pkl").open("wb") as stream:
            cloudpickle.dump((positional, {k: v for k, v in keywords.items() if k != "progress"}), stream)
        print("Captured pre-frame-link checkpoint", flush=True)
        return original_link(*positional, **keywords)

    strategy._overlay_control_or_require_subdivision = overlay
    shell._class_surface_ssa_program = link
    try:
        if args.fresh_validator_build:
            from src.compiler.vehicle_validator_simulation import build_simulation
            result = build_simulation(args.output, args.lanes,
                                      progress=lambda message: print(message, flush=True))
            print(json.dumps(result, indent=2), flush=True)
            return
        result = shell._lower_resolved_process_graph_deployment(
            graph, args.entrypoint, name=args.name, runtime_closure_only=True,
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda message: print(message, flush=True))
        with (args.output / "repository-ssa.pkl").open("wb") as stream:
            pickle.dump(result, stream)
    finally:
        strategy._overlay_control_or_require_subdivision = original_overlay
        shell._class_surface_ssa_program = original_link


if __name__ == "__main__":
    main()
