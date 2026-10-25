"""Inspect one function's region planning from a trusted saved ProcessGraph."""
import argparse
import ast
import json
import pickle
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--function")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--baseline-fusion", action="store_true")
    parser.add_argument("--precompile", action="store_true")
    parser.add_argument("--control-only", action="store_true")
    parser.add_argument("--saved-function", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    if args.baseline_fusion:
        import subprocess
        from src.compiler import process_graph_fusion
        source = subprocess.run(["git", "show", "HEAD:src/compiler/process_graph_fusion.py"],
                                cwd=args.repo, text=True, capture_output=True, check=True).stdout
        exec(compile(source, "HEAD:process_graph_fusion.py", "exec"), vars(process_graph_fusion))
    from src.compiler.glsl_deployment_strategy import strategize_shell_deployment
    from src.compiler.process_graph_fusion import extract_clean_process_subgraph
    from src.compiler.project_compilation_product import _dump_resolved_process_graph
    from src.compiler.work_contract import set_active_contract
    set_active_contract("develop")
    graph = pickle.loads(args.graph.read_bytes())
    if isinstance(graph, dict):
        graph = graph["graph"]
    args.output.mkdir(parents=True, exist_ok=True)
    entries = [entry for entry in graph.function_table
               if entry.graph is not None and (
                   args.function is None or args.function in entry.qualified_name)]
    if args.saved_function:
        from types import SimpleNamespace
        entries = [SimpleNamespace(graph=graph, qualified_name=graph.G.graph["function_name"])]
    print([(entry.qualified_name, len(entry.graph.G)) for entry in entries], flush=True)
    if args.list:
        return 0
    for entry in entries:
        print("Planning " + entry.qualified_name, flush=True)
        candidate = extract_clean_process_subgraph(entry.graph, entry.graph.G)
        try:
            shell = strategize_shell_deployment(
                candidate, backend="fortran", _function_table_stack=(id(candidate.function_table),))
            instance = shell(profiling=False, shell_language="glsl")
            if args.control_only:
                # The same control-planning cut used by prepare_graph_precompile;
                # no child lowering or simulated successful compilation flag.
                from src.compiler import glsl_deployment_strategy as strategy
                from src.compiler.control_source import project_control_regions
                from src.compiler.hierarchical_plan import PlanClosure, PlanLine
                complete, dependencies = strategy._topological_region_schedule(
                    instance, range(len(instance.dispatch_subgraphs)))
                retained = {int(value) for item in instance.hierarchy_plan.items
                            if isinstance(item, PlanClosure) and item.name.startswith("region_")
                            for line in item.items if isinstance(line, PlanLine)
                            for value in (*line.inputs, *line.outputs)}
                reductions = tuple(row for row in instance.loop_shader_reductions
                                   if row.control_program is not None)
                owned = {int(region) for row in reductions for region in row.structurally_owned_region_indices}
                owned.update(int(region) for row in instance.loop_shader_reductions
                             for region in row.domain_region_indices)
                runtime = tuple(int(region) for region in complete if int(region) not in owned)
                loops = tuple(project_control_regions(row.control_program, runtime,
                              retained_value_ids=retained, preserve_source_loop_carries=True)
                              for row in reductions)
                conditionals = strategy._ordinary_conditional_control_programs(
                    candidate, runtime, instance.dispatch_subgraphs)
                nesting = strategy._source_control_nesting_hints(
                    reductions, conditionals, instance.loop_plans, candidate)
                strategy._overlay_control_or_require_subdivision(candidate, runtime,
                    reductions, loops, conditionals, nesting, region_dependencies=dependencies)
            if args.precompile:
                instance.compile_process_graph(prepare_ephemerals=False)
                instance.prepare_graph_precompile(structural_ssa_only=True,
                    progress=lambda message: print(message, flush=True))
        except Exception as error:
            frame = error.__traceback__
            while frame is not None:
                if frame.tb_frame.f_code.co_name == "_overlay_control_or_require_subdivision":
                    local = frame.tb_frame.f_locals
                    held = {key: local[key] for key in ("graph", "runtime_regions", "reductions",
                        "loop_controls", "conditional_controls", "nesting", "region_dependencies")}
                    held["dispatch_subgraphs"] = instance.dispatch_subgraphs
                    held["loop_plans"] = instance.loop_plans
                    with (args.output / "control-snapshot.pkl").open("wb") as stream:
                        _dump_resolved_process_graph(held, stream)
                    report = dict(function=entry.qualified_name, error=str(error),
                        regions=[dict(index=i, nodes=sub.G.graph.get("deployment_nodes"))
                                 for i, sub in enumerate(instance.dispatch_subgraphs)],
                        loops=[dict(node=plan.loop.node_id, body=plan.loop.body_nodes,
                                    condition=getattr(plan.loop, "condition_nodes", ()))
                               for plan in instance.loop_plans],
                        dependencies=local["region_dependencies"], nesting=local["nesting"],
                        loop_controls=[repr(value) for value in local["loop_controls"]],
                        conditionals=[repr(value) for value in local["conditional_controls"]],
                        nodes={str(node): dict(op=data.get("op"), parents=data.get("parents"),
                            attributes=data.get("attributes"),
                            expression=ast.unparse(data["expr_obj"]) if isinstance(data.get("expr_obj"), ast.AST) else None)
                            for node, data in local["graph"].G.nodes(data=True)})
                    (args.output / "control-cycle.json").write_text(json.dumps(report, indent=2, default=str))
                    print(str(error), flush=True)
                    return 1
                if frame.tb_frame.f_code.co_name == "_atomic_region_node_order":
                    local = frame.tb_frame.f_locals
                    units = local["unit_graph"]
                    import networkx as nx
                    cycles = list(nx.simple_cycles(units))
                    members = local["members"]
                    implicated = {node for cycle in cycles for unit in cycle for node in members[unit]}
                    data = dict(function=entry.qualified_name, error=str(error), cycles=cycles,
                                regions={str(k): v for k, v in members.items() if k[0] == "region"},
                                nodes={str(node): dict(
                                    op=candidate.G.nodes[node].get("op"),
                                    parents=candidate.G.nodes[node].get("parents"),
                                    expression=ast.unparse(candidate.G.nodes[node]["expr_obj"])
                                    if isinstance(candidate.G.nodes[node].get("expr_obj"), ast.AST) else None,
                                ) for node in implicated})
                    (args.output / "cycle.json").write_text(json.dumps(data, indent=2, default=str))
                    with (args.output / "function-graph.pkl").open("wb") as stream:
                        _dump_resolved_process_graph(candidate, stream)
                    print(json.dumps(data, indent=2, default=str), flush=True)
                    return 1
                frame = frame.tb_next
            raise
        print("Atomic ordering passed " + entry.qualified_name, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
