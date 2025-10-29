import importlib
from typing import Any

from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph.state import CompiledGraph


def import_strategy_class(strategies_path: str) -> Any:
    """
    Import a class from a dotted path like 'myproject.module.MyClass'
    """
    module_path, class_name = strategies_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def save_graph_as_png(app: CompiledGraph, output_file_path: str) -> None:
    png_image = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    file_path = output_file_path if len(output_file_path) > 0 else "graph.png"
    with open(file_path, "wb") as f:
        f.write(png_image)


def deep_merge_dicts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """
     deep merge two dicts
    :param a:
    :param b:
    :return: new dict
    """
    result = a.copy()
    for key, value in b.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
