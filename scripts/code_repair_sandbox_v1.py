"""Evaluate submitted task functions without executing candidate Python code."""
from __future__ import annotations

import ast
import math
from typing import Any


MAX_SOURCE_CHARS = 4096
MAX_AST_NODES = 256
MAX_SEQUENCE = 256
MAX_DEPTH = 32
MAX_ABS_NUMBER = 10**9


def _check_value(value: Any, depth: int = 0) -> Any:
    if depth > MAX_DEPTH:
        raise ValueError("result nesting limit exceeded")
    if type(value) in (int, bool, str) or value is None:
        if type(value) is int and abs(value) > MAX_ABS_NUMBER:
            raise ValueError("integer result exceeds limit")
        return value
    if type(value) is float:
        if not math.isfinite(value) or abs(value) > MAX_ABS_NUMBER:
            raise ValueError("non-finite or oversized float result")
        return value
    if type(value) in (list, tuple):
        if len(value) > MAX_SEQUENCE:
            raise ValueError("sequence result exceeds limit")
        return type(value)(_check_value(x, depth + 1) for x in value)
    raise ValueError("unsupported value type")


def _eval(node: ast.AST, env: dict[str, Any], depth: int = 0) -> Any:
    if depth > MAX_DEPTH:
        raise ValueError("expression nesting limit exceeded")
    if isinstance(node, ast.Constant):
        if type(node.value) not in (int, bool, float, str, type(None)):
            raise ValueError("unsupported constant")
        return _check_value(node.value)
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError("unknown identifier")
        return env[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_eval(item, env, depth + 1) for item in node.elts]
        return _check_value(values if isinstance(node, ast.List) else tuple(values))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
        value = _eval(node.operand, env, depth + 1)
        if isinstance(node.op, ast.Not):
            return not value
        if type(value) not in (int, float):
            raise ValueError("unary arithmetic requires a number")
        return _check_value(value if isinstance(node.op, ast.UAdd) else -value)
    if isinstance(node, ast.BinOp):
        left, right = _eval(node.left, env, depth + 1), _eval(node.right, env, depth + 1)
        operations = {
            ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
            ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
        }
        operation = operations.get(type(node.op))
        if operation is None or type(left) not in (int, float) or type(right) not in (int, float):
            raise ValueError("unsupported arithmetic operation")
        return _check_value(operation(left, right))
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        if isinstance(node.op, ast.And):
            result = True
            for item in node.values:
                result = _eval(item, env, depth + 1)
                if not result:
                    return result
            return result
        result = False
        for item in node.values:
            result = _eval(item, env, depth + 1)
            if result:
                return result
        return result
    if isinstance(node, ast.Compare):
        left = _eval(node.left, env, depth + 1)
        operations = {ast.Lt: lambda a,b:a < b, ast.LtE: lambda a,b:a <= b,
            ast.Gt: lambda a,b:a > b, ast.GtE: lambda a,b:a >= b,
            ast.Eq: lambda a,b:a == b, ast.NotEq: lambda a,b:a != b}
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval(comparator, env, depth + 1)
            function = operations.get(type(op))
            if function is None or not function(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        branch = node.body if _eval(node.test, env, depth + 1) else node.orelse
        return _eval(branch, env, depth + 1)
    if isinstance(node, ast.ListComp):
        if len(node.generators) != 1:
            raise ValueError("only one list-comprehension clause is allowed")
        generator = node.generators[0]
        if generator.is_async or not isinstance(generator.target, ast.Name):
            raise ValueError("unsupported comprehension target")
        source = _eval(generator.iter, env, depth + 1)
        if type(source) not in (list, tuple) or len(source) > MAX_SEQUENCE:
            raise ValueError("comprehension source is not a bounded sequence")
        output = []
        for value in source:
            local = dict(env)
            local[generator.target.id] = value
            if all(_eval(condition, local, depth + 1) for condition in generator.ifs):
                output.append(_eval(node.elt, local, depth + 1))
                if len(output) > MAX_SEQUENCE:
                    raise ValueError("comprehension result exceeds limit")
        return _check_value(output)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        arguments = [_eval(item, env, depth + 1) for item in node.args]
        functions = {"sum": sum, "len": len, "min": min, "max": max, "sorted": sorted}
        function = functions.get(node.func.id)
        if function is None:
            raise ValueError("function call is not allowed")
        if any(type(arg) in (list, tuple) and len(arg) > MAX_SEQUENCE for arg in arguments):
            raise ValueError("function input sequence exceeds limit")
        return _check_value(function(*arguments))
    raise ValueError(f"unsupported expression: {type(node).__name__}")


def evaluate_function(source: str, function_name: str, inputs: dict[str, Any]) -> Any:
    if not isinstance(source, str) or len(source) > MAX_SOURCE_CHARS:
        raise ValueError("source length is invalid")
    if not isinstance(inputs, dict):
        raise ValueError("function inputs must be an object")
    tree = ast.parse(source, mode="exec")
    if len(list(ast.walk(tree))) > MAX_AST_NODES or len(tree.body) != 1:
        raise ValueError("source structure exceeds the frozen subset")
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef) or function.name != function_name:
        raise ValueError("expected exactly one named function")
    if (function.decorator_list or function.returns is not None or function.type_comment is not None or
        function.args.vararg is not None or function.args.kwarg is not None or
        function.args.kwonlyargs or function.args.defaults or function.args.kw_defaults or
        function.args.posonlyargs or len(function.body) != 1 or
        not isinstance(function.body[0], ast.Return) or function.body[0].value is None):
        raise ValueError("function is outside the frozen source subset")
    names = [arg.arg for arg in function.args.args]
    if len(set(names)) != len(names) or set(inputs) != set(names):
        raise ValueError("function arguments do not match task inputs")
    env = {name: _check_value(inputs[name]) for name in names}
    return _check_value(_eval(function.body[0].value, env))
