#!/usr/bin/env python3
"""Safe AST-only grader for MusicGen Melody cached position growth."""
import ast
import json
import sys
from pathlib import Path


def attr(node, owner, name):
    return (isinstance(node, ast.Attribute) and node.attr == name
            and isinstance(node.value, ast.Name) and node.value.id == owner)


def is_weight_size(node):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'size' and attr(node.func.value, 'self', 'weights')
            and len(node.args) == 1 and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == 0)


def is_required_sum(node):
    return (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)
            and isinstance(node.left, ast.Name) and node.left.id == 'seq_len'
            and isinstance(node.right, ast.Name) and node.right.id == 'past_key_values_length')


def find_contract(source):
    tree = ast.parse(source)
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == 'MusicgenMelodySinusoidalPositionalEmbedding'), None)
    if cls is None:
        return None, 'target_class_missing'
    forward = next((n for n in cls.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'forward'), None)
    if forward is None:
        return None, 'forward_missing'
    matched = []
    for node in ast.walk(forward):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        test = node.test
        if (is_required_sum(test.left) and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Gt) and len(test.comparators) == 1
                and is_weight_size(test.comparators[0])):
            calls = [n for stmt in node.body for n in ast.walk(stmt)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and n.func.attr == 'make_weights'
                     and isinstance(n.func.value, ast.Name) and n.func.value.id == 'self']
            if calls and calls[0].args:
                matched.append(calls[0].args[0])
    if not matched:
        return None, 'cached_length_growth_condition_missing'
    if not any(is_required_sum(arg) for arg in matched):
        return None, 'growth_uses_wrong_requested_length'
    position_ids = any(
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)
        and isinstance(node.left, ast.Call) and isinstance(node.left.func, ast.Attribute)
        and node.left.func.attr == 'arange'
        and any(isinstance(item, ast.Name) and item.id == 'past_key_values_length'
                for item in ast.walk(node.right))
        for node in ast.walk(forward)
    )
    if not position_ids:
        return None, 'cached_position_offset_missing'
    index_select = any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'index_select'
        and any(isinstance(item, ast.Name) and item.id == 'position_ids'
                for arg in node.args for item in ast.walk(arg))
        for node in ast.walk(forward)
    )
    if not index_select:
        return None, 'position_ids_not_used_for_embedding_lookup'
    return matched[0], None


def main():
    path = Path(sys.argv[1])
    try:
        source = path.read_text(encoding='utf-8')
        argument, reason = find_contract(source)
    except (OSError, UnicodeError, SyntaxError) as exc:
        result = {'success': False, 'reason': 'source_parse_failure',
                  'error': f'{type(exc).__name__}: {exc}'}
    else:
        if reason:
            result = {'success': False, 'reason': reason}
        else:
            # Evaluate only the admitted arithmetic AST; never import or run candidate code.
            needed = 2 + 5
            requested = (argument.left.id == 'seq_len'
                         and argument.right.id == 'past_key_values_length')
            result = {
                'success': requested and needed > 4 and needed >= 2 + 5,
                'reason': 'contract_pass' if requested else 'wrong_growth_expression',
                'counterexample': {'seq_len': 2, 'past_key_values_length': 5,
                                   'initial_weight_rows': 4, 'required_weight_rows': needed,
                                   'observed_growth_argument': 'seq_len + past_key_values_length'
                                   if requested else 'not_the_required_sum'},
            }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
