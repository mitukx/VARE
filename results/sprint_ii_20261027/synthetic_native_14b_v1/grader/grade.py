#!/usr/bin/env python3
"""AST-only grader for a synthetic native-tool interaction fixture."""
import ast
import json
import sys
from pathlib import Path


def main():
    try:
        tree = ast.parse(Path(sys.argv[1]).read_text(encoding='utf-8'))
    except (OSError, UnicodeError, SyntaxError) as exc:
        print(json.dumps({'success': False, 'reason': 'source_parse_failure',
                          'error': f'{type(exc).__name__}: {exc}'}))
        return 0
    function = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == 'add'), None)
    valid = False
    if function:
        for node in ast.walk(function):
            if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)
                    and isinstance(node.left, ast.Name) and node.left.id == 'a'
                    and isinstance(node.right, ast.Name) and node.right.id == 'b'):
                valid = True
    print(json.dumps({'success': valid, 'reason': 'contract_pass' if valid else 'addition_not_implemented'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
