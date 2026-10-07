"""Common CLI for evaluation execution and the experimental control plane."""
import sys


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else '--help'
    if command in {'--help', '-h'}:
        print('VARE commands:\n'
              '  run, audit                  bounded evaluation campaigns\n'
              '  durable-init, durable-work  persistent evaluation enrollment/execution\n'
              '  durable-export, durable-audit  retained recovery evidence\n'
              '  demo, rvl-inspect, rvl-plan experimental improvement controls\n'
              '  env-validate, env-verify, env-smoke, env-campaign environment experiments\n'
              '  lock-protocol, score, manifest  protocol/evidence utilities\n'
              'Use COMMAND --help for arguments. Full control-plane commands need Python >=3.11.')
        return 0
    if command.startswith('durable-'):
        from .durable import main as selected
    elif command in {'run', 'audit'}:
        from .runner import main as selected
    else:
        from .cli import main as selected
    return selected()
