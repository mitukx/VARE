import json
import sys


def main() -> None:
    request = json.loads(sys.stdin.read() or "{}")
    number = request["n"]
    value = number + number
    sys.stdout.write(json.dumps({"value": value}))


if __name__ == "__main__":
    main()
