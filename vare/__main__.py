import sys

if len(sys.argv) > 1 and sys.argv[1].startswith('durable-'):
    from .durable import main
else:
    from .runner import main

raise SystemExit(main())
