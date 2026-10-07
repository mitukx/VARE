import json
from subject import logsumexp

# Deterministic work-count proxy rather than wall-clock timing. Real task packs
# may replace this with a measured benchmark command.
xs = [1.0, 2.0, 3.0, 4.0]
logsumexp(xs)
print(json.dumps({"work_units": 4.0}))
