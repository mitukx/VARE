"""Independent artifact checks reject altered policy and source identities."""
import json
from pathlib import Path
import tarfile

import pytest

from scripts.audit_flagship_results_v1 import audit_teacher

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/flagship_cached_mistral_teacher_v1.json"
ARCHIVE = ROOT / "results/flagship-cached-mistral-teacher-v1/run-1.tar.gz"


@pytest.mark.parametrize("tamper", [None, "policy", "source"])
def test_teacher_provenance_audit(tamper):
    protocol = json.loads(PROTOCOL.read_text())
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        def read(name):
            value = archive.extractfile("run-1/"+name).read()
            if tamper == "policy" and name.endswith("agent.json"):
                altered = json.loads(value)
                altered["model_snapshot"] = "wrong-policy"
                return json.dumps(altered).encode()
            if tamper == "source" and name.endswith("workspace/target.py"):
                return value + b"\n# altered\n"
            return value

        if tamper:
            with pytest.raises(ValueError, match="policy revision|unchanged baseline"):
                audit_teacher(read, protocol, protocol["qualification_run"])
        else:
            report = audit_teacher(read, protocol, protocol["qualification_run"])
            assert report["attempts"] == 4
            assert report["generated_tokens"] == 3072
            assert report["independent_successes"] == 0
