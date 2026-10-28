"""Small visible smoke test; the independent grader covers the full matrix."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from transformers.data.data_collator import default_data_collator  # noqa: E402


def test_numpy_int32_scalar_label_keeps_numpy_integer_dtype():
    batch = default_data_collator(
        [
            {"input_ids": [1, 2], "label": np.int32(-7)},
            {"input_ids": [3, 4], "label": np.int32(12)},
        ],
        return_tensors="np",
    )

    assert batch["labels"].dtype == np.dtype(np.int64)
    assert batch["labels"].tolist() == [-7, 12]


if __name__ == "__main__":
    test_numpy_int32_scalar_label_keeps_numpy_integer_dtype()

