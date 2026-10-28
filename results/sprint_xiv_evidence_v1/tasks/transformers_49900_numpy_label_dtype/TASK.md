# Preserve integer labels from NumPy scalar values

Repair the default data collator in src/transformers/data/data_collator.py.
When a batch contains NumPy integer scalar labels, the collator must preserve
integer label dtypes for both PyTorch and NumPy outputs, including the label
and label_ids fields. Keep existing behavior for ordinary Python integer and
floating-point labels.

Use the provided source-navigation and edit tools. The public test is a small
smoke check; the independent grader checks additional integer widths, label
forms, output backends, values, and floating-point controls.

