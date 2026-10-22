# Task: preserve 4D attention masks during batch collation

When a training batch contains examples with four-dimensional attention masks, collation must preserve the mask's batch, head, query, and key dimensions and values for both supported tensor input types. Keep existing tokenizer and ordinary padded batch behavior intact. Make the smallest change in the permitted file, `src/transformers/tokenization_utils_base.py`.

The candidate workspace contains the pinned public source package and this task statement only. Do not rely on tests, Git history, network access, or external files.
