# Task

Fix DataCollatorForLanguageModeling so seed=0 initializes its seeded random generator for both torch_call and numpy_call. Preserve the existing behavior when seed is None and for all nonzero seeds.
