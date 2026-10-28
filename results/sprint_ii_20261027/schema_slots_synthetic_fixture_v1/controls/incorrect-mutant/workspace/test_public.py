from retry_policy import retry_budget

assert retry_budget(0) == 0
assert retry_budget(1) == 1
assert retry_budget(5) == 5
assert retry_budget(32) == 32
print("4 public checks passed")
