# Task: Enable MPS in kernel configuration

Update kernel_config.py so MPS is a supported device type consistently across add_to_mapping, add_to_mapping_local, and sanitize_kernel_mapping. Centralize the supported device list so these validation paths cannot drift apart. Preserve all existing supported devices and continue rejecting CPU.
