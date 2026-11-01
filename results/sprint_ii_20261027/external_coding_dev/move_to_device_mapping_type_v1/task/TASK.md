# Task

Fix move_to_device so recursively moved dict-like generation inputs keep their concrete mapping type, including nested mapping subclasses, while tensor values are still moved to the requested device.
