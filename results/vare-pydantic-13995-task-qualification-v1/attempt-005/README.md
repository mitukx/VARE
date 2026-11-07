# Attempt 005: preflight stopped before grading

The qualification adapter referenced a mutant hash key that was absent from its frozen run lock. Its preflight stopped before launching a grader container. No arm was run, no reward was assigned, and no model prompt or held-out task was exposed. This rejected freeze is preserved; a corrected lock and adapter are used for the next run.
