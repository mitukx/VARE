# Restore backward propagation through routed expert blocks

Training several mixture-of-experts models fails during backward propagation through top-k expert routing. Inspect the provided model implementations and correct the routing behavior so discrete routing choices do not interrupt gradient computation. Preserve the selected experts, forward behavior, and gradients through the selected expert computation. Make a focused source change.
