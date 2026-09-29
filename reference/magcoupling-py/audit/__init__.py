"""Independent verification of the magcoupling engine (M1 of the magcoupling spec).

The vendored engine under ``magcoupling/`` is a read-only oracle here: nothing in
``audit/`` modifies it. Each check compares an engine value against an
independent reference (re-derivation, separate numerical model, limit/scaling
law, or literature formula). A failing check is a *candidate finding*.
"""
