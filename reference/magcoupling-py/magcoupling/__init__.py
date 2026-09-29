"""Magnetic slip coupling design calculator (Python port of magnetic_coupling_torque_calculator.xlsx).

Quick use:
    from magcoupling import DesignInputs, compute_all, headline
    res = compute_all(DesignInputs())
    print(headline(res))
"""
from .api import DesignInputs, DesignResults, compute_all, headline, input_schema, result_schema, set_input, to_dict

__all__ = ["DesignInputs", "DesignResults", "compute_all", "headline", "input_schema", "result_schema", "set_input", "to_dict"]
__version__ = "1.0.0"
