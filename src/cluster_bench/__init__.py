"""cluster-bench — acceptance and interconnect benchmarking for new cluster nodes.

Current study (Tier 2): which sharding strategy costs least on a fabric whose
three tiers are NVLink (in-pair) >> PCIe (cross-pair) ~ RoCE (cross-node), and
in particular whether the cross-pair PCIe hop is worse than the network.

See README.md for the matrices and CLAUDE.md for the standing plan.
"""

__all__ = [
    "accel_config",
    "config",
    "data",
    "ds_config",
    "metrics",
    "modeling",
    "placement",
    "provenance",
    "strategies",
]
