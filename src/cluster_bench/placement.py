"""GPU placement — Matrix 2B.

Each node is 4 H200 in two NVLinked pairs: devices (0,1) and (2,3). A pair has
its own rail-aligned NIC. So the three link tiers are selectable purely by
choosing which devices on which hosts take part:

    NVLink  -> both ranks inside one pair
    PCIe    -> two ranks on one node, different pairs
    RoCE    -> two ranks on different nodes

`pair-per-node` vs `one-node` is the sharp comparison the whole study turns on:
same 4 GPUs' worth of compute, one of them using zero PCIe.
"""

from __future__ import annotations

from dataclasses import dataclass

from .strategies import Strategy

# Device ids that share an NVLink domain, in order. Change here if a node is
# ever cabled differently -- nothing else hardcodes the pairing.
PAIRS: tuple[tuple[int, ...], ...] = ((0, 1), (2, 3))


@dataclass(frozen=True)
class Placement:
    name: str
    # devices[i] = CUDA_VISIBLE_DEVICES for machine_rank i. Length = num_machines.
    devices: tuple[tuple[int, ...], ...]
    links: str

    @property
    def num_machines(self) -> int:
        return len(self.devices)

    @property
    def procs_per_machine(self) -> int:
        counts = {len(d) for d in self.devices}
        if len(counts) != 1:
            raise ValueError(
                f"placement {self.name!r} is ragged: {self.devices}. "
                "accelerate assumes a uniform process count per machine."
            )
        return counts.pop()

    @property
    def world_size(self) -> int:
        return sum(len(d) for d in self.devices)

    def cuda_visible_devices(self, machine_rank: int) -> str:
        return ",".join(str(d) for d in self.devices[machine_rank])


PLACEMENTS: dict[str, Placement] = {
    "within-pair": Placement("within-pair", ((0, 1),), "NVLink only"),
    "across-pairs": Placement("across-pairs", ((0, 2),), "PCIe only"),
    "across-nodes": Placement("across-nodes", ((0,), (0,)), "RoCE only"),
    "one-node": Placement("one-node", ((0, 1, 2, 3),), "NVLink + PCIe"),
    "pair-per-node": Placement(
        "pair-per-node", ((0, 1), (0, 1)), "NVLink + RoCE, zero PCIe"
    ),
    "full": Placement("full", ((0, 1, 2, 3), (0, 1, 2, 3)), "everything"),
}


def get(name: str) -> Placement:
    try:
        return PLACEMENTS[name]
    except KeyError:
        raise SystemExit(
            f"unknown placement {name!r}; choose from: {', '.join(PLACEMENTS)}"
        ) from None


def validate(strategy: Strategy, placement: Placement) -> list[str]:
    """Return reasons this (strategy, placement) cell is not measurable.

    An empty list means the cell is fine. The sweep driver skips cells with
    reasons rather than running something that silently degrades to a
    different config -- e.g. DeepSpeed clamps hpZ to the world size, so
    hpZ=4 on 2 GPUs runs as flat ZeRO-3 while still being labelled hpz4.
    """
    reasons: list[str] = []
    ws = placement.world_size

    if ws < strategy.min_world_size:
        reasons.append(
            f"{strategy.name} needs world_size >= {strategy.min_world_size}, "
            f"placement gives {ws}"
        )
    if ws % strategy.world_size_multiple_of:
        reasons.append(
            f"{strategy.name} needs world_size divisible by "
            f"{strategy.world_size_multiple_of}, placement gives {ws}"
        )
    if strategy.hpz and strategy.hpz > ws:
        reasons.append(f"hpz={strategy.hpz} exceeds world_size {ws}")
    if strategy.hpz and strategy.hpz == ws:
        reasons.append(
            f"hpz={strategy.hpz} equals world_size {ws}: degenerates to "
            "replication, not a distinct config"
        )
    if strategy.autotp and ws % strategy.autotp:
        reasons.append(f"autotp={strategy.autotp} does not divide world_size {ws}")

    # hpZ=4 is "node-local" only if a gather group is exactly one node's GPUs.
    if strategy.hpz == 4 and placement.procs_per_machine != 4:
        reasons.append(
            f"hpz=4 is only node-local when procs_per_machine == 4; placement "
            f"{placement.name} has {placement.procs_per_machine}"
        )
    # hpZ=2 is "pair-local" only if consecutive rank pairs are NVLinked.
    if strategy.hpz == 2:
        for rank_devs in placement.devices:
            for lo in range(0, len(rank_devs), 2):
                chunk = tuple(rank_devs[lo : lo + 2])
                if len(chunk) == 2 and chunk not in PAIRS:
                    reasons.append(
                        f"hpz=2 assumes adjacent ranks are NVLinked, but "
                        f"placement {placement.name} puts devices {chunk} together"
                    )
    return reasons
