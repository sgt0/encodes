from __future__ import annotations

from pathlib import Path

from vsmasktools import CustomMaskFromRanges
from vspreview import is_preview
from vstools import core, vs

from dededede_common import filterchain, mux, sources

core.set_affinity(22, 2 << 13)


EPISODE = "05"
OP = (2722, 2722 + 2095)
# Insert ED.

# Ranges with Arial signs.
ARIAL_SIGNS = [
    (10393, 10446),
    (13367, 13450),
    (21152, 21199),
    (22325, 22429),
    (22730, 22813),
    (23237, 23284),
    (23468, 23527),
    (23881, 24066),
    (25354, 25449),
    (30514, 30561),
    (30709, 30798),
]

# Ranges with good signs.
GOOD_SIGNS = [
    (2536, 2613),
    (2871, 2976),
    (4761, 4818),
]


def custom_dehardsub(src: vs.VideoNode, dehardsub: vs.VideoNode) -> vs.VideoNode:
    custom_dehardsub_mask = CustomMaskFromRanges(
        ranges={
            Path("masks/05_30514.png"): (30514, 30561),  # Fade.
        }
    ).get_mask(dehardsub)
    return dehardsub.std.MaskedMerge(src, custom_dehardsub_mask)


filterchain_results = filterchain(
    op=OP,
    arial_signs=ARIAL_SIGNS,
    good_signs=GOOD_SIGNS,
    jp_src_path=sources[5].jp_src_path,
    dub_src_path=sources[5].dub_src_path,
    custom_dehardsub=custom_dehardsub,
)

if not is_preview():
    mux(episode=EPISODE, op=OP, filterchain_results=filterchain_results)
