from __future__ import annotations

from pathlib import Path

from vsmasktools import CustomMaskFromRanges
from vspreview import is_preview
from vstools import core, vs

from dededede_common import filterchain, mux, sources

core.set_affinity(22, 2 << 13)


EPISODE = "02"
OP = (708, 708 + 2095)
ED = (32144, 32144 + 2124)

# Ranges with Arial signs.
ARIAL_SIGNS = [
    (5245, 5341),
    (9842, 9931),
    (11861, 11991),
    (16371, 16422),  # fade in
    (16507, 16578),
    (20887, 20970),
    (21724, 21837),
    (22270, 22293),
    (22420, 22485),
    (22493, 22545),
    (22594, 22653),
    (23002, 23050),  # fade out
    (23092, 23163),
    (23899, 23946),
]

# Ranges with good signs.
GOOD_SIGNS = [
    (528, 599),
    (857, 962),
    (2747, 3003),
    (3115, 3186),
    (4777, 4848),
    (15790, 15885),
    (17281, 17463),
    (23164, 23463),
    (28096, 28155),
    (28483, 28743),  # both good and bad together
    (30994, 31077),
    (31171, 31194),
    (31513, 31596),
]


def custom_dehardsub(src: vs.VideoNode, dehardsub: vs.VideoNode) -> vs.VideoNode:
    custom_dehardsub_mask = (
        CustomMaskFromRanges(
            ranges={
                Path("masks/02_16371.png"): (16371, 16422),  # Fade in.
                Path("masks/02_23002.png"): (23002, 23050),  # Fade out.
                Path("masks/02_28483.png"): (28483, 28743),  # Both kinds together.
            }
        )
        .get_mask(dehardsub)
        .std.Maximum()
        .std.BoxBlur()
    )
    return dehardsub.std.MaskedMerge(src, custom_dehardsub_mask)


filterchain_results = filterchain(
    ed=ED,
    arial_signs=ARIAL_SIGNS,
    good_signs=GOOD_SIGNS,
    jp_src_path=sources[2].jp_src_path,
    dub_src_path=sources[2].dub_src_path,
    custom_dehardsub=custom_dehardsub,
)

if not is_preview():
    mux(episode=EPISODE, op=OP, ed=ED, filterchain_results=filterchain_results)
