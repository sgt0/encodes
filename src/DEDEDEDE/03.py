from __future__ import annotations

from pathlib import Path

from vsmasktools import CustomMaskFromRanges
from vspreview import is_preview
from vstools import core, vs

from dededede_common import filterchain, mux, sources

core.set_affinity(22, 2 << 13)


EPISODE = "03"
OP = (1404, 1404 + 2095)
ED = (32144, 32144 + 2124)

# Ranges with Arial signs.
ARIAL_SIGNS = [
    (5571, 5710),  # Fade.
    (6394, 6491),  # Fade.
    (9967, 10056),
    (10964, 10997),
    (11931, 11969),  # Fade.
    (12813, 12896),
    (22329, 22362),
    (23276, 23356),  # Fade.
    (28537, 28608),
    (28618, 28656),  # Fade.
]

# Ranges with good signs.
GOOD_SIGNS = [
    (324, 395),
    (1224, 1295),
    (1553, 1658),
    (3443, 4461),
    (22477, 22592),
    (22901, 22954),
    (24089, 24160),
    (25122, 25193),
    (26198, 26269),
    (27785, 28030),
    (28249, 28380),
    (28717, 28800),
]


def custom_dehardsub(src: vs.VideoNode, dehardsub: vs.VideoNode) -> vs.VideoNode:
    custom_dehardsub_mask = (
        CustomMaskFromRanges(
            ranges={
                Path("masks/03_5571.png"): (5571, 5710),  # Fade.
                Path("masks/03_6394.png"): (6394, 6491),  # Fade.
                Path("masks/03_11931.png"): (11931, 11969),  # Fade.
                Path("masks/03_23276.png"): (23276, 23356),  # Fade.
                Path("masks/03_28618.png"): (28618, 28656),  # Fade.
            }
        )
        .get_mask(dehardsub)
        .std.Maximum()
        .std.BoxBlur()
    )
    return dehardsub.std.MaskedMerge(src, custom_dehardsub_mask)


filterchain_results = filterchain(
    op=OP,
    ed=ED,
    arial_signs=ARIAL_SIGNS,
    good_signs=GOOD_SIGNS,
    jp_src_path=sources[3].jp_src_path,
    dub_src_path=sources[3].dub_src_path,
    custom_dehardsub=custom_dehardsub,
)

if not is_preview():
    mux(episode=EPISODE, op=OP, ed=ED, filterchain_results=filterchain_results)
