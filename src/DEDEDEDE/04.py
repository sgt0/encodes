from __future__ import annotations

from vsmasktools import BoundingBox
from vspreview import is_preview
from vstools import core, replace_ranges, vs

from dededede_common import filterchain, mux, sources

core.set_affinity(22, 2 << 13)


EPISODE = "04"
OP = (2578, 2578 + 2095)
ED = (32144, 32144 + 2124)

# Ranges with Arial signs.
ARIAL_SIGNS = [
    (1170, 1217),
    (6097, 6132),
    (9476, 9523),
    (14806, 14919),
    (20441, 20500),  # motors
    (21262, 21321),
    (21358, 21630),
    (21823, 21909),
    (22036, 22095),
    (22174, 22251),
    (24593, 24644),  # Fade.
    (24748, 24770),
    (25596, 25628),
]

# Ranges with good signs.
GOOD_SIGNS = [
    (66, 452),
    (597, 920),
    (2398, 2469),
    (2727, 2832),
    (4617, 4674),
    (13772, 14023),
    (14372, 14449),
    (26127, 26210),
]


def custom_dehardsub(src: vs.VideoNode, dehardsub: vs.VideoNode) -> vs.VideoNode:
    # (24593, 24644) Fade.
    faded = BoundingBox((865, 355), (860, 440)).apply_mask(dehardsub, src)
    return replace_ranges(dehardsub, faded, (24593, 24644))


filterchain_results = filterchain(
    op=OP,
    ed=ED,
    arial_signs=ARIAL_SIGNS,
    good_signs=GOOD_SIGNS,
    jp_src_path=sources[4].jp_src_path,
    dub_src_path=sources[4].dub_src_path,
    custom_dehardsub=custom_dehardsub,
)

if not is_preview():
    mux(episode=EPISODE, op=OP, ed=ED, filterchain_results=filterchain_results)
