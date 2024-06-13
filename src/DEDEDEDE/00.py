from __future__ import annotations

from vspreview import is_preview
from vstools import core, vs

from dededede_common import filterchain, mux, sources

core.set_affinity(22, 2 << 13)


EPISODE = "00"
OP = (32188, 32188 + 2095)

# Ranges with Arial signs.
ARIAL_SIGNS = [
    (2040, 2090),
    (2523, 2618),
    (6648, 6722),
    (6939, 7067),
    (8847, 9038),
    (9906, 10076),
    (14949, 15116),
    (15633, 15752),
    (21290, 21349),
    (22331, 22402),
    (23975, 24040),
    (25340, 25477),
    (25737, 25844),
]

# Ranges with good signs.
GOOD_SIGNS = [
    (2835, 2981),
    (3837, 3908),
    (7685, 7718),
    (14565, 14720),
    (16365, 16436),
    (32337, 32442),
    (34227, 34285),
]


def custom_dehardsub(_src: vs.VideoNode, dehardsub: vs.VideoNode) -> vs.VideoNode:
    return dehardsub


filterchain_results = filterchain(
    arial_signs=ARIAL_SIGNS,
    good_signs=GOOD_SIGNS,
    jp_src_path=sources[0].jp_src_path,
    dub_src_path=sources[0].dub_src_path,
    custom_dehardsub=custom_dehardsub,
)

if not is_preview():
    mux(episode=EPISODE, op=OP, filterchain_results=filterchain_results)
