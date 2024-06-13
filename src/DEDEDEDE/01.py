from __future__ import annotations

from pathlib import Path

from vskernels import Point
from vsmasktools import CustomMaskFromRanges
from vspreview import is_preview
from vssource import source
from vstools import Matrix, core, vs

from dededede_common import filterchain, mux, sources

core.set_affinity(22, 2 << 13)


EPISODE = "01"
OP = (6558, 6558 + 2095)
ED = (32144, 32144 + 2124)

# Ranges with Arial signs.
ARIAL_SIGNS = [
    (1138, 1221),
    (11045, 11143),
    (14079, 14173),
    (18102, 18170),
    (20613, 20768),
    (24692, 24739),
    (25148, 25219),
]

# Ranges with good signs.
GOOD_SIGNS = [
    (6378, 6449),
    (6707, 6812),
    (8597, 8653),
    (8708, 8755),
    (9338, 9457),
    (9587, 9670),
    (10871, 10972),  # A character goes missing here, RIP.
    (11144, 11251),
    (11528, 11602),
    (11807, 12301),
    # JP hardsubs. Not "good" signs but technically get the same treatment as
    # the rest of these.
    (12932, 12985),
    (12992, 13024),
    (13033, 13090),
    # End JP hardsubs.
    (13457, 13783),
    (13784, 13843),  # + special handling.
    (13844, 13981),
    (14174, 14239),
    (15384, 15485),
    (18762, 18887),
    (19821, 19892),  # + special handling.
    (22985, 23050),
    (23111, 23296),
]


def custom_dehardsub(src: vs.VideoNode, dehardsub: vs.VideoNode) -> vs.VideoNode:
    custom_dehardsub_mask = (
        CustomMaskFromRanges(
            ranges={
                Path("masks/01_12.png"): (12, 131),  # Fade.
                Path("masks/01_1462.png"): (1462, 1524),  # Fade.
                Path("masks/01_9459.png"): (9459, 9511),  # Fade.
                Path("masks/01_13784.png"): (13784, 13843),  # Both signs together.
                Path("masks/01_19821.png"): (19821, 19892),  # Both signs together.
            }
        )
        .get_mask(dehardsub)
        .std.Maximum()
        .std.BoxBlur()
    )
    dehardsub = dehardsub.std.MaskedMerge(src, custom_dehardsub_mask)

    # Hand-cleaned scene where both types of signs overlapped.
    clean = source(Path("masks/01_19821_clean.png"))
    clean = Point.resample(clean, format=dehardsub.format, matrix=Matrix.from_video(dehardsub))
    clean_mask = (
        CustomMaskFromRanges(ranges={Path("masks/01_19821_clean_mask.png"): (19821, 19892)})
        .get_mask(dehardsub)
        .std.Maximum()
        .std.Maximum()
        .std.Maximum()
        .std.BoxBlur()
    )
    return dehardsub.std.MaskedMerge(clean, clean_mask)


filterchain_results = filterchain(
    ed=ED,
    arial_signs=ARIAL_SIGNS,
    good_signs=GOOD_SIGNS,
    jp_src_path=sources[1].jp_src_path,
    dub_src_path=sources[1].dub_src_path,
    custom_dehardsub=custom_dehardsub,
)

if not is_preview():
    mux(episode=EPISODE, op=OP, ed=ED, filterchain_results=filterchain_results)
