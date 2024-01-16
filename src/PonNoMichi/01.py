from functools import partial
from pathlib import Path

from vodesfunc import DescaleTarget, Waifu2x_Doubler, adaptive_grain, ntype4
from vsdeband import Placebo
from vsdehalo import smooth_dering
from vsdenoise import dpir, dpir_mask, frequency_merge
from vskernels import Bilinear, Hermite
from vsmasktools import KirschTCanny, dre_edgemask
from vsmuxtools import (
    Setup,
    TmdbConfig,
    VideoFile,
    do_audio,
    mux,
    settings_builder_x265,
    src_file,
    x265,
)
from vspreview import is_preview
from vsscale import descale_detail_mask
from vssource import source
from vstools import (
    Keyframes,
    core,
    depth,
    finalize_clip,
    replace_ranges,
    set_output,
    vs,
)

import sgtfunc

core.set_affinity(22, 2 << 13)

OP = (10093, 12251)
ED = (32584, 34740)
GUEST_ILLUSTRATION = (35101, 35172)
DEBLOCK = [
    OP,
    ED,
]
NO_DESCALE = [
    (10239, 10459),  # Pixel art.
    (11880, 12180),  # Pixel art.
    (12181, 12250),  # Message bubbles.
    (34996, 35100),  # Next episode title.
    GUEST_ILLUSTRATION,
]
NO_DERING = [
    (34741, 34995),  # Next episode preview.
]

# fmt: off
ADN_SCENES = frozenset([
    6, 28, 36, 38, 47, 52, 54, 55, 56, 57, 60, 63, 68, 69, 99, 140, 141, 148,
    150, 152, 154, 155, 159, 166, 184, 186, 188, 193, 194, 204, 206, 214, 217,
    221, 222, 239, 251, 254, 260, 264, 268, 269, 273, 274, 280, 284, 294, 297,
    299, 310, 315, 316, 317, 318, 327, 328, 338, 348, 349, 350
])
AMZN_SCENES = frozenset([
    0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 26, 30,
    39, 43, 45, 46, 49, 53, 65, 67, 81, 83, 84, 85, 86, 98, 100, 101, 103, 105,
    107, 108, 110, 111, 112, 116, 117, 119, 120, 122, 123, 124, 125, 126, 127,
    130, 133, 134, 135, 136, 138, 149, 153, 156, 167, 177, 191, 213, 219, 227,
    229, 230, 231, 232, 234, 241, 242, 244, 249, 250, 267, 275, 276, 277, 279,
    283, 295, 301, 306, 329, 330, 331, 332, 333, 334, 336, 339, 341, 342, 343,
    352
])
BGLOBAL_SCENES = frozenset([
    121,  # This scene is horrendous in the other sources.
])
MERGE_SCENES = frozenset([
    12, 21, 22, 23, 24, 25, 27, 29, 31, 32, 33, 34, 35, 37, 40, 41, 42, 44, 48,
    50, 51, 58, 59, 61, 62, 64, 66, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
    82, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 102, 104, 106, 109, 113,
    114, 115, 118, 128, 129, 131, 132, 137, 139, 142, 143, 144, 145, 146, 147,
    151, 157, 158, 160, 161, 162, 163, 164, 165, 168, 169, 170, 171, 172, 173,
    174, 175, 176, 178, 179, 180, 181, 182, 183, 185, 187, 189, 190, 192, 195,
    196, 197, 198, 199, 200, 201, 202, 203, 205, 207, 208, 209, 210, 211, 212,
    215, 216, 218, 220, 223, 224, 225, 226, 228, 233, 235, 236, 237, 238, 240,
    243, 245, 246, 247, 248, 252, 253, 255, 256, 257, 258, 259, 261, 262, 263,
    265, 266, 270, 271, 272, 278, 281, 282, 285, 286, 287, 288, 289, 290, 291,
    292, 293, 296, 298, 300, 302, 303, 304, 305, 307, 308, 309, 311, 312, 313,
    314, 319, 320, 321, 322, 323, 324, 325, 326, 335, 337, 340, 344, 345, 346,
    347, 351, 353
])
# fmt: on


amzn_file = src_file(
    r"X:\path\to\Pon no Michi - 01 (Amazon dAnime CBR 1080p).mkv",
    idx=source,
)
amzn = amzn_file.init_cut()
keyframes = Keyframes.unique(amzn, "01 AMZN")
amzn = keyframes.to_clip(amzn, scene_idx_prop=True)

adn_file = src_file(
    r"X:\path\to\Pon no Michi - 01 (ADN 1080p).mkv",
    idx=source,
)
adn = adn_file.init_cut()
adn = Keyframes.unique(adn, "01 ADN").to_clip(adn, scene_idx_prop=True)
adn = adn + adn.std.BlankClip(length=len(amzn) - len(adn))

bglobal = src_file(
    r"X:\path\to\Pon no Michi - 01 (B-Global HEVC 1080p).mkv",
    idx=source,
)
bglobal = bglobal.init_cut()
bglobal = Keyframes.unique(bglobal, "01 BGLOBAL").to_clip(bglobal, scene_idx_prop=True)

merged = frequency_merge([adn, amzn, bglobal]).std.SetFrameProp("SceneMerged", True)


def scene_select(
    n: int,
    keyframes: Keyframes,
    adn_scenes: frozenset[int],
    amzn_scenes: frozenset[int],
    bglobal_scenes: frozenset[int],
    adn: vs.VideoNode,
    amzn: vs.VideoNode,
    bglobal: vs.VideoNode,
    merged: vs.VideoNode,
) -> vs.VideoNode:
    scene_idx = keyframes.scenes.indices[n]
    if scene_idx in adn_scenes:
        return adn
    elif scene_idx in amzn_scenes:
        return amzn
    elif scene_idx in bglobal_scenes:
        return bglobal
    else:
        return merged


selected = amzn.std.FrameEval(
    # All this stuff is passed as arguments in order to please vspreview
    # reloading.
    partial(
        scene_select,
        keyframes=keyframes,
        adn_scenes=ADN_SCENES,
        amzn_scenes=AMZN_SCENES,
        bglobal_scenes=BGLOBAL_SCENES,
        adn=adn,
        amzn=amzn,
        bglobal=bglobal,
        merged=merged,
    ),
    clip_src=[amzn, adn, bglobal, merged],
)
selected = replace_ranges(selected, amzn, GUEST_ILLUSTRATION)

# Deblock
deblocked = dpir(selected, cuda="trt", strength=dpir_mask(selected, lines=40, relative=True))
deblocked = replace_ranges(selected, deblocked, DEBLOCK)

# Rescale
deblocked_32 = depth(deblocked, 32)
dt = DescaleTarget(
    height=954,
    kernel=Bilinear,
    border_handling=1,
    upscaler=Waifu2x_Doubler(cuda="trt"),
    line_mask=(KirschTCanny, Bilinear, None, None),
    downscaler=Hermite(linear=True),
    credit_mask=False,
).generate_clips(deblocked_32)
dt.credit_mask = descale_detail_mask(
    deblocked_32, dt.rescale, thr=0.07, inflate=8, xxpand=(12, 17)
)
upscaled = dt.get_upscaled(deblocked_32)
upscaled = replace_ranges(upscaled, deblocked_32, NO_DESCALE)
upscaled = depth(upscaled, 16)

# Denoise
denoised = sgtfunc.denoise(upscaled, sigma=0.76, strength=0.26, thSAD=146, tr=3)

# Dering
dering = smooth_dering(denoised, mrad=2, thr=7, darkthr=4, contra=True)
dering = replace_ranges(dering, denoised, NO_DERING)

# Deband
debanded = Placebo.deband(dering, thr=1.9, iterations=16)
debanded = debanded.std.MaskedMerge(dering, dre_edgemask(dering, brz=12 / 255))

# Regrain
grained = adaptive_grain(debanded, [2.0, 0.4], 3.0, temporal_average=50, seed=234176, **ntype4)

final = finalize_clip(grained)


if is_preview():
    set_output(adn, "ADN")
    set_output(amzn, "AMZN")
    set_output(bglobal, "B-Global (HEVC)")
    set_output(final, "filter")
else:
    setup = Setup("01")

    # Video
    settings = settings_builder_x265(
        preset="placebo",
        crf=13.5,
        bframes=12,
        rd=3,
        rect=False,
        ref=5,
        limit_refs=1,
        merange=57,
    )
    encoded = Path(setup.work_dir).joinpath("encoded.265").resolve()
    video = (
        VideoFile(encoded)
        if encoded.exists()
        else x265(
            settings,
            zones=[(OP[0], OP[1], 1.2), (ED[0], ED[1], 1.2)],
            qp_clip=selected,
            resumable=False,
        ).encode(final)
    )

    # Audio
    audio = do_audio(amzn_file, track=0)

    mux(
        video.to_track("WEB encode by sgt", "jpn", default=True, forced=False),
        audio.to_track("E-AC-3 2.0", "jpn", default=True, forced=False),
        tmdb=TmdbConfig(
            234176,
            write_date=True,
            write_ids=True,
            write_summary=True,
            write_synopsis=True,
            write_title=True,
        ),
    )
