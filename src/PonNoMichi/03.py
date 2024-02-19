from pathlib import Path

from vodesfunc import DescaleTarget, Waifu2x_Doubler, adaptive_grain, ntype4
from vsdeband import Placebo
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
    SceneBasedDynamicCache,
    core,
    depth,
    finalize_clip,
    join,
    replace_ranges,
    set_output,
    vs,
)

import sgtfunc

core.set_affinity(22, 2 << 13)

EPISODE = "03"
OP = (984, 3140)
ED = (32583, 34739)
NEXT_EPISODE_TITLE = (35011, 35099)
GUEST_ILLUSTRATION = (35100, 35171)
NO_DESCALE = [
    (OP[0] + 146, OP[0] + 366),  # Pixel art.
    (OP[0] + 1787, OP[0] + 2087),  # Pixel art.
    (OP[0] + 2088, OP[0] + 2157),  # Message bubbles.
    NEXT_EPISODE_TITLE,
    GUEST_ILLUSTRATION,
]
NO_DPIR = [
    OP,
    ED,
    NEXT_EPISODE_TITLE,
    GUEST_ILLUSTRATION,
]

# fmt: off
ADN_SCENES = frozenset([
    49, 51, 53, 58, 59, 67, 74, 82, 83, 84, 87, 91, 92, 102, 109, 110, 114,
    117, 120, 130, 135, 142, 143, 144, 145, 150, 152, 157, 161, 163, 164,
    174, 185, 187, 192, 196, 200, 211, 214, 217, 219, 222, 229, 230, 231,
    233, 234, 235, 239, 244, 245, 246, 251, 273, 277, 278, 279, 281, 282,
    283
])
AMZN_SCENES = frozenset([
    1, 2, 3, 4, 5, 9, 57, 77, 79, 81, 85, 99, 101, 105, 112, 121, 133,
    154, 155, 181, 183, 186, 221, 236, 247, 248, 249, 276, 280
])
BGLOBAL_SCENES = frozenset([
    *range(11, 46),  # OP
    *range(256, 273),  # ED
])
# fmt: on


adn_file = src_file(
    r"X:\path\to\Pon no Michi - 03 (ADN 1080p).mkv",
    idx=source,
)
adn = adn_file.init_cut().std.SetFrameProp("SceneSource", data="ADN")

amzn_file = src_file(
    r"X:\path\to\Pon no Michi - 03 (Amazon dAnime CBR 1080p).mkv",
    idx=source,
)
amzn = amzn_file.init_cut().std.SetFrameProp("SceneSource", data="AMZN")
keyframes = Keyframes.unique(amzn, EPISODE)
amzn = keyframes.to_clip(amzn, scene_idx_prop=True)

bglobal_file = src_file(
    r"X:\path\to\Pon no Michi - 03 (B-Global HEVC 1080p).mkv",
    idx=source,
)
bglobal = bglobal_file.init_cut().std.SetFrameProp("SceneSource", data="B-Global")

bglobal_4k_file = src_file(
    r"X:\path\to\Pon no Michi - 03 (B-Global 2160p).mkv",
    idx=source,
)
bglobal_4k = bglobal_4k_file.init_cut().std.SetFrameProp("SceneSource", data="B-Global 4k")
bglobal_4k = Hermite(linear=True).scale(bglobal_4k, 1920, 1080)

bglobal_4k_hevc_file = src_file(
    r"X:\path\to\Pon no Michi - 03 (B-Global HEVC 2160p).mkv",
    idx=source,
)
bglobal_4k_hevc = bglobal_4k_hevc_file.init_cut()
bglobal_4k_hevc = Hermite(linear=True).scale(bglobal_4k_hevc, 1920, 1080)


merged = frequency_merge([adn, amzn, bglobal], planes=0).std.SetFrameProp("SceneSource", data="merged")


class PonNoMichiScenes(SceneBasedDynamicCache):
    def get_clip(self, scene_idx: int) -> vs.VideoNode:
        if scene_idx in ADN_SCENES:
            return adn
        if scene_idx in AMZN_SCENES:
            return amzn
        if scene_idx in BGLOBAL_SCENES:
            return bglobal
        return merged


selected = PonNoMichiScenes.from_clip(amzn, keyframes)
selected = keyframes.to_clip(selected, scene_idx_prop=True)


# Rescale
src_32 = depth(selected, 32)
dt = DescaleTarget(
    height=954,
    kernel=Bilinear,
    border_handling=1,
    upscaler=Waifu2x_Doubler(cuda="trt"),
    line_mask=(KirschTCanny, Bilinear, None, None),
    downscaler=Hermite(linear=True),
    credit_mask=False,
).generate_clips(src_32)
dt.credit_mask = descale_detail_mask(src_32, dt.rescale, thr=0.035, inflate=6, xxpand=(14, 14))
upscaled = dt.get_upscaled(src_32)
upscaled = depth(upscaled, 16)
upscaled = join(replace_ranges(upscaled, selected, NO_DESCALE), bglobal_4k)

# Denoise
denoised = dpir(upscaled, cuda="trt", strength=dpir_mask(upscaled, low=10, high=20))
denoised = replace_ranges(denoised, upscaled, NO_DPIR)
denoised = join(denoised, upscaled)
denoised = sgtfunc.denoise(denoised, sigma=0.7, strength=0.24, thSAD=95, tr=3)

# Deband
debanded = Placebo.deband(denoised, thr=1.9, iterations=16)
debanded = debanded.std.MaskedMerge(denoised, dre_edgemask(denoised, brz=16 / 255))

# Regrain
grained = adaptive_grain(debanded, [2.0, 0.4], 3.0, temporal_average=50, seed=234176, **ntype4)

final = finalize_clip(grained)


if is_preview():
    set_output(adn, "ADN")
    set_output(amzn, "AMZN")
    set_output(bglobal, "B-Global")
    set_output(bglobal_4k, "B-Global (4k)")
    set_output(bglobal_4k_hevc, "B-Global (HEVC 4k)")
    set_output(final, "filter")
else:
    setup = Setup(EPISODE)

    # Video
    settings = settings_builder_x265(
        preset="placebo",
        crf=13.2,
        bframes=12,
        rd=3,
        rect=False,
        ref=5,
        qcomp=0.72,
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
            resumable=False,
        ).encode(final)
    )

    # Audio
    audio = do_audio(amzn_file)

    mux(
        video.to_track("WEB encode by sgt", "jpn", default=True, forced=False, args=["--deterministic", "234176"]),
        audio.to_track("E-AC-3 2.0", "jpn", default=True, forced=False),
        tmdb=TmdbConfig(234176, write_date=True, write_ids=True),
    )
