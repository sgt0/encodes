from pathlib import Path

from vodesfunc import DescaleTarget, Waifu2x_Doubler, adaptive_grain, ntype4
from vsdeband import Placebo
from vskernels import Bilinear, Hermite
from vsmasktools import KirschTCanny, dre_edgemask
from vsmuxtools import (
    FLAC,
    Setup,
    VideoFile,
    do_audio,
    mux,
    settings_builder_x265,
    src_file,
    x265,
)
from vspreview import is_preview
from vssource import source
from vstools import core, depth, finalize_clip, set_output

import sgtfunc

core.set_affinity(22, 2 << 13)


jpnbd = src_file(
    r"X:\path\to\WORLD_DAI_STAR_3\BDMV\STREAM\00008.m2ts",
    idx=source,
)
src = jpnbd.init_cut()


# Rescale
src_32 = depth(src, 32)
dt = DescaleTarget(
    height=720,
    kernel=Bilinear,
    border_handling=1,
    upscaler=Waifu2x_Doubler(cuda="trt"),
    line_mask=(KirschTCanny, Bilinear, None, None),
    downscaler=Hermite(linear=True),
    credit_mask=False,
).generate_clips(src_32)
upscaled = dt.get_upscaled(src_32)
upscaled = depth(upscaled, 16)

# Denoise
denoised = sgtfunc.denoise(upscaled, sigma=0.59, strength=0.29, thSAD=89, tr=3)

# Deband
debanded = Placebo.deband(denoised, thr=1.6, iterations=16)
debanded = debanded.std.MaskedMerge(denoised, dre_edgemask(denoised, brz=10 / 255))

# Regrain
grained = adaptive_grain(debanded, [1.92, 0.4], 3.11, temporal_average=50, seed=215198, **ntype4)

final = finalize_clip(grained)


if is_preview():
    set_output(src, "JPN BD")
    set_output(final, "filter")
else:
    setup = (
        Setup("NCED2")
        .edit("mkv_title_naming", "$show$ - $ep$")
        .edit("out_name", "[sgt] $show$ - $ep$ (BD 1080p HEVC FLAC) [#crc32#]")
    )

    # Video
    settings = settings_builder_x265(
        preset="placebo",
        crf=13.5,
        rd=3,
        rect=False,
        ref=5,
        bframes=12,
        qcomp=0.72,
        limit_refs=1,
        merange=57,
        keyint=round(final.fps) * 10,
    )
    encoded = Path(setup.work_dir).joinpath("encoded.265").resolve()
    video = VideoFile(encoded) if encoded.exists() else x265(settings, add_props=True, resumable=False).encode(final)

    # Audio
    audio = do_audio(jpnbd, encoder=FLAC())

    mux(
        video.to_track("JPNBD encode by sgt", "jpn", default=True, forced=False, args=["--deterministic", "215198"]),
        audio.to_track("FLAC 2.0", "jpn", default=True, forced=False),
    )
