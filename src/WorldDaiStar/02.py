from pathlib import Path

from vodesfunc import DescaleTarget, Waifu2x_Doubler, adaptive_grain, ntype4
from vsdeband import Placebo
from vsexprtools import norm_expr
from vskernels import Bilinear, Hermite
from vsmasktools import KirschTCanny, diff_creditless_oped, dre_edgemask
from vsmuxtools import (
    FLAC,
    Chapters,
    Setup,
    SubFile,
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
from vstools import Keyframes, core, depth, finalize_clip, set_output

import sgtfunc

core.set_affinity(22, 2 << 13)


EPISODE = "02"
OP = (1296, 3453)
ED = (31771, 33928)


jpnbd = src_file(
    r"X:\path\to\WORLD_DAI_STAR_1\BDMV\STREAM\00001.m2ts",
    idx=source,
)
src = jpnbd.init_cut()
keyframes = Keyframes.unique(src, EPISODE)
src = keyframes.to_clip(src, scene_idx_prop=True)

ncop = src_file(
    r"X:\path\to\WORLD_DAI_STAR_1\BDMV\STREAM\00006.m2ts",
    idx=source,
    trim=(24, 24 + OP[1] - OP[0] + 1),
)
ncop = ncop.init_cut()

nced = src_file(
    r"X:\path\to\WORLD_DAI_STAR_1\BDMV\STREAM\00007.m2ts",
    idx=source,
    trim=(24, 24 + ED[1] - ED[0] + 1),
)
nced = nced.init_cut()


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

detail_mask = descale_detail_mask(src_32, dt.rescale, thr=0.0314, inflate=7, xxpand=(2, 1))
oped_credit_mask = diff_creditless_oped(
    src,
    ncop=ncop,
    nced=nced,
    thr=0.81,
    opstart=OP[0],
    opend=OP[1],
    edstart=ED[0],
    edend=ED[1],
    prefilter=True,
)
oped_credit_mask = depth(oped_credit_mask, 32)
dt.credit_mask = norm_expr([detail_mask, oped_credit_mask], "x y +").std.Limiter()

upscaled = dt.get_upscaled(src_32)
upscaled = depth(upscaled, 16)

# Letterbox at scene 201 (19025, 19360) but it looks fine as-is.

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
    setup = Setup(EPISODE)

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
    video = (
        VideoFile(encoded)
        if encoded.exists()
        else x265(
            settings,
            zones=[(OP[0], OP[1], 1.2), (ED[0], ED[1], 1.2)],
            qp_clip=src,
            add_props=True,
            resumable=False,
        ).encode(final)
    )

    # Audio
    audio = do_audio(jpnbd, encoder=FLAC())

    # Subs
    subs = (
        SubFile(rf"X:\path\to\{setup.episode}.ass")
        .truncate_by_video(final)
        .clean_styles()
        .clean_garbage()
    )
    subs_enm = subs.copy().autoswapper(allowed_styles=None)
    fonts = subs.collect_fonts()

    # Chapters
    chapters = Chapters(jpnbd).set_names(["Prologue", "Opening", "Part A", "Part B", "Ending", "Preview"])

    mux(
        video.to_track("JPNBD encode by sgt", "jpn", default=True, forced=False, args=["--deterministic", "215198"]),
        audio.to_track("FLAC 2.0", "jpn", default=True, forced=False),
        subs.to_track("Full Subtitles [sgt]", "eng", default=True, forced=False),
        subs_enm.to_track("Honorifics [sgt]", "enm", default=True, forced=False),
        *fonts,
        chapters,
        tmdb=TmdbConfig(
            215198,
            write_cover=True,
            write_date=True,
            write_ids=True,
            write_summary=True,
            write_synopsis=True,
            write_title=True,
        ),
    )
