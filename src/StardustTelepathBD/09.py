from pathlib import Path

from vodesfunc import adaptive_grain, ntype4, set_output
from vsaa import based_aa
from vsdeband import Placebo
from vsdehalo import dehalo_sigma, fine_dehalo
from vsdenoise import Prefilter
from vsexprtools import norm_expr
from vsmasktools import dre_edgemask
from vsmuxtools import (
    FLAC,
    Chapters,
    Setup,
    TmdbConfig,
    VideoFile,
    do_audio,
    mux,
    settings_builder_x265,
    src_file,
    x265,
)
from vsmuxtools.extension import SubFile
from vspreview import is_preview
from vsrgtools import bilateral
from vsscale import Waifu2x
from vssource import source
from vstools import Keyframes, core, finalize_clip

import sgtfunc

core.set_affinity(22, 2 << 13)


EPISODE = "09"


cr = src_file(
    r"X:\path\to\stardust-telepath\[SubsPlease] Hoshikuzu Telepath - 09 (1080p) [77144206].mkv",
    idx=source,
).init_cut()
cr = cr.std.BlankClip(length=24) + cr
cr = cr[:5514] + cr[5514] + cr[5514:]

jpnbd = src_file(
    r"X:\path\to\HOSHIKUZU_TELEPATH_VOL2\BDMV\STREAM\00002.m2ts",
    idx=source,
)
src = jpnbd.init_cut()
keyframes = Keyframes.unique(src, EPISODE)
src = keyframes.to_clip(src, scene_idx_prop=True)


# Denoise
denoised = sgtfunc.denoise(src, sigma=0.52, strength=0.26, thSAD=101, tr=3)

# AA
aa = based_aa(denoised, rfactor=2.0, supersampler=Waifu2x(cuda="trt"))

# Dehalo
dehaloed = dehalo_sigma(aa, blur_func=Prefilter.GAUSSBLUR2, pre_ss=2)
bilateref = bilateral(aa, ref=dehaloed, sigmaS=6, sigmaR=5 / 255, gpu=True)
clamped = norm_expr([aa, dehaloed, bilateref], "x y z max min")
dehaloed = aa.std.MaskedMerge(clamped, fine_dehalo.mask(aa))

# Deband
debanded = Placebo.deband(dehaloed, thr=1.25)
debanded = debanded.std.MaskedMerge(dehaloed, dre_edgemask(dehaloed, brz=10 / 255))

# Regrain
grained = adaptive_grain(debanded, strength=[1.9, 0.4], size=3.3, temporal_average=50, seed=217404, **ntype4)
final = finalize_clip(grained)


if is_preview():
    set_output(cr, "CR")
    set_output(src, "JPNBD")
    set_output(final, "filter")
else:
    setup = Setup(EPISODE)
    assert setup.work_dir

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
        else x265(settings, qp_clip=src, add_props=True, resumable=False).encode(final)
    )

    # Audio
    audio = do_audio(jpnbd, encoder=FLAC())

    # Subs
    # These have already been shifted beforehand because not only do they need
    # the usual 24 frame shift, there is 1 additional frame around frame 5514 so
    # all lines after that moment need a 1 frame shift as well.
    subs = (
        SubFile(rf"X:\path\to\stardust-telepath\{setup.episode}.ass")
        .truncate_by_video(final)
        .clean_styles()
        .clean_garbage()
    )
    fonts = subs.collect_fonts()

    # Chapters
    chapters = Chapters(jpnbd).set_names(["Part A", "Part B", "Part C", "Preview"])

    mux(
        video.to_track("JPNBD encode by sgt", "jpn", default=True, forced=False, args=["--deterministic", "217404"]),
        audio.to_track("FLAC 2.0", "jpn", default=True, forced=False),
        subs.to_track("Full Subtitles [RRA]", "eng", default=True, forced=False),
        *fonts,
        chapters,
        tmdb=TmdbConfig(217404, write_cover=True, write_date=True, write_ids=True, write_title=True),
    )
