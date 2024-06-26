from pathlib import Path

from vodesfunc import adaptive_grain, ntype4, set_output
from vsaa import based_aa
from vsdeband import Placebo
from vsdehalo import dehalomicron
from vsdenoise import Prefilter
from vsmasktools import dre_edgemask
from vsmuxtools import (
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
from vsscale import FSRCNNXShader
from vssource import source
from vstools import Sar, core, finalize_clip, replace_ranges

import sgtfunc

core.set_affinity(22, 2 << 13)

OP = (696, 2854)
ED = (30475, 32634)

NO_AA = [
    (11268, 11495),  # Laptop
]

cr = src_file(
    r"X:\path\to\stardust-telepath\[SubsPlease] Hoshikuzu Telepath - 11 (1080p) [138C1FB6].mkv",
    idx=source,
)
src = cr.init_cut()

# Denoise
denoised = sgtfunc.denoise(src, strength=0.31, tr=3)

# AA
aa = based_aa(denoised, 1.5, supersampler=FSRCNNXShader.x56)
aa = replace_ranges(aa, denoised, NO_AA)

# Native resolution is ~900p but nevertheless it's not descaleable.

# Dehalo
dehaloed = dehalomicron(
    aa,
    brz=10 / 255,
    sigma=0.99,
    sigma0=0.28,
    dampen=(0.65, True),
    blur_func=Prefilter.GAUSSBLUR2,
)

# dehalomicron changes the SAR to 9435136:9389601, so change it back.
dehaloed = Sar(1, 1).apply(dehaloed)

# Deband
debanded = Placebo.deband(dehaloed, thr=2, iterations=16)
debanded = debanded.std.MaskedMerge(dehaloed, dre_edgemask(dehaloed, brz=11 / 255))

# Regrain
grained = adaptive_grain(debanded, strength=[1.9, 0.4], size=3.3, temporal_average=50, seed=217404, **ntype4)
final = finalize_clip(grained)


if is_preview():
    set_output(src, "CR")
    set_output(denoised, "denoise")
    set_output(aa, "aa")
    set_output(dehaloed, "dehalo")
    set_output(final, "filter")
else:
    setup = Setup("11")
    assert setup.work_dir

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
    video_hevc = (
        VideoFile(encoded)
        if encoded.exists()
        else x265(settings, zones=[(OP[0], OP[1], 1.2), (ED[0], ED[1], 1.2)], qp_clip=src, resumable=False).encode(
            final
        )
    )

    # Audio
    audio = do_audio(cr)

    # Subs
    subs = (
        SubFile(rf"X:\path\to\stardust-telepath\{setup.episode}.ass")
        .truncate_by_video(final)
        .clean_styles()
        .clean_garbage()
    )
    fonts = subs.collect_fonts()

    # Chapters
    chapters = Chapters(
        [
            (0, "Prologue"),
            (OP[0], "Opening"),
            (OP[1], "Part A"),
            (14627, "Part B"),
            (ED[0], "Ending"),
            (ED[1], "Epilogue"),
            (33808, "Preview"),
        ],
        final.fps,
    )

    mux(
        video_hevc.to_track("WEB encode by sgt", "jpn", default=True, forced=False),
        audio.to_track("AAC 2.0", "jpn", default=True, forced=False),
        subs.to_track("Full Subtitles [RRA]", "eng", default=True, forced=False),
        *fonts,
        chapters,
        tmdb=TmdbConfig(
            217404,
            write_cover=True,
            write_date=True,
            write_ids=True,
            write_title=True,
        ),
    )
