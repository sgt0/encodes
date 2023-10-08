from fractions import Fraction
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
    TmdbConfig,
    VideoFile,
    do_audio,
    mux,
    settings_builder_x265,
    src_file,
    x265,
)
from vspreview import is_preview
from vssource import source
from vstools import Sar, core, finalize_clip

import sgtfunc

core.set_affinity(22, 2 << 13)

# Frame ranges with high motion and maybe some blocking. They won't get special
# deblocking treatment, because the existing denoise is good enough, but they
# will be zoned for more bitrate.
BLOCK_RISK: list[tuple[int, int]] = [
    (791, 908),
    (20393, 20482),
    (21947, 22078),
    (22653, 22985),
]

ABEMA = src_file(
    r"X:\path\to\[SubsPlus+] The Vexations of a Shut-In Vampire Princess - S01E01 (WEB 1080p ABEMA) [06D49CEB].mkv",
    idx=source,
)
src = ABEMA.init_cut()

# Denoise
denoised = sgtfunc.denoise(
    src,
    limit=255,
    sigma=0.85,
    thSAD=100,
    tr=3,
)

# AA
aa = based_aa(denoised, 1.5)

# Dehalo
dehaloed = dehalomicron(
    aa,
    brz=70 / 255,
    sigma=0.393,
    sigma0=0.3057,
    dampen=(0.65, True),
    blur_func=Prefilter.GAUSSBLUR2,
)

# dehalomicron changes the SAR to 9435136:9389601, don't ask me why.
dehaloed = Sar(1, 1).apply(dehaloed)

# Deband
debanded = Placebo.deband(dehaloed, thr=3, iterations=16)
debanded = debanded.std.MaskedMerge(dehaloed, dre_edgemask(dehaloed, brz=22 / 255))

# Regrain
grained = adaptive_grain(
    debanded, strength=[1.8, 0.4], size=2.8, temporal_average=50, seed=217755, **ntype4
)
final = finalize_clip(grained)


if is_preview():
    set_output(src, "src")
    set_output(final, "filter")
else:
    setup = Setup("01")

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
        else x265(settings, zones=[(x, y, 1.2) for (x, y) in BLOCK_RISK], qp_clip=src).encode(
            final
        )
    )

    # Audio
    audio = do_audio(ABEMA, track=0)

    # TODO: subs

    # Chapters
    chapters = Chapters(
        [
            (0, "Prologue"),
            (2830, "Part A"),
            (17216, "Part B"),
            (32216, "Epilogue"),
            (33688, "Preview"),
        ],
        Fraction(final.fps_num, final.fps_den),
    )

    mux(
        video_hevc.to_track("WEB encode by sgt", "jpn", default=True, forced=False),
        audio.to_track("AAC 2.0", "jpn", default=True, forced=False),
        chapters,
        tmdb=TmdbConfig(
            217755,
            write_cover=True,
            write_date=True,
            write_ids=True,
            write_summary=True,
            write_synopsis=True,
            write_title=True,
        ),
    )
