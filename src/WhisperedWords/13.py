from fractions import Fraction
from pathlib import Path

from vodesfunc import (
    Clamped_Doubler,
    DescaleTarget,
    Waifu2x_Doubler,
    adaptive_grain,
    set_output,
)
from vsaa import based_aa
from vsdeband import Placebo
from vskernels import Bilinear, Mitchell
from vsmasktools import Kirsch, dre_edgemask
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
from vstools import FieldBased, core, finalize_clip

import sgtfunc

core.set_affinity(16, 17180)

# Frame range (inclusive) of the ED.
ED = (31056, 33215)

# 3 episodes in one file.
JPNBD = src_file(
    r"X:\path\to\JPNBD\Disc2\BDMV\STREAM\00004.m2ts",
    idx=source(field_based=FieldBased.PROGRESSIVE),
    trim=(34344 + 34344, None),
)
src = JPNBD.init_cut()


# Rescale
dt = DescaleTarget(
    height=719.8,
    base_height=720,
    kernel=Mitchell,
    upscaler=Clamped_Doubler(False, Waifu2x_Doubler(cuda="trt"), 70),
    line_mask=False,
    credit_mask=False,
).generate_clips(src)

dt.credit_mask = descale_detail_mask(src, dt.rescale, thr=0.045, xxpand=(20, 20))

# Generate doubled clip for line mask.
dt.get_upscaled(src)
dt.line_mask = Kirsch.edgemask(dt.doubled, 80 / 250, 150 / 250, planes=0).std.Maximum().std.Inflate()
dt.line_mask = Bilinear.scale(dt.line_mask, src.width, src.height).std.Limiter()

upscaled = dt.generate_clips(src).get_upscaled(src)

# Denoise
denoised = sgtfunc.denoise(
    upscaled,
    limit=255,
    sigma=0.8,
    strength=0.6,
    thSAD=40,
    tr=3,
)

# AA
aa = based_aa(denoised, 1.5)

# Deband
debanded = Placebo.deband(aa, thr=2, iterations=16)
debanded = debanded.std.MaskedMerge(aa, dre_edgemask(aa, brz=0.15))

# Regrain
grained = adaptive_grain(debanded, strength=3, size=3.5, seed=42407)
final = finalize_clip(grained)


if is_preview():
    set_output(finalize_clip(src), "src")
    set_output(final, "filter")
else:
    setup = Setup("13")

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
    video_hevc = VideoFile(encoded) if encoded.exists() else x265(settings, qp_clip=src).encode(final)

    # Audio
    audio = do_audio(JPNBD, track=0, encoder=FLAC())

    # Subs
    subs_tsundere = (
        SubFile(rf"X:\path\to\subs\{setup.episode}\{setup.episode}-tsundere-drag-sgt.ass")
        .truncate_by_video(final)
        .clean_styles()
        .clean_garbage()
    )
    subs_shingx = (
        SubFile(rf"X:\path\to\subs\{setup.episode}\{setup.episode}-shin-gx-sgt.ass")
        .truncate_by_video(final)
        .clean_styles()
        .clean_garbage()
    )
    subs_tsundere.collect_fonts()
    fonts = subs_shingx.collect_fonts()

    # Chapters
    # Manually defined because muxtools is not equipped to handle 3 episodes
    # being in one file and the 19 chapters they "share." The timestamps of
    # these chapters are past the length of this episode so they get discarded
    # even before muxtools trims them.
    chapters = Chapters(
        [
            (0, "Prologue"),
            (69720 - 68688, "Part A"),
            (85056 - 68688, "Part B"),
            (ED[0], "Ending"),
            (ED[1], "Epilogue"),
        ],
        Fraction(final.fps_num, final.fps_den),
    )

    mux(
        video_hevc.to_track("JPNBD encode by sgt", "jpn", default=True, forced=False),
        audio.to_track("FLAC 2.0", "jpn", default=True, forced=False),
        subs_tsundere.to_track("Full Subtitles [Tsundere]", "eng", default=True, forced=False),
        subs_shingx.to_track("Full Subtitles [SHiN-gx]", "eng", default=False, forced=False),
        *fonts,
        chapters,
        tmdb=TmdbConfig(
            42407,
            write_cover=True,
            write_date=True,
            write_ids=True,
            write_summary=True,
            write_synopsis=True,
            write_title=True,
        ),
    )
