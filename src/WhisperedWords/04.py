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
from vsexprtools import norm_expr
from vskernels import Bilinear, Mitchell
from vsmasktools import Kirsch, diff_creditless_oped, dre_edgemask
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
from vstools import FieldBased, core, depth, finalize_clip, iterate

import sgtfunc

core.set_affinity(16, 17180)

# Frame range (inclusive) of the OP.
OP = (696, 2855)

# Frame range (inclusive) of the ED.
ED = (31536, 33695)

JPNBD = src_file(
    r"X:\path\to\JPNBD\Disc1\BDMV\STREAM\00003.m2ts",
    idx=source(field_based=FieldBased.PROGRESSIVE),
)
preview = src_file(
    r"X:\path\to\JPNBD\Disc1\BDMV\STREAM\00004.m2ts",
    idx=source(field_based=FieldBased.PROGRESSIVE),
)
src = JPNBD.init_cut() + preview.init_cut()

NCOP = src_file(
    r"X:\path\to\JPNBD\Disc2\BDMV\STREAM\00019.m2ts",
    idx=source(field_based=FieldBased.PROGRESSIVE),
    trim=(24, 24 + OP[1] - OP[0] + 1),
)
ncop = NCOP.init_cut()


# Rescale
dt = DescaleTarget(
    height=719.8,
    base_height=720,
    kernel=Mitchell,
    upscaler=Clamped_Doubler(False, Waifu2x_Doubler(cuda="trt"), 70),
    line_mask=False,
    credit_mask=False,
).generate_clips(src)

oped_credit_mask = diff_creditless_oped(
    src,
    thr=0.8,
    ncop=ncop,
    nced=core.std.BlankClip(src),
    opstart=OP[0],
    opend=OP[1],
    prefilter=True,
)
oped_credit_mask = iterate(oped_credit_mask, core.std.Maximum, 8)
detail_mask = descale_detail_mask(src, dt.rescale, thr=0.045, xxpand=(20, 20))
dt.credit_mask = depth(norm_expr([depth(detail_mask, 32), depth(oped_credit_mask, 32)], "x y +"), 16)

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
debanded = Placebo.deband(aa, iterations=16)
debanded = debanded.std.MaskedMerge(aa, dre_edgemask(aa, brz=0.15))

# Regrain
grained = adaptive_grain(debanded, strength=3, size=3.5, seed=42407)
final = finalize_clip(grained)


if is_preview():
    set_output(finalize_clip(src), "src")
    set_output(final, "filter")
else:
    setup = Setup("04")

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
    audio = do_audio(JPNBD.get_audio_trimmed(0) + preview.get_audio_trimmed(0), track=0, encoder=FLAC())

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
    chapters = (
        Chapters(JPNBD)
        .set_names(["Prologue", "Opening", "Part A", "Part B", "Ending", "Epilogue"])
        .add((33984, "Preview"), 6)
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
