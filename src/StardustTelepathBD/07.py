from pathlib import Path

from awsmfunc import bbmod
from rekt.rektlvls import rektlvls
from vodesfunc import adaptive_grain, ntype4, set_output
from vsaa import based_aa
from vsdeband import Placebo
from vsdehalo import dehalo_sigma, fine_dehalo
from vsdenoise import Prefilter
from vsexprtools import ExprOp, norm_expr
from vsmasktools import CustomMaskFromRanges, diff_creditless_oped, dre_edgemask
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
from vstools import Keyframes, core, finalize_clip, replace_ranges

import sgtfunc

core.set_affinity(22, 2 << 13)


EPISODE = "07"
ED = (30041, 30041 + 2158)
WEIRD_EDGEFIX = (15316, 15333)


cr = src_file(
    r"X:\path\to\stardust-telepath\[SubsPlease] Hoshikuzu Telepath - 07 (1080p) [B5C05DCB].mkv",
    idx=source,
).init_cut()
cr = cr.std.BlankClip(length=24) + cr

jpnbd = src_file(
    r"X:\path\to\HOSHIKUZU_TELEPATH_VOL2\BDMV\STREAM\00000.m2ts",
    idx=source,
)
src = jpnbd.init_cut()
keyframes = Keyframes.unique(src, EPISODE)
src = keyframes.to_clip(src, scene_idx_prop=True)

nced = src_file(
    r"X:\path\to\HOSHIKUZU_TELEPATH_VOL1\BDMV\STREAM\00009.m2ts",
    idx=source,
    trim=(24, -24),
).init_cut()


# Edgefix the one scene with dirty borders and a dirty column 4.
edgefix = bbmod(src, 1, 1, 1, 1, blur=1, planes=0)
edgefix = rektlvls(edgefix, colnum=[0, 4], colval=[9, 7])
edgefix = replace_ranges(src, edgefix, WEIRD_EDGEFIX)

# Denoise
denoised = sgtfunc.denoise(edgefix, sigma=0.52, strength=0.26, thSAD=101, tr=3)

# AA
aa = based_aa(denoised, rfactor=2.0, supersampler=Waifu2x(cuda="trt"))

# Dehalo
dehaloed = dehalo_sigma(aa, blur_func=Prefilter.GAUSSBLUR2, pre_ss=2)
bilateref = bilateral(aa, ref=dehaloed, sigmaS=6, sigmaR=5 / 255, gpu=True)
clamped = norm_expr([aa, dehaloed, bilateref], "x y z max min")
dehaloed = aa.std.MaskedMerge(clamped, fine_dehalo.mask(aa))

# Credit mask
op_credit_mask = CustomMaskFromRanges(
    ranges={
        Path("masks/07_op_credits_00.png"): [(30, 119)],
        Path("masks/07_op_credits_01.png"): [(139, 197)],
        Path("masks/07_op_credits_02.png"): [(221, 279)],
        Path("masks/07_op_credits_03.png"): [(305, 363)],
        Path("masks/07_op_credits_04.png"): [(392, 444)],
        Path("masks/07_op_credits_05.png"): [(451, 503)],
        Path("masks/07_op_credits_06.png"): [(515, 567)],
        Path("masks/07_op_credits_07.png"): [(577, 623)],
    }
).get_mask(src)
ed_credit_mask = diff_creditless_oped(
    src, ncop=nced.std.BlankClip(), nced=nced, thr=0.3, edstart=ED[0], edend=ED[1], prefilter=True
)
credit_mask = ExprOp.MAX.combine(op_credit_mask, ed_credit_mask)
dehaloed = dehaloed.std.MaskedMerge(denoised, credit_mask)

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
    set_output(credit_mask, "credit mask")
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
        else x265(settings, qp_clip=src, zones=[(ED[0], ED[1], 1.2)], add_props=True, resumable=False).encode(final)
    )

    # Audio
    audio = do_audio(jpnbd, encoder=FLAC())

    # Subs
    subs = (
        SubFile(rf"X:\path\to\stardust-telepath\{setup.episode}.ass")
        .shift(24, final.fps)
        .truncate_by_video(final)
        .clean_styles()
        .clean_garbage()
    )
    fonts = subs.collect_fonts()

    # Chapters
    chapters = Chapters(jpnbd).set_names(["Part A", "Part B", "Part C", "Ending", "Epilogue", "Preview"])

    mux(
        video.to_track("JPNBD encode by sgt", "jpn", default=True, forced=False, args=["--deterministic", "217404"]),
        audio.to_track("FLAC 2.0", "jpn", default=True, forced=False),
        subs.to_track("Full Subtitles [RRA]", "eng", default=True, forced=False),
        *fonts,
        chapters,
        tmdb=TmdbConfig(217404, write_cover=True, write_date=True, write_ids=True, write_title=True),
    )
