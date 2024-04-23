from pathlib import Path

from vodesfunc import adaptive_grain, ntype4, set_output
from vsaa import based_aa
from vsdeband import Placebo
from vsdehalo import dehalo_sigma, fine_dehalo
from vsdenoise import Prefilter, dpir, dpir_mask
from vsexprtools import ExprOp, norm_expr
from vsmasktools import CustomMaskFromRanges, diff_creditless_oped, dre_edgemask
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
from vsrgtools import bilateral
from vsscale import Waifu2x
from vssource import source
from vstools import Keyframes, core, finalize_clip, replace_ranges

import sgtfunc

core.set_affinity(22, 2 << 13)


EPISODE = "03"
OP = (624, 624 + 2157)
NO_AA_DEHALO = [
    (28, 146),  # Television
    (327, 386),  # Television
]


cr = src_file(
    r"X:\path\to\stardust-telepath\[SubsPlease] Hoshikuzu Telepath - 03 (1080p) [00D25245].mkv",
    idx=source,
).init_cut()
cr = cr.std.BlankClip(length=24) + cr

jpnbd = src_file(
    r"X:\path\to\HOSHIKUZU_TELEPATH_VOL1\BDMV\STREAM\00002.m2ts",
    idx=source,
)
src = jpnbd.init_cut()
keyframes = Keyframes.unique(src, EPISODE)
src = keyframes.to_clip(src, scene_idx_prop=True)

ncop = src_file(
    r"X:\path\to\HOSHIKUZU_TELEPATH_VOL1\BDMV\STREAM\00007.m2ts",
    idx=source,
    trim=(24, -24),
).init_cut()


# Denoise
denoised = sgtfunc.denoise(src, sigma=0.52, strength=0.26, thSAD=101, tr=3)
stronger_denoise = dpir(src, strength=dpir_mask(src))
denoised = replace_ranges(denoised, stronger_denoise, OP)

# AA
aa = based_aa(denoised, rfactor=2.0, supersampler=Waifu2x(cuda="trt"))

# Dehalo
dehaloed = dehalo_sigma(aa, blur_func=Prefilter.GAUSSBLUR2, pre_ss=2)
bilateref = bilateral(aa, ref=dehaloed, sigmaS=6, sigmaR=5 / 255, gpu=True)
clamped = norm_expr([aa, dehaloed, bilateref], "x y z max min")
dehaloed = aa.std.MaskedMerge(clamped, fine_dehalo.mask(aa))
dehaloed = replace_ranges(dehaloed, denoised, NO_AA_DEHALO)

# Credit mask
op_credit_mask = diff_creditless_oped(
    src, ncop=ncop, nced=ncop.std.BlankClip(), thr=0.3, opstart=OP[0], opend=OP[1], prefilter=True
)
ed_custom_mask = CustomMaskFromRanges(
    ranges={
        Path("masks/03_ed_credits_00.png"): [(28906, 28989)],
        Path("masks/03_ed_credits_01.png"): [(29015, 29098)],
        Path("masks/03_ed_credits_02.png"): [(29186, 29269)],
        Path("masks/03_ed_credits_03.png"): [(29353, 29436)],
        Path("masks/03_ed_credits_04.png"): [(29752, 29835)],
        Path("masks/03_ed_credits_05.png"): [(29894, 29977)],
        Path("masks/03_ed_credits_06.png"): [(30169, 30252)],
        Path("masks/03_ed_credits_07.png"): [(30754, 30837)],
        Path("masks/03_ed_credits_08.png"): [(31203, 31286)],
        Path("masks/03_ed_credits_09.png"): [(31425, 31508)],
        Path("masks/03_ed_credits_10.png"): [(31668, 31751)],
        Path("masks/03_ed_credits_11.png"): [(32368, 32451)],
        Path("masks/03_ed_credits_12.png"): [(32524, 32607)],
        Path("masks/03_ed_credits_13.png"): [(32968, 33051)],
        Path("masks/03_ed_credits_14.png"): [(33127, 33210)],
        Path("masks/03_ed_credits_15.png"): [(33448, 33531)],
        Path("masks/03_ed_credits_16.png"): [(33748, 33831)],
    }
).get_mask(src)
credit_mask = ExprOp.MAX([op_credit_mask, ed_custom_mask])
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
        else x265(settings, qp_clip=src, zones=[(OP[0], OP[1], 1.2)], add_props=True, resumable=False).encode(final)
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
    chapters = Chapters(jpnbd).set_names(["Prologue", "Opening", "Part A", "Part B", "Preview"])

    mux(
        video.to_track("JPNBD encode by sgt", "jpn", default=True, forced=False, args=["--deterministic", "217404"]),
        audio.to_track("FLAC 2.0", "jpn", default=True, forced=False),
        subs.to_track("Full Subtitles [RRA]", "eng", default=True, forced=False),
        *fonts,
        chapters,
        tmdb=TmdbConfig(217404, write_cover=True, write_date=True, write_ids=True, write_title=True),
    )
