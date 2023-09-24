from pathlib import Path

from awsmfunc import bbmod
from sgtfunc import deband, denoise
from vodesfunc import Clamped_Doubler, DescaleTarget, Waifu2x_Doubler, grain, set_output
from vskernels import Mitchell
from vsmasktools import SobelStd, diff_creditless_oped
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
from vstools import FieldBased, core, finalize_clip, replace_ranges

core.set_affinity(max_cache=17180)

# Frame range (inclusive) of the ED.
ED = (15609, 16566)

# Frame ranges (inclusive) that should not be descaled.
NO_DESCALE = [
    (16688, 16807),  # Next episode title.
    (16808, 16927),  # Guest illustration.
]

JPNBD = src_file(Path(r"X:\path\to\JPBD\Volume1\BDMV\STREAM\00009.m2ts").resolve(True))
src = JPNBD.init_cut(field_based=FieldBased.PROGRESSIVE)

NCED = src_file(
    Path(r"X:\path\to\JPBD\Volume1\BDMV\STREAM\00012.m2ts").resolve(True),
    trim=(2045, 2970),
)
nced = NCED.init_cut(field_based=FieldBased.PROGRESSIVE)
nced = core.std.Splice([nced[0] * (ED[1] - ED[0] + 1 - nced.num_frames), nced])
assert nced.num_frames == ED[1] - ED[0] + 1

# Fix edges
nced = bbmod(nced, 1, 1, 1, 1, 30)
edgefix = bbmod(src, 1, 1, 1, 1, 30)

# Rescale
lmask = SobelStd.edgemask(edgefix, multi=2, planes=(0, True)).std.Maximum()
rescale_target = DescaleTarget(
    height=878,
    kernel=Mitchell,
    upscaler=Clamped_Doubler(False, Waifu2x_Doubler(cuda=True, tiles=2)),
    line_mask=lmask,
).generate_clips(edgefix)
ed_credit_mask = diff_creditless_oped(
    edgefix, thr=0.7, ncop=core.std.BlankClip(), nced=nced, edstart=ED[0], edend=ED[1]
)
rescale_target.credit_mask = replace_ranges(rescale_target.credit_mask, ed_credit_mask, ED)
upscaled = rescale_target.get_upscaled(edgefix)
upscaled = replace_ranges(upscaled, edgefix, NO_DESCALE)

# Denoise
denoised = denoise(upscaled, strength=1.2, tr=3)

# Deband
debanded = deband(upscaled, denoised, thr=2)

# Regrain
grained = grain(debanded, seed=84869)

final = finalize_clip(grained)

if __name__ != "__main__":
    set_output(finalize_clip(src), "JPBD")
    set_output(final, "filtered")
else:
    setup = Setup("03")

    # Video
    settings = settings_builder_x265(preset="placebo")
    encoded = Path(setup.work_dir).joinpath("encoded.265").resolve()
    video_hevc = (
        VideoFile(encoded) if encoded.exists() else x265(settings, qp_clip=src).encode(final)
    )

    # Audio
    audio = do_audio(JPNBD, track=0, encoder=FLAC(), quiet=True)

    # Subs
    subs_gjm = (
        SubFile(Path(__file__).parent.joinpath("input", "03-gjm-sgt.ass").resolve(True))
        .resample(use_arch=True)
        .clean_garbage()
    )
    subs_niisama = (
        SubFile(Path(__file__).parent.joinpath("input", "Nii-sama", "03.ass").resolve(True))
        .resample(use_arch=True)
        .clean_garbage()
    )
    subs_gjm.collect_fonts()
    fonts = subs_niisama.collect_fonts()

    # Chapters
    chapters = Chapters(JPNBD)

    mux(
        video_hevc.to_track("JPNBD encode by sgt", "jpn", default=True, forced=False),
        audio.to_track("FLAC 2.0", "jpn", default=True, forced=False),
        subs_gjm.to_track("Full Subtitles [GJM]", "eng", default=True, forced=False),
        subs_niisama.to_track("Full Subtitles [Nii-sama]", "eng", default=False, forced=False),
        *fonts,
        chapters,
        tmdb=TmdbConfig(
            84869,
            write_cover=True,
            write_date=True,
            write_ids=True,
            write_summary=True,
            write_synopsis=True,
            write_title=True,
        )
    )
