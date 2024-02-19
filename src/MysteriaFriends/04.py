from pathlib import Path

from awsmfunc import bbmod
from jvsfunc import retinex_edgemask
from vodesfunc import Clamped_Doubler, DescaleTarget, Waifu2x_Doubler, grain, set_output
from vsdeband import Placebo
from vskernels import Mitchell
from vsmasktools import SobelStd
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
from vsrgtools import contrasharpening
from vstools import FieldBased, core, finalize_clip, replace_ranges

from sgtfunc import denoise

core.set_affinity(max_cache=17180)

# Frame range (inclusive) of the ED.
ED = (15611, 16544)

# Frame ranges (inclusive) that should not be descaled.
NO_DESCALE = [
    ED,
    (16688, 16807),  # Next episode title.
    (16808, 16927),  # Guest illustration.
]

# Note that this episode (4) does not have the same ED as the NCED but rather
# a simple scrolling credits ED, so we do not do any creditless masking.

JPNBD = src_file(Path(r"X:\path\to\JPBD\Volume1\BDMV\STREAM\00010.m2ts").resolve(True))
src = JPNBD.init_cut(field_based=FieldBased.PROGRESSIVE)

# Fix edges
edgefix = bbmod(src, 1, 1, 1, 1, 30)

# Rescale
lmask = SobelStd.edgemask(edgefix, multi=2, planes=(0, True)).std.Maximum()
rescale_target = DescaleTarget(
    height=878,
    kernel=Mitchell,
    upscaler=Clamped_Doubler(False, Waifu2x_Doubler(cuda=True, tiles=2)),
    line_mask=lmask,
).generate_clips(edgefix)
upscaled = rescale_target.get_upscaled(edgefix)
upscaled = replace_ranges(upscaled, edgefix, NO_DESCALE)

# Denoise
denoised = denoise(upscaled, strength=1.2, tr=3)

# Deband
mask_deband = retinex_edgemask(denoised).rgvs.RemoveGrain(3)
debanded = Placebo.deband(denoised, thr=2, iterations=16)
debanded = core.std.MaskedMerge(debanded, denoised, mask_deband)
debanded = contrasharpening(debanded, upscaled, mode=3)

# Regrain
grained = grain(debanded, seed=84869)

final = finalize_clip(grained)

if __name__ != "__main__":
    set_output(finalize_clip(src), "JPBD")
    set_output(final, "filtered")
else:
    setup = Setup("04")

    # Video
    settings = settings_builder_x265(preset="placebo")
    encoded = Path(setup.work_dir).joinpath("encoded.265").resolve()
    video_hevc = VideoFile(encoded) if encoded.exists() else x265(settings, qp_clip=src).encode(final)

    # Audio
    audio = do_audio(JPNBD, track=0, encoder=FLAC(), quiet=True)

    # Subs
    subs_gjm = (
        SubFile(Path(__file__).parent.joinpath("input", "04-gjm-sgt.ass").resolve(True))
        .resample(use_arch=True)
        .clean_garbage()
    )
    subs_niisama = (
        SubFile(Path(__file__).parent.joinpath("input", "Nii-sama", "04.ass").resolve(True))
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
