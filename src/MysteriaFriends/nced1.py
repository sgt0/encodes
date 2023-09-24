from pathlib import Path

from awsmfunc import bbmod
from jvsfunc import retinex_edgemask
from vodesfunc import Clamped_Doubler, DescaleTarget, Waifu2x_Doubler, grain, set_output
from vsdeband import Placebo
from vskernels import Mitchell
from vsmasktools import SobelStd
from vsmuxtools import (
    FLAC,
    Setup,
    VideoFile,
    do_audio,
    mux,
    settings_builder_x265,
    src_file,
    x265,
)
from vsrgtools import contrasharpening
from vstools import FieldBased, core, finalize_clip

from sgtfunc import denoise

core.set_affinity(max_cache=17180)

JPNBD = src_file(Path(r"X:\path\to\JPBD\Volume1\BDMV\STREAM\00012.m2ts").resolve(True))
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
    setup = (
        Setup("NCED1")
        .edit("mkv_title_naming", "$show$ - $ep$")
        .edit("out_name", "[sgt] $show$ - $ep$ (BD 1080p HEVC FLAC) [#crc32#]")
    )

    # Video
    settings = settings_builder_x265(preset="placebo")
    encoded = Path(setup.work_dir).joinpath("encoded.265").resolve()
    video_hevc = (
        VideoFile(encoded) if encoded.exists() else x265(settings, qp_clip=src).encode(final)
    )

    # Audio
    audio = do_audio(JPNBD, track=0, encoder=FLAC(), quiet=True)

    mux(
        video_hevc.to_track("JPNBD encode by sgt", "jpn", default=True, forced=False),
        audio.to_track("FLAC 2.0", "jpn", default=True, forced=False),
    )
