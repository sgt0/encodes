# Gets the kernel with the minimum average rescale error per scene. This is
# based on sample code from Setsu and is just an example of what can be done
# with `vstools.Keyframes`.

from dataclasses import dataclass
from statistics import fmean, median, stdev

import vskernels
from vodesfunc import DescaleTarget, set_output
from vsmasktools import Sobel, normalize_mask
from vspreview import is_preview
from vssource import source
from vstools import (
    DynamicClipsCache,
    FieldBased,
    Keyframes,
    clip_data_gather,
    core,
    get_prop,
    get_y,
    vs,
)

FILE = r"X:\path\to\video.m2ts"
HEIGHT: float = 719.8
BASE_HEIGHT: int | None = 720
KERNELS: dict[str, vskernels.KernelT] = {
    "Bilinear": vskernels.Bilinear,
    "Mitchell": vskernels.Mitchell,
    "Catrom": vskernels.Catrom,
}


core.set_affinity(16, 17180)
clip = source(FILE, bits=16, field_based=FieldBased.PROGRESSIVE)
keyframes = Keyframes.unique(clip, FILE)

# Prepare rescaled clips.
src = get_y(clip)
line_mask = normalize_mask(Sobel, src)
rescales: dict[str, vs.VideoNode] = {}
for name, kernel in KERNELS.items():
    dt = DescaleTarget(
        height=HEIGHT, base_height=BASE_HEIGHT, kernel=kernel, credit_mask=False, line_mask=False
    ).generate_clips(clip)
    rescaled = get_y(dt.rescale)
    rescales[name] = src.std.MaskedMerge(rescaled, line_mask).std.PlaneStats(src)


@dataclass
class SceneRescaleStats:
    kernel: str
    """Kernel name."""

    scene_idx: int = 0
    """Scene index."""

    mean: float = 0.0
    """Arithmetic mean (average)."""

    median: float = 0.0
    """Median."""

    stdev: float = 0.0
    """Standard deviation."""


class SceneRescaleErrors(dict[int, SceneRescaleStats]):
    def __getitem__(self, idx: int) -> SceneRescaleStats:
        # If previously calculated, return early.
        if idx in self:
            return super().__getitem__(idx)

        frame_range = keyframes.scenes[idx]
        min_avg_error = SceneRescaleStats(kernel="UndefinedKernel", scene_idx=idx, mean=999.9)
        for name, rescale in rescales.items():
            cut_rescale = rescale[frame_range.start : frame_range.stop]
            frames_errors = clip_data_gather(
                cut_rescale, None, lambda _, f: get_prop(f, "PlaneStatsDiff", float)
            )
            avg_error = fmean(frames_errors)
            if avg_error < min_avg_error.mean:
                min_avg_error.kernel = name
                min_avg_error.mean = avg_error
                min_avg_error.median = median(frames_errors)
                min_avg_error.stdev = stdev(frames_errors) if len(frames_errors) >= 2 else 0.0

        self[idx] = min_avg_error
        return min_avg_error


scene_errors = SceneRescaleErrors()


class SceneRescaleCache(DynamicClipsCache[int]):
    def get_clip(self, key: int) -> vs.VideoNode:
        min_avg_error = scene_errors[key]
        return clip.std.SetFrameProps(
            SceneIndex=key,
            SceneRescaleKernel=min_avg_error.kernel,
            SceneRescaleAverageError=min_avg_error.mean,
            SceneRescaleStdDevError=min_avg_error.stdev,
            SceneRescaleMedianError=min_avg_error.median,
        )


scenes_cache = SceneRescaleCache(3)
output = clip.std.FrameEval(lambda n: scenes_cache[keyframes.scenes.indices[n]])

if is_preview():
    set_output(output)
else:
    # Go through each scene and print the kernel with the minimum average
    # rescale error.
    for s in keyframes.scenes:
        print(f"{keyframes.scenes[s]} {scene_errors[s]}")
