from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal, NamedTuple

from muxtools import GJM_GANDHI_PRESET, edit_style, gandhi_default
from vskernels import Catrom, KernelT
from vsmasktools import EdgeDetectT, GenericMaskT, PrewittStd
from vstools import Keyframes, MatrixT, SceneBasedDynamicCache, SingleOrArr, vs

SGT_SUBS_STYLES = [
    *GJM_GANDHI_PRESET,
    edit_style(gandhi_default, "Signs", margin_l=10, margin_r=10, margin_v=10),
]
"""
GJM Gandhi Sans preset with an additional "Signs" style.
"""


def denoise(
    clip: vs.VideoNode,
    block_size: int = 16,
    limit: int | tuple[int, int] = 255,
    overlap: int = 8,
    sigma: SingleOrArr[float] = 0.7,
    sr: int = 2,
    strength: float = 0.2,
    thSAD: int | tuple[int, int | tuple[int, int]] | None = 115,  # noqa: N803
    tr: int = 2,
) -> vs.VideoNode:
    """
    MVTools + BM3D + NLMeans denoise.
    """

    from vsdenoise import (
        BM3DCudaRTC,
        DeviceType,
        MotionMode,
        MVTools,
        PelType,
        Prefilter,
        Profile,
        SADMode,
        SearchMode,
        WeightMode,
        nl_means,
    )
    from vstools import ChromaLocation

    ref = MVTools.denoise(
        clip,
        sad_mode=SADMode.SPATIAL.same_recalc,
        motion=MotionMode.HIGH_SAD,
        prefilter=Prefilter.DFTTEST,
        pel_type=PelType.WIENER,
        search=SearchMode.DIAMOND.defaults,
        block_size=block_size,
        overlap=overlap,
        thSAD=thSAD,
        limit=limit,
    )

    denoised_luma = BM3DCudaRTC.denoise(clip, ref=ref, sigma=sigma, tr=tr, profile=Profile.NORMAL, planes=0)
    denoised_luma = ChromaLocation.ensure_presence(denoised_luma, ChromaLocation.from_video(clip, strict=True))

    return nl_means(
        denoised_luma,
        ref=ref,
        strength=strength,
        tr=tr,
        sr=sr,
        wmode=WeightMode.BISQUARE_HR,  # wmode=3
        planes=[1, 2],
        device_type=DeviceType.CUDA,
    )


# Fork of `sscomp.lazylist()` without reimplementing
# `vstools.clip_async_render()` and some typing improvements.
#
# MIT License
#
# Copyright (c) 2022 notSeaSmoke
# Copyright (c) 2023 sgt
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
def lazylist(
    clip: vs.VideoNode,
    dark_frames: int = 8,
    light_frames: int = 4,
    seed: int = 20202020,
    diff_thr: int = 15,
    d_start_thresh: float = 0.075000,  # 0.062745 is solid black
    d_end_thresh: float = 0.380000,
    l_start_thresh: float = 0.450000,
    l_end_thresh: float = 0.750000,
) -> list[int]:
    """
    Generates a list of frames for comparison purposes.
    """

    import random
    from functools import partial

    from vstools import clip_async_render, core, get_prop

    dark = []
    light = []

    def checkclip(n: int, f: vs.VideoFrame, clip: vs.VideoNode) -> vs.VideoNode:
        avg: float = get_prop(f, "PlaneStatsAverage", float)

        if d_start_thresh <= avg <= d_end_thresh:
            dark.append(n)

        elif l_start_thresh <= avg <= l_end_thresh:
            light.append(n)

        return clip

    s_clip = clip.std.PlaneStats()

    eval_frames = core.std.FrameEval(clip, partial(checkclip, clip=s_clip), prop_src=s_clip)
    clip_async_render(eval_frames, progress="Rendering...")

    dark.sort()
    light.sort()

    dark_dedupe = [dark[0]]
    light_dedupe = [light[0]]

    thr = round(clip.fps_num / clip.fps_den * diff_thr)
    lastvald = dark[0]
    lastvall = light[0]

    for i in range(1, len(dark)):
        checklist = dark[0:i]
        x = dark[i]

        for y in checklist:
            if x >= y + thr and x >= lastvald + thr:
                dark_dedupe.append(x)
                lastvald = x
                break

    for i in range(1, len(light)):
        checklist = light[0:i]
        x = light[i]

        for y in checklist:
            if x >= y + thr and x >= lastvall + thr:
                light_dedupe.append(x)
                lastvall = x
                break

    if len(dark_dedupe) > dark_frames:
        random.seed(seed)
        dark_dedupe = random.sample(dark_dedupe, dark_frames)

    if len(light_dedupe) > light_frames:
        random.seed(seed)
        light_dedupe = random.sample(light_dedupe, light_frames)

    return dark_dedupe + light_dedupe


def sample_ptype(
    clips: Sequence[vs.VideoNode], n: int = 50, picture_types: Iterable[Literal["I", "P", "B"]] = {"I", "P", "B"}
) -> list[int]:
    """
    Randomly samples `n` frame numbers from the given clips, selecting only
    those of the given picture types. This is similar to the frame selection
    of vspreview's comp feature. One difference here is that the sampled frames
    will have the same picture type across the clips.

    :param clips: Clips to sample frames from.
    :param n: Number of frames to sample. Defaults to 50.
    :param picture_types: Set of picture types to select. Defaults to all of "I", "P", and "B".

    Seeding the RNG can be done beforehand::

        from random import seed

        seed(1)
        frames = sgtfunc.sample_ptype(clips, n=50, picture_types={"B", "P"})
    """

    from random import randrange

    from vstools import get_prop

    # Work with the smallest frame range.
    num_frames = min(clip.num_frames for clip in clips)

    picture_types_b = {p.encode() for p in picture_types}

    # Frame numbers that have been checked already.
    checked = set[int]()

    samples = set[int]()
    while len(samples) < n:
        if len(checked) > 50 * n:
            raise RecursionError("Reached rejection sampling limit.")

        if len(checked) >= num_frames:
            raise ValueError("Could not find enough frames.")

        # Rejection sample until we get a frame we haven't seen before.
        x = randrange(start=0, stop=num_frames)
        while x in checked:
            x = randrange(start=0, stop=num_frames)
        checked.add(x)

        # Get this frame's picture type from the first clip.
        common_picture_type = get_prop(clips[0][x], "_PictType", bytes)
        if common_picture_type not in picture_types_b:
            continue

        # Check if the same frame in all other clips are of the same picture
        # type.
        if all(
            get_prop(f, "_PictType", bytes) == common_picture_type
            for f in vs.core.std.Splice([clip[x] for clip in clips], mismatch=True).frames(close=True)
        ):
            samples.add(x)
            continue

    return list(samples)


def screengen(
    clip: vs.VideoNode,
    directory_path: Path,
    prefix: str,
    frame_numbers: Sequence[int] = [],
) -> None:
    """
    Writes images from a list of frames.

    :param clip: Clip to fetch frames from.
    :param directory_path: Path to the directory that should receive the frame images. Will be created if it does not exist.
    :param prefix: String that each file name will begin with.
    :param frame_numbers: Frame numbers to save images of.
    """

    from vskernels import Catrom
    from vstools import Matrix, core

    if not Path.is_dir(directory_path):
        Path.mkdir(directory_path)

    for i, num in enumerate(frame_numbers, start=1):
        filename = directory_path.joinpath(f"{prefix}-{num:05d}.png")
        matrix = Matrix.from_video(clip)
        if matrix == Matrix.UNKNOWN:
            matrix = Matrix.BT709

        print(f"Saving frame {i}/{len(frame_numbers)} from {prefix}", end="\r")
        core.imwri.Write(
            Catrom.resample(
                clip,
                format=vs.RGB24,
                matrix_in=matrix,
                dither_type="error_diffusion",
            ),
            "PNG",
            filename,
            overwrite=True,
        ).get_frame(num)


def descale_errors_async(
    src: vs.VideoNode,
    rescaled: vs.VideoNode,
    thr: float | list[float] = 0.038,
    tr: int = 1,
    ref: float = 0.0002,
    range_length: int = 5,
) -> list[tuple[int, int]]:
    """
    Finds descale errors.
    """

    from vsscale import descale_error_mask
    from vstools import find_prop

    return find_prop(
        descale_error_mask(src, rescaled, thr=thr, tr=tr).std.PlaneStats(),
        "PlaneStatsAverage",
        ">",
        ref=ref,
        range_length=range_length,
    )


def get_rescale_error(
    source: vs.VideoNode,
    rescaled: vs.VideoNode,
    *,
    line_mask: GenericMaskT | bool = True,
    crop: int = 0,
) -> float:
    """
    Gets the absolute error of a rescale, optionally after a line mask and crop.
    """

    from vsmasktools import Sobel, normalize_mask, replace_squaremask
    from vstools import get_prop, get_y

    source = get_y(source)
    rescaled = get_y(rescaled)

    if line_mask is True:
        line_mask = Sobel

    if line_mask:
        line_mask = normalize_mask(line_mask, source)
        rescaled = source.std.MaskedMerge(rescaled, line_mask)

    if crop:
        rescaled = replace_squaremask(
            rescaled,
            source,
            (source.width - crop * 2, source.height - crop * 2, crop, crop),
            invert=True,
        )

    return get_prop(rescaled.std.PlaneStats(source), "PlaneStatsDiff", float)


def pretty_kernel_name(kernel: KernelT) -> str:
    from vskernels import Bicubic, Kernel, Lanczos

    kernel = Kernel.ensure_obj(kernel)
    kernel_name = kernel.__class__.__name__

    if isinstance(kernel, Bicubic):
        kernel_name += f" (Bicubic b={kernel.b:.2f}, c={kernel.c:.2f})"
    elif isinstance(kernel, Lanczos):
        kernel_name += f" (taps={kernel.taps})"

    return kernel_name


# MIT License
#
# Copyright (c) 2022 LightArrowsEXE
# Copyright (c) 2023 sgt
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
def adb_heuristics(
    clip: vs.VideoNode,
    edge_mask: EdgeDetectT = PrewittStd,
    matrix: MatrixT | None = None,
    kernel: KernelT = Catrom,
) -> vs.VideoNode:
    """
    Variation of `lvsfunc.autodb_dpir()` that doesn't do any deblocking, instead
    this only sets the heuristic frame props.

    :param clip: Clip to process.
    :param edge_mask: Edge mask to use for calculating the edge values.
    :param matrix: Matrix of the processed clip.
    :param kernel: Kernel to use for conversions between YUV and RGB.

    :return: Clip with "Adb_*" props.
    """

    from functools import partial

    from vskernels import Kernel
    from vsmasktools import normalize_mask
    from vsrgtools import BlurMatrix
    from vstools import Matrix, check_variable, get_prop, shift_clip

    def adb_eval(n: int, f: Sequence[vs.VideoFrame], clip: vs.VideoNode) -> vs.VideoNode:  # noqa: ARG001
        evref_diff, y_next_diff, y_prev_diff = (
            get_prop(f[i], prop, float)
            for i, prop in zip(range(3), ["EdgeValRefDiff", "YNextDiff", "YPrevDiff"], strict=True)
        )

        f_type = get_prop(f[0], "_PictType", bytes).decode("utf-8")
        if f_type == "I":
            y_next_diff = (y_next_diff + evref_diff) / 2

        return clip.std.SetFrameProps(
            Adb_EdgeValRefDiff=max(evref_diff * 255, -1),
            Adb_YNextDiff=max(y_next_diff * 255, -1),
            Adb_YPrevDiff=max(y_prev_diff * 255, -1),
        )

    assert check_variable(clip, adb_heuristics)
    kernel = Kernel.ensure_obj(kernel)
    is_rgb = clip.format.color_family is vs.RGB
    if not is_rgb:
        if matrix is None:
            matrix = Matrix.from_video(clip)

        targ_matrix = Matrix(matrix)

        rgb = kernel.resample(clip, format=vs.RGBS, matrix_in=targ_matrix)
    else:
        rgb = clip

    evref = normalize_mask(edge_mask, rgb)

    # Note that `mode="s"` is not equivalent to `mode=ConvMode.SQUARE`.
    evref_rm = BlurMatrix.WMEAN(evref.std.Median(), mode="s")  # type: ignore[arg-type]

    diffevref = evref.std.PlaneStats(evref_rm, prop="EdgeValRef")
    diffnext = rgb.std.PlaneStats(shift_clip(rgb, 1), prop="YNext")
    diffprev = rgb.std.PlaneStats(shift_clip(rgb, -1), prop="YPrev")

    ret = rgb.std.FrameEval(partial(adb_eval, clip=rgb), prop_src=[diffevref, diffnext, diffprev])

    return kernel.resample(ret, format=clip.format, matrix=targ_matrix if not is_rgb else None)


class AdbPropsTuple(NamedTuple):
    evref_diff: float
    y_next_diff: float
    y_prev_diff: float


def adb_props(f: vs.VideoFrame) -> AdbPropsTuple:
    from vstools import get_prop

    return AdbPropsTuple(
        get_prop(f, "Adb_EdgeValRefDiff", float),
        get_prop(f, "Adb_YNextDiff", float),
        get_prop(f, "Adb_YPrevDiff", float),
    )


def avg_adb_props(clip: vs.VideoNode) -> AdbPropsTuple:
    from statistics import fmean

    from vstools import clip_data_gather

    stats = clip_data_gather(clip, None, lambda _, f: adb_props(f))
    return AdbPropsTuple(
        fmean(x.evref_diff for x in stats),
        fmean(x.y_next_diff for x in stats),
        fmean(x.y_prev_diff for x in stats),
    )


class SceneBasedAdbHeuristics(SceneBasedDynamicCache):
    """
    Calculates heuristics from `adb_heuristics()` on a per-scene basis, setting
    frame props for the average heuristic value within a scene.

    Example usage::

      keyframes = Keyframes.from_clip(clip)
      propped_clip = SceneBasedAdbHeuristics.from_clip(clip, keyframes)
      # Can now build upon the "Scene_Avg_*" props.
    """

    def __init__(
        self,
        clip: vs.VideoNode,
        keyframes: Keyframes | str,
        cache_size: int = 5,
    ) -> None:
        super().__init__(adb_heuristics(clip), keyframes, cache_size)

    def get_clip(self, scene_idx: int) -> vs.VideoNode:
        frame_range = self.keyframes.scenes[scene_idx]
        cut = self.clip[frame_range.start : frame_range.stop]
        evref_diff, y_next_diff, y_prev_diff = avg_adb_props(cut)
        return self.clip.std.SetFrameProps(
            Scene_Avg_EdgeValRefDiff=evref_diff,
            Scene_Avg_YNextDiff=y_next_diff,
            Scene_Avg_YPrevDiff=y_prev_diff,
        )
