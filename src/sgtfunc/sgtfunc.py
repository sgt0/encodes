from muxtools import GJM_GANDHI_PRESET, edit_style, gandhi_default
from vsmasktools import GenericMaskT
from vstools import SingleOrArr, vs

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
    thSAD: int | tuple[int, int | tuple[int, int]] | None = 115,
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

    denoised_luma = BM3DCudaRTC.denoise(
        clip, ref=ref, sigma=sigma, tr=tr, profile=Profile.NORMAL, planes=0
    )

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


def screengen(
    clip: vs.VideoNode,
    folder: str,
    suffix: str,
    frame_numbers: list[int] = [],
    start: int = 1,
) -> None:
    """
    Generates screenshots from a list of frames.
    """

    import os

    from vstools import Matrix, core

    folder_path = "./{name}".format(name=folder)

    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

    for i, num in enumerate(frame_numbers, start=start):
        filename = "{path}/{suffix}-{:05d}.png".format(num, path=folder_path, suffix=suffix)
        matrix = Matrix.from_video(clip)
        if matrix == Matrix.UNKNOWN:
            matrix = Matrix.BT709

        print(f"Saving Frame {i}/{len(frame_numbers)} from {suffix}", end="\r")
        core.imwri.Write(
            clip.resize.Spline36(
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
