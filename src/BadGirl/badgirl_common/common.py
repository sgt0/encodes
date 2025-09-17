from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

from jetpytools import inject_self
from muxtools import Setup, VideoFile
from muxtools import mux as vsmux
from pydantic import BaseModel, ConfigDict
from vodesfunc import adaptive_grain, ntype4
from vsdeband import Placebo, PlaceboDither, pfdeband
from vsdenoise import DFTTest, frequency_merge
from vsexprtools import norm_expr
from vskernels import Catrom
from vsmuxtools import SourceFilter, do_audio, settings_builder_x265, src_file, x265
from vspreview import is_preview
from vsscale import Rescale, Waifu2x
from vstools import (
    FrameRangeN,
    FrameRangesN,
    Keyframes,
    PlanesT,
    Sar,
    VideoPackets,
    depth,
    finalize_clip,
    get_prop,
    insert_clip,
    replace_ranges,
    set_output,
    vs,
)

import sgtfunc
from badgirl_common.sources import Source, sources


class Placebo2(Placebo):
    @inject_self
    def deband(  # type: ignore[override]
        self,
        clip: vs.VideoNode,
        radius: float = 16.0,
        thr: float | list[float] = 3.0,
        iterations: int = 4,
        grain: float | list[float] = 0.0,
        dither: PlaceboDither = PlaceboDither.DEFAULT,
        planes: PlanesT = None,
    ) -> vs.VideoNode:
        # Drop `planes`.
        return super().deband(clip, radius, thr, iterations, grain, dither)


class FilterchainResults(BaseModel):
    src: vs.VideoNode
    final: vs.VideoNode
    audio_file: src_file

    model_config = ConfigDict(arbitrary_types_allowed=True)


def filterchain(
    *,
    source: Source,
    no_descale: FrameRangeN | FrameRangesN,
    force_adn: FrameRangeN | FrameRangesN | None = None,
    force_amzn: FrameRangeN | FrameRangesN | None = None,
    post_double: Callable[[vs.VideoNode], vs.VideoNode] | None = None,
) -> FilterchainResults:
    adn_file = src_file(str(source.adn_path), preview_sourcefilter=SourceFilter.BESTSOURCE)
    amzn_file = src_file(str(source.amzn_path), preview_sourcefilter=SourceFilter.BESTSOURCE)
    # hidi_file = src_file(str(source.hidi_path), preview_sourcefilter=SourceFilter.BESTSOURCE)

    adn = adn_file.init_cut().std.SetFrameProps(source="ADN")
    amzn = amzn_file.init_cut().std.SetFrameProps(source="AMZN")
    amzn = norm_expr(amzn, "x 128 +", planes=[1, 2])
    # hidi = hidi_file.init_cut().std.SetFrameProps(source="HIDI")
    # hidi = change_fps(hidi, Fraction(numerator=24000, denominator=1001))[24:]

    min_frames = min(adn.num_frames, amzn.num_frames)
    adn = adn[:min_frames]
    amzn = amzn[:min_frames]

    keyframes = Keyframes.unique(adn, str(source.adn_path))
    adn = keyframes.to_clip(adn, scene_idx_prop=True)
    amzn = keyframes.to_clip(amzn, scene_idx_prop=True)

    adn = VideoPackets.from_video(source.adn_path).apply_props(adn, keyframes)
    amzn = VideoPackets.from_video(source.amzn_path).apply_props(amzn, keyframes)

    def scene_select(n: int, f: Sequence[vs.VideoFrame]) -> vs.VideoNode:
        adn_avg = get_prop(f[0], "PktSceneAvgSize", float, default=1)
        amzn_avg = get_prop(f[1], "PktSceneAvgSize", float, default=1)

        if abs(adn_avg - amzn_avg) / adn_avg < 0.1:
            adn_max = get_prop(f[0], "PktSceneMaxSize", int, default=1)
            amzn_max = get_prop(f[1], "PktSceneMaxSize", int, default=1)

            return adn if adn_max > amzn_max else amzn

        return adn if adn_avg > amzn_avg else amzn

    src = adn.std.FrameEval(partial(scene_select), clip_src=[adn, amzn], prop_src=[adn, amzn])
    if force_adn:
        src = replace_ranges(src, adn, force_adn)
    if force_amzn:
        src = replace_ranges(src, amzn, force_amzn)

    # OP/ED handling.
    if source.op:
        src = replace_ranges(src, amzn, source.op)

        # OP inter-merge
        ops = [
            src_file(str(x.amzn_path), trim=x.op, preview_sourcefilter=SourceFilter.BESTSOURCE).init_cut()[:2156]
            for x in sources.values()
            if x.op
        ]
        src = insert_clip(
            src,
            frequency_merge(*ops, lowpass=lambda clip: DFTTest().denoise(clip)),
            source.op[0],
        )
    if source.ed:
        src = replace_ranges(src, amzn, source.ed)

    # Denoise
    denoised = sgtfunc.denoise(src, sigma=0.65, strength=0.3, thSAD=133, tr=3)

    # Rescale
    rs = Rescale(depth(denoised, 32), 880.9, Catrom, upscaler=Waifu2x.Cunet, crop=(1, 1, 0, 0))
    if post_double:
        rs.doubled = post_double(rs.doubled)
    rs.default_credit_mask()
    rs.default_line_mask()
    rescaled = replace_ranges(depth(rs.upscale, 16), denoised, no_descale)
    rescaled = Sar(1, 1).apply(rescaled)

    # Deband
    debanded = pfdeband(rescaled, debander=Placebo2, thr=1.6)

    # Regrain
    grained = adaptive_grain(debanded, strength=[1.99, 0.4], size=3.16, temporal_average=50, seed=258000, **ntype4)

    final = finalize_clip(grained)

    if is_preview():
        set_output(adn, "adn", scenes=keyframes)
        set_output(amzn, "amzn", scenes=keyframes)
        set_output(src, "src", scenes=keyframes)
        set_output(final, "final", scenes=keyframes)

    return FilterchainResults(src=src, final=final, audio_file=adn_file)


def mux(
    *,
    episode: str,
    source: Source,
    filterchain_results: FilterchainResults,
) -> Path | str:
    setup = Setup(episode)
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
        asm="avx512",
        keyint=round(filterchain_results.final.fps) * 10,
    )
    encoded = Path(setup.work_dir).joinpath("encoded.265").resolve()
    zones: list[tuple[int, int, float]] = []
    if source.op is not None:
        zones.append((source.op[0], source.op[1], 1.2))
    if source.ed is not None:
        zones.append((source.ed[0], source.ed[1], 1.2))
    video = (
        VideoFile(encoded)
        if encoded.exists()
        else x265(settings, zones=zones, qp_clip=filterchain_results.src, resumable=False).encode(
            filterchain_results.final
        )
    )

    return vsmux(
        video.to_track("WEB encode by sgt", "jpn", default=True, forced=False, args=["--deterministic", "258000"]),
        do_audio(filterchain_results.audio_file).to_track("AAC 2.0", "jpn", default=True, forced=False),
    )
