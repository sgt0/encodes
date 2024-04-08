# encodes

sgt encodes.

## Setup

1. Install Python v3.11.5.
2. Install Poetry.
3. Install [VapourSynth][] globally.
4. Install a bunch of VapourSynth plugins. Inexhaustive list:
    1. [vs-dfttest2][]. Add `dfttest2.py` to global `site-packages/`.
    2. [vs-mlrt][]. Add `vsmlrt.py` to global `site-packages/`.
    3. [adaptivegrain][].
    4. [vapoursynth-BilateralGPU][].
    5. [VapourSynth-BM3DCUDA][].
    6. [vs-miscfilters][].
    7. [vs-nlm-cuda][].
5.
    ```bash
    # Windows
    $ poetry install
    $ .venv/Scripts/activate.bat
    $ patch -Np1 -i patches/fontvalidator-validate-all-tracks.patch
    $ patch -Np1 -i patches/muxtools-sanitize-font-names.patch
    $ patch -Np1 -i patches/muxtools-tags-utf8.patch
    $ patch -Np1 -i patches/vsdeband-grain-post-process-type.patch
    ```

    ```bash
    # Linux
    $ poetry install
    $ source .venv/bin/activate

    # Patches in `patches/` would need to be updated to target
    # `.venv/lib/python3.11/site-packages/`.
    ```



   [VapourSynth]: https://github.com/vapoursynth/vapoursynth
   [vs-dfttest2]: https://github.com/AmusementClub/vs-dfttest2
   [vs-mlrt]: https://github.com/AmusementClub/vs-mlrt
   [adaptivegrain]: https://github.com/Irrational-Encoding-Wizardry/adaptivegrain
   [vapoursynth-BilateralGPU]: https://github.com/Rational-Encoding-Thaumaturgy/vapoursynth-BilateralGPU
   [VapourSynth-BM3DCUDA]: https://github.com/WolframRhodium/VapourSynth-BM3DCUDA
   [vs-miscfilters]: https://github.com/vapoursynth/vs-miscfilters-obsolete
   [vs-nlm-cuda]: https://github.com/AmusementClub/vs-nlm-cuda
