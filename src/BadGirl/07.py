from __future__ import annotations

from vspreview import is_preview

from badgirl_common import filterchain, mux, sources

EPISODE = "07"
source = sources[EPISODE]
NO_DESCALE = [
    source.op,
    source.ed,
    (33925, None),  # Guest illustration.
]
FORCE_AMZN = [
    (14407, 14561),
]


filterchain_results = filterchain(source=source, no_descale=NO_DESCALE, force_amzn=FORCE_AMZN)

if not is_preview():
    mux(episode=EPISODE, source=source, filterchain_results=filterchain_results)
