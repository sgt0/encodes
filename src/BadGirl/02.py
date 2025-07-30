from __future__ import annotations

from vspreview import is_preview

from badgirl_common import filterchain, mux, sources

EPISODE = "02"
source = sources[EPISODE]
NO_DESCALE = [
    source.op,
    source.ed,
    (33925, None),  # Guest illustration.
]


filterchain_results = filterchain(source=source, no_descale=NO_DESCALE)

if not is_preview():
    mux(episode=EPISODE, source=source, filterchain_results=filterchain_results)
