from __future__ import annotations

from vspreview import is_preview

from badgirl_common import filterchain, mux, sources

EPISODE = "06"
source = sources[EPISODE]
NO_DESCALE = [
    source.op,
    source.ed,
    (33927, None),  # Guest illustration.
]
FORCE_ADN = [
    (0, 128),
]


filterchain_results = filterchain(source=source, no_descale=NO_DESCALE, force_adn=FORCE_ADN)

if not is_preview():
    mux(episode=EPISODE, source=source, filterchain_results=filterchain_results)
