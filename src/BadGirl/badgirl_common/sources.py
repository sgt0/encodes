from __future__ import annotations

from pathlib import Path, PurePath

from pydantic import BaseModel

RAWS_DIRECTORY = Path(r"E:\raws\bad-girl\web")


class Source(BaseModel):
    adn_path: PurePath | str
    amzn_path: PurePath | str
    hidi_path: PurePath | str | None = None
    op: tuple[int, int] | None = None
    ed: tuple[int, int] | None = None


sources = {
    "01": Source(
        adn_path=RAWS_DIRECTORY / "01 (ADN).mkv",
        amzn_path=RAWS_DIRECTORY / "01 (AMZN).mkv",
        hidi_path=RAWS_DIRECTORY / "01 (HIDI).mkv",
        op=(3237, 3237 + 2156 + 1),
        ed=(30689, 30689 + 2156 + 2),
    ),
    "02": Source(
        adn_path=RAWS_DIRECTORY / "02 (ADN).mkv",
        amzn_path=RAWS_DIRECTORY / "02 (AMZN).mkv",
        op=(1176, 1176 + 2156),
        ed=(30808, 30808 + 2156),
    ),
    "03": Source(
        adn_path=RAWS_DIRECTORY / "03 (ADN).mkv",
        amzn_path=RAWS_DIRECTORY / "03 (AMZN).mkv",
        op=(864, 864 + 2156),
        ed=(30928, 30928 + 2156),
    ),
    "04": Source(
        adn_path=RAWS_DIRECTORY / "04 (ADN).mkv",
        amzn_path=RAWS_DIRECTORY / "04 (AMZN).mkv",
        op=(432, 432 + 2156 + 1),
        ed=(30449, 30449 + 2156 + 2),
    ),
}
