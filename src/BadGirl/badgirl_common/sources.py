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
    "05": Source(
        adn_path=RAWS_DIRECTORY / "05 (ADN).mkv",
        amzn_path=RAWS_DIRECTORY / "05 (AMZN).mkv",
        op=(840, 840 + 2156),
        ed=(29875, 29875 + 2156),
    ),
    "06": Source(
        adn_path=RAWS_DIRECTORY / "06 (ADN).mkv",
        amzn_path=RAWS_DIRECTORY / "06 (AMZN).mkv",
        op=(1296, 1296 + 2156),
        ed=(26951, 26951 + 2156),
    ),
    "07": Source(
        adn_path=RAWS_DIRECTORY / "07 (ADN).mkv",
        amzn_path=RAWS_DIRECTORY / "07 (AMZN).mkv",
        op=(1152, 1152 + 2156),
        ed=(30400, 30400 + 2156 + 2),
    ),
    "08": Source(
        adn_path=RAWS_DIRECTORY / "08 (ADN).mkv",
        amzn_path=RAWS_DIRECTORY / "08 (AMZN).mkv",
        op=(840, 840 + 2156),
        ed=(30520, 30520 + 2156 + 2),
    ),
    "09": Source(
        adn_path=RAWS_DIRECTORY / "09 (ADN).mkv",
        amzn_path=RAWS_DIRECTORY / "09 (AMZN).mkv",
        op=(1032, 1032 + 2156),
        ed=(30664, 30664 + 2156 + 2),
    ),
    "10": Source(
        adn_path=RAWS_DIRECTORY / "10 (ADN).mkv",
        amzn_path=RAWS_DIRECTORY / "10 (AMZN).mkv",
        op=(936, 936 + 2156),
        ed=(30329, 30329 + 2156 + 2),
    ),
    "11": Source(
        adn_path=RAWS_DIRECTORY / "11 (ADN).mkv",
        amzn_path=RAWS_DIRECTORY / "11 (AMZN).mkv",
        op=(720, 720 + 2156 + 1),
        ed=(30426, 30426 + 2156 + 2),
    ),
}
