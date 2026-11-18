"""Builds the calculator's font: Inter Regular, cut to the characters the app draws, with
tabular digits as its default digits.

    python subset_inter.py Inter-Regular.ttf Inter-Regular.subset.ttf

Why a built file: egui cannot switch OpenType features, so the ten digits are remapped here to
Inter's tabular ones (its `tnum` feature, the digits only: the whole feature would also widen
every space, full stop, comma, colon, hyphen and bracket in running text). The app build runs
no font tools; it embeds the built file (`src/app/fonts.rs`).

Source: Inter 4.1, https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip
  (SHA-256 9883fdd4a49d4fb66bd8177ba6625ef9a64aa45899767dde3d36aa425756b11e),
  its file extras/ttf/Inter-Regular.ttf
  (SHA-256 40d692fce188e4471e2b3cba937be967878f631ad3ebbbdcd587687c7ebe0c82).
Licence: SIL Open Font License 1.1 (OFL.txt beside this file, the zip's LICENSE.txt). Inter
  declares no Reserved Font Name, so the cut file keeps the name.
Tool: fontTools 4.51.0 on Python 3.12. The output does not depend on the date (the head table's
  timestamps are kept), so the same input and tool give the same bytes.
"""
import sys

from fontTools import subset
from fontTools.ttLib import TTFont

DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

# What the app draws (UX overhaul spec 6.2): Latin, Greek, the maths and unit signs in use.
# A character outside these falls back to egui's built-in fonts.
UNICODES = (
    list(range(0x0020, 0x007F))  # Basic Latin
    + list(range(0x00A0, 0x0100))  # Latin-1: the degree sign, the middle dot, the micro sign
    + list(range(0x0370, 0x0400))  # Greek: the equations' symbols
    + list(range(0x2010, 0x2028))  # dashes, quotes, the bullet, the ellipsis
    + [0x2030, 0x2032, 0x2033]  # per mille, prime, double prime
    + list(range(0x2070, 0x209D))  # superscripts and subscripts
    + list(range(0x2190, 0x219A))  # arrows
    + [0x2202, 0x2206, 0x220F, 0x2211, 0x2212, 0x2215, 0x2219, 0x221A, 0x221E, 0x222B]
    + [0x2248, 0x2260, 0x2264, 0x2265]  # about equal, not equal, at most, at least
)


def main(source: str, target: str) -> None:
    font = TTFont(source, recalcTimestamp=False)
    order = set(font.getGlyphOrder())
    tabular = {name: f"{name}.tf" for name in DIGITS}
    missing = [glyph for glyph in tabular.values() if glyph not in order]
    if missing:
        sys.exit(f"{source} has no tabular digits: {missing}")
    for table in font["cmap"].tables:
        for codepoint, name in list(table.cmap.items()):
            if name in tabular:
                table.cmap[codepoint] = tabular[name]

    options = subset.Options()
    options.layout_features = []  # egui applies no GSUB or GPOS
    options.hinting = False  # egui's rasterizer does not hint
    options.name_IDs = [0, 1, 2, 3, 4, 5, 6, 13, 14]  # the copyright and licence stay in the file
    options.notdef_outline = True
    options.recalc_timestamp = False
    subsetter = subset.Subsetter(options)
    subsetter.populate(unicodes=UNICODES)
    subsetter.subset(font)

    cmap = font.getBestCmap()
    widths = {font["hmtx"][cmap[ord(digit)]][0] for digit in "0123456789"}
    if len(widths) != 1:
        sys.exit(f"the digits have {len(widths)} advance widths, not one")
    font.save(target)
    print(f"{target}: {len(font.getGlyphOrder())} glyphs, {len(cmap)} characters, digit advance {widths.pop()}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
