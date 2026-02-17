#    This file is part of ansi2html.
#    Copyright (C) 2012  Kuno Woudt <kuno@frob.nl>
#    Copyright (C) 2013  Sebastian Pipping <sebastian@pipping.org>
#
#    This program is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 3 of the License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with this program; if not, write to the Free Software Foundation,
#    Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


from typing import Dict, List


class Rule:
    def __init__(self, klass: str, **kw: str) -> None:
        self.klass = klass
        self.kw = "; ".join(
            [(k.replace("_", "-") + ": " + kw[k]) for k in sorted(kw.keys())]
        ).strip()
        self.kwl = [(k.replace("_", "-"), kw[k][1:]) for k in sorted(kw.keys())]

    def __str__(self) -> str:
        return "%s { %s; }" % (self.klass, self.kw)


def index(r: int, g: int, b: int) -> str:
    """
    Implements the 6x6x6 color cube location of 8bit mode described at
    https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit
    """
    return str(16 + (r * 36) + (g * 6) + b)


def color_component(x: int) -> int:
    """
    Implements the 6x6x6 color cube values of 8bit mode described at
    https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit
    """
    if x == 0:
        return 0
    return 0x37 + (0x28 * x)


def color(r: int, g: int, b: int) -> str:
    return "#%.2x%.2x%.2x" % (
        color_component(r),
        color_component(g),
        color_component(b),
    )


def level(grey: int) -> str:
    """
    Implements 24 grey values of 8bit mode described at
    https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit
    """
    return "#%.2x%.2x%.2x" % (((grey * 10) + 8,) * 3)


def index2(grey: int) -> str:
    """
    Implements 24 grey location of 8bit mode described at
    https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit
    """
    return str(232 + grey)


# http://en.wikipedia.org/wiki/ANSI_escape_code#Colors
SCHEME = {
    # black red green brown/yellow blue magenta cyan grey/white
    "ansi2html": (
        "#000316",
        "#aa0000",
        "#00aa00",
        "#aa5500",
        "#0000aa",
        "#E850A8",
        "#00aaaa",
        "#F5F1DE",
        "#7f7f7f",
        "#ff0000",
        "#00ff00",
        "#ffff00",
        "#5c5cff",
        "#ff00ff",
        "#00ffff",
        "#ffffff",
    ),
    "xterm": (
        "#000000",
        "#cd0000",
        "#00cd00",
        "#cdcd00",
        "#0000ee",
        "#cd00cd",
        "#00cdcd",
        "#e5e5e5",
        "#7f7f7f",
        "#ff0000",
        "#00ff00",
        "#ffff00",
        "#5c5cff",
        "#ff00ff",
        "#00ffff",
        "#ffffff",
    ),
    # Based on the "osx" palette in previous versions of ansi2html.
    "osx": (
        "#000000",
        "#c23621",
        "#25bc24",
        "#adad27",
        "#492ee1",
        "#d338d3",
        "#33bbc8",
        "#cbcccd",
        "#404040",
        "#ff7661",
        "#65fc64",
        "#eded67",
        "#896eff",
        "#ff78ff",
        "#73fbff",
        "#ffffff",
    ),
    # Based on the "Basic" palette in macOS Terminal.
    "osx-basic": (
        "#000000",
        "#800000",
        "#008000",
        "#808000",
        "#000080",
        "#800080",
        "#008080",
        "#808080",
        "#666666",
        "#e60000",
        "#00d900",
        "#e6e600",
        "#0000ff",
        "#e600e6",
        "#00e6e6",
        "#e6e6e6",
    ),
    # Based on the "Solid Colors" palette in macOS Terminal.
    # The colors are brighter than osx-basic.
    "osx-solid-colors": (
        "#000000",
        "#990000",
        "#00a600",
        "#999900",
        "#0000b3",
        "#b300b3",
        "#00a6b3",
        "#bfbfbf",
        "#666666",
        "#e60000",
        "#00d900",
        "#e6e600",
        "#0000ff",
        "#e600e6",
        "#00e6e6",
        "#e6e6e6",
    ),
    # http://ethanschoonover.com/solarized
    "solarized": (
        "#262626",
        "#d70000",
        "#5f8700",
        "#af8700",
        "#0087ff",
        "#af005f",
        "#00afaf",
        "#e4e4e4",
        "#1c1c1c",
        "#d75f00",
        "#585858",
        "#626262",
        "#808080",
        "#5f5faf",
        "#8a8a8a",
        "#ffffd7",
    ),
    "mint-terminal": (
        "#2E3436",
        "#CC0000",
        "#4E9A06",
        "#C4A000",
        "#3465A4",
        "#75507B",
        "#06989A",
        "#D3D7CF",
        "#555753",
        "#EF2929",
        "#8AE234",
        "#FCE94F",
        "#729FCF",
        "#AD7FA8",
        "#34E2E2",
        "#EEEEEC",
    ),
    "dracula": (
        "#2E3436",
        "#FF5555",
        "#50FA7B",
        "#F1FA8C",
        "#BD93F9",
        "#FF79C6",
        "#8BE9FD",
        "#BFBFBF",
        "#4D4D4D",
        "#FF6E67",
        "#5AF78E",
        "#F4F99D",
        "#CAA9FA",
        "#FF92D0",
        "#9AEDFE",
        "#E6E6E6",
    ),
}

# to be filled in runtime, when truecolor found
truecolor_rules: List[Rule] = []


def intensify(color: str, dark_bg: bool, amount: int = 64) -> str:
    if not dark_bg:
        amount = -amount
    rgb = tuple(max(0, min(255, amount + int(color[i : i + 2], 16))) for i in (1, 3, 5))
    return "#%.2x%.2x%.2x" % rgb


def get_styles(
    dark_bg: bool = True,
    line_wrap: bool = True,
    scheme: str = "ansi2html",
) -> List[Rule]:
    css = [
        Rule(
            ".ansi2html-content",
            white_space=("pre", "pre-wrap")[line_wrap],
            word_wrap="break-word",
            display="inline",
        ),
        Rule(".body_foreground", color=("#000000", "#AAAAAA")[dark_bg]),
        Rule(".body_background", background_color=("#AAAAAA", "#000000")[dark_bg]),
        Rule(".inv_foreground", color=("#000000", "#AAAAAA")[not dark_bg]),
        Rule(".inv_background", background_color=("#AAAAAA", "#000000")[not dark_bg]),
        # These effects are "SGR (Select Graphic Rendition) parameters"
        # https://en.wikipedia.org/wiki/ANSI_escape_code#SGR_(Select_Graphic_Rendition)_parameters
        Rule(".ansi1", font_weight="bold"),
        Rule(".ansi2", font_weight="lighter"),
        Rule(".ansi3", font_style="italic"),
        Rule(".ansi4", text_decoration="underline"),
        Rule(".ansi5", text_decoration="blink"),
        Rule(".ansi6", text_decoration="blink"),
        Rule(".ansi8", visibility="hidden"),
        Rule(".ansi9", text_decoration="line-through"),
    ]

    try:
        pal = SCHEME[scheme]
    except KeyError as e:
        raise ValueError(f"Unsupported color scheme {scheme!r}") from e

    if len(pal) < 16:
        raise RuntimeError(
            f"Color scheme {scheme!r} specifies fewer than 16 colors. 16 colors are required."
        )

    # This is 8x2 palette of 3/4-bit color mode described at
    # https://en.wikipedia.org/wiki/ANSI_escape_code#3-bit_and_4-bit
    # .ansi{30..37} is foreground
    # .ansi{40..47} is background
    for _index in range(8):
        css.append(Rule(".ansi3%s" % _index, color=pal[_index]))
        css.append(Rule(".inv3%s" % _index, background_color=pal[_index]))
    for _index in range(8):
        css.append(Rule(".ansi4%s" % _index, background_color=pal[_index]))
        css.append(Rule(".inv4%s" % _index, color=pal[_index]))

    # This is the 8x2 bright(!) palette of 4-bit color mode described at
    # https://en.wikipedia.org/wiki/ANSI_escape_code#3-bit_and_4-bit
    # .ansi{90..97} is foreground
    # .ansi{100..107} is background
    for _index in range(8):
        css.append(Rule(".ansi9%s" % _index, color=pal[_index + 8]))
        css.append(Rule(".inv9%s" % _index, background_color=pal[_index + 8]))
    for _index in range(8):
        css.append(Rule(".ansi10%s" % _index, background_color=pal[_index + 8]))
        css.append(Rule(".inv10%s" % _index, color=pal[_index + 8]))

    # This is the first 16 palette slots of 8-bit color mode described at
    # https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit
    for _index in range(len(pal)):
        css.append(Rule(".ansi38-%s" % _index, color=pal[_index]))
        css.append(Rule(".inv38-%s" % _index, background_color=pal[_index]))
    for _index in range(len(pal)):
        css.append(Rule(".ansi48-%s" % _index, background_color=pal[_index]))
        css.append(Rule(".inv48-%s" % _index, color=pal[_index]))

    # css.append("/* Define the explicit color codes (obnoxious) */\n\n")

    # This is the 6x6x6 color cube of 8-bit mode described at
    # https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit
    # .ansi38-{16..231} is foreground
    # .ansi48-{16..231} is background
    for green in range(0, 6):
        for red in range(0, 6):
            for blue in range(0, 6):
                css.append(
                    Rule(
                        ".ansi38-%s" % index(red, green, blue),
                        color=color(red, green, blue),
                    )
                )
                css.append(
                    Rule(
                        ".inv38-%s" % index(red, green, blue),
                        background=color(red, green, blue),
                    )
                )
                css.append(
                    Rule(
                        ".ansi48-%s" % index(red, green, blue),
                        background=color(red, green, blue),
                    )
                )
                css.append(
                    Rule(
                        ".inv48-%s" % index(red, green, blue),
                        color=color(red, green, blue),
                    )
                )

    # This is the 24 shades of grey of 8-bit mode described at
    # https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit
    # .ansi38-{232..255} is foreground
    # .ansi48-{232..255} is background
    for grey in range(0, 24):
        css.append(Rule(".ansi38-%s" % index2(grey), color=level(grey)))
        css.append(Rule(".inv38-%s" % index2(grey), background=level(grey)))
        css.append(Rule(".ansi48-%s" % index2(grey), background=level(grey)))
        css.append(Rule(".inv48-%s" % index2(grey), color=level(grey)))

    css.extend(truecolor_rules)

    return css


# as truecolor encoding has 16 millions colors, adding only used colors during parsing
def add_truecolor_style_rule(
    is_foreground: bool, ansi_code: int, r: int, g: int, b: int, parameter: str
) -> None:
    rule_name = ".ansi{}-{}".format(ansi_code, parameter)
    color = "#{:02X}{:02X}{:02X}".format(r, g, b)
    if is_foreground:
        rule = Rule(rule_name, color=color)
    else:
        rule = Rule(rule_name, background_color=color)
    truecolor_rules.append(rule)


def pop_truecolor_styles() -> Dict[str, Rule]:
    global truecolor_rules  # pylint: disable=global-statement
    styles = dict([(item.klass.strip("."), item) for item in truecolor_rules])
    truecolor_rules = []
    return styles
