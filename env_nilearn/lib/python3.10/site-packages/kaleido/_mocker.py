from __future__ import annotations

import argparse
import asyncio
import multiprocessing
import sys
import time
import warnings
from pathlib import Path
from pprint import pp
from random import sample
from typing import TypedDict

import logistro
import orjson

import kaleido

_logger = logistro.getLogger(__name__)

cpus = multiprocessing.cpu_count()

# Extract jsons of mocks
test_dir = Path(__file__).resolve().parent.parent / "integration_tests"
in_dir = test_dir / "mocks"
out_dir = test_dir / "renders"


def _get_jsons_in_paths(path: str | Path) -> list[Path]:
    # Work with Paths and directories
    path = Path(path) if isinstance(path, str) else path

    if path.is_dir():
        _logger.info(f"Input is path {path}")
        return list(path.glob("*.json"))
    elif path.is_file():
        _logger.info(f"Input is file {path}")
        return [path]
    else:
        raise TypeError("--input must be file or directory")


class Param(TypedDict):
    name: str
    opts: dict[str, int | float]


def _load_figures_from_paths(paths: list[Path]):
    # Set json
    params: list[Param]
    for path in paths:
        if path.is_file():
            with path.open(encoding="utf-8") as file:
                figure = orjson.loads(file.read())
                _logger.info(f"Yielding {path.stem}")
                if args.parameterize_opts is False:
                    params = [
                        {
                            "name": f"{path.stem}.{args.format or 'png'}",
                            "opts": {
                                "scale": args.scale,
                                "width": args.width,
                                "height": args.height,
                            },
                        },
                    ]
                else:
                    widths = [args.width] if args.width else [200, 700, 1000]
                    heights = [args.height] if args.height else [200, 500, 1000]
                    scales = [args.scale] if args.scale else [0.5, 1, 2]
                    formats = (
                        [args.format]
                        if args.format
                        else [
                            "png",
                            "pdf",
                            "jpg",
                            "webp",
                            "svg",
                            "json",
                        ]
                    )
                    params = []
                    for w in widths:
                        for h in heights:
                            for s in scales:
                                for f in formats:
                                    params.append(
                                        {
                                            "name": (
                                                f"{path.stem!s}-{w!s}"
                                                f"x{h!s}X{s!s}.{f!s}"
                                            ),
                                            "opts": {
                                                "scale": s,
                                                "width": w,
                                                "height": h,
                                            },
                                        },
                                    )
                for p in params:
                    yield {
                        "fig": figure,
                        "path": str(Path(args.output) / p["name"]),
                        "opts": p["opts"],
                    }
        else:
            raise RuntimeError(f"Path {path} is not a file.")


# Set the arguments
description = """kaleido_mocker will load up json files of plotly figs and export them.

If you set multiple process, -n, non-headless mode won't function well because
chrome will actually throttle tabs or windows/visibile- unless that tab/window
is headless.

The export of the program is a json object containing information about the execution.
"""

if "--headless" in sys.argv and "--no-headless" in sys.argv:
    raise ValueError(
        "Choose either '--headless' or '--no-headless'.",
    )

parser = argparse.ArgumentParser(
    add_help=True,
    parents=[logistro.parser],
    conflict_handler="resolve",
    description=description,
)
parser.add_argument(
    "--logistro-level",
    default="INFO",
    dest="log",
    help="Set the logging level (default INFO)",
)
parser.add_argument(
    "--n",
    type=int,
    default=cpus,
    help="Number of tabs, defaults to # of cpus",
)
parser.add_argument(
    "--input",
    type=str,
    default=in_dir,
    help="Directory of mock file/s or single file (default tests/mocks)",
)
parser.add_argument(
    "--output",
    type=str,
    default=out_dir,
    help="DIRECTORY of mock file/s (default tests/renders)",
)
parser.add_argument(
    "--format",
    type=str,
    default=None,
    help="png (default), pdf, jpg, webp, svg, json",
)
parser.add_argument(
    "--width",
    type=str,
    default=None,
    help="width in pixels (default 700)",
)
parser.add_argument(
    "--height",
    type=str,
    default=None,
    help="height in pixels (default 500)",
)
parser.add_argument(
    "--scale",
    type=str,
    default=None,
    help="Scale ratio, acts as multiplier for height/width (default 1)",
)
parser.add_argument(
    "--parameterize_opts",
    action="store_true",
    default=False,
    help="Run mocks w/ different configurations.",
)
parser.add_argument(
    "--timeout",
    type=int,
    default=90,
    help="Set timeout in seconds for any 1 mock (default 60 seconds)",
)
parser.add_argument(
    "--headless",
    action="store_true",
    default=True,
    help="Set headless as True (default)",
)
parser.add_argument(
    "--no-headless",
    action="store_false",
    dest="headless",
    help="Set headless as False",
)
parser.add_argument(
    "--stepper",
    action="store_true",
    default=False,
    dest="stepper",
    help="Stepper sets n to 1, headless to False, no timeout "
    "and asks for confirmation before printing.",
)
parser.add_argument(
    "--random",
    type=int,
    default=0,
    help="Will select N random jsons- or if 0 (default), all.",
)
parser.add_argument(
    "--fail-fast",
    action="store_true",
    default=False,
    help="Throw first error encountered and stop execution.",
)

args = parser.parse_args()
logistro.getLogger().setLevel(args.log)

if not Path(args.output).is_dir():
    raise ValueError(f"Specified output must be existing directory. Is {args.output!s}")


# Function to process the images
async def _main(error_log=None, profiler=None):
    paths = _get_jsons_in_paths(args.input)
    if args.random:
        if args.random > len(paths):
            raise ValueError(
                f"Input discover {len(paths)} paths, but a sampling of"
                f"{args.random} was asked for.",
            )
        paths = sample(paths, args.random)
    if args.stepper:
        _logger.info("Setting stepper.")
        args.n = 1
        args.headless = False
        args.timeout = 0
        if args.format == "svg":
            warnings.warn(
                "Stepper won't render svgs. It's feasible, "
                "but the adaption is just slightly more involved.",
                stacklevel=1,
            )
            await asyncio.sleep(3)
        # sets a global in kaleido, gross huh

    async with kaleido.Kaleido(
        page_generator=kaleido.PageGenerator(force_cdn=True),
        n=args.n,
        headless=args.headless,
        timeout=args.timeout,
        stepper=args.stepper,
    ) as k:
        await k.write_fig_from_object(
            _load_figures_from_paths(paths),
            error_log=error_log,
            profiler=profiler,
        )


def build_mocks():
    start = time.perf_counter()
    try:
        error_log = [] if not args.fail_fast else None
        profiler = {}
        asyncio.run(_main(error_log, profiler))
    finally:
        # ruff: noqa: PLC0415
        from operator import itemgetter

        for tab, tab_profile in profiler.items():
            profiler[tab] = sorted(
                tab_profile,
                key=itemgetter("duration"),
                reverse=True,
            )

        elapsed = time.perf_counter() - start
        with_error_log = error_log is not None
        results = {
            "error_log": [str(log) for log in error_log] if with_error_log else None,
            "profiles": profiler,
            "total_time": f"Time taken: {elapsed:.6f} seconds",
            "total_errors": len(error_log) if with_error_log else "untracked",
        }
        pp(results)
        if error_log:
            sys.exit(1)


if __name__ == "__main__":
    build_mocks()
