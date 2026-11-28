# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "rich",
#   "pillow",
#   "pixelmatch"
# ]
# ///

"""Compare gallery example images between two doc builds.

Compares 'sphx_glr_plot_*.png' images between the 'dev' build of the doc
and a baseline build
(published 'stable' by default, or any other version),
both published on nilearn.github.io.
Gives an idea of how much example outputs visually change
across nilearn versions.

See https://github.com/nilearn/nilearn/issues/6342

The scripts expects the documentation repo to be in a tmp directory.
Use a sparse checkout for speed:

    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/nilearn/nilearn.github.io.git \
        tmp/nilearn.github.io
    git -C tmp/nilearn.github.io sparse-checkout set dev/_images stable/_images


"""

import re
import sys
from argparse import ArgumentParser
from fnmatch import fnmatchcase
from pathlib import Path

from PIL import Image
from pixelmatch.contrib.PIL import pixelmatch
from rich.console import Console
from rich.markup import escape

# expects a clone of nilearn/nilearn.github.io at this path, with at least
# dev/_images and <baseline>/_images checked out (see CI workflow;
# <baseline> is 'stable' by default, see --baseline)
ROOT = Path(__file__).resolve().parent.parent
TMP_DIR = ROOT / "tmp"
CLONE_DIR = TMP_DIR / "nilearn.github.io"
DIFF_DIR = TMP_DIR / "doc_image_diffs"
IGNORE_FILE = Path(__file__).resolve().parent / "compare_doc_images_ignore.txt"

# per-pixel color threshold, same value used in tests/js/template.js
PIXELMATCH_THRESHOLD = 0.2
# flag images with more than 1% of pixels differing
DIFF_RATIO_TOLERANCE = 0.01

# disable ReprHighlighter (auto-styling of numbers/paths) and word-wrapping
# (which would break long URLs across lines) to keep this report's own
# markup as the only styling and its lines intact
console = Console(highlight=False, soft_wrap=True)
console_err = Console(stderr=True, highlight=False, soft_wrap=True)


def load_ignore_patterns(file_path):
    """Load glob patterns from an ignore file.

    One pattern per line; blank lines and lines starting with '#' are
    skipped. Returns an empty list if the file doesn't exist.
    """
    if not file_path.exists():
        return []
    lines = file_path.read_text().splitlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]


def is_ignored(name, patterns):
    """Check whether a file name matches any of the given glob patterns."""
    return any(fnmatchcase(name, pattern) for pattern in patterns)


def example_name(image_name):
    """Derive the example name from one of its gallery image file names.

    e.g. 'sphx_glr_plot_foo_001.png' -> 'plot_foo', by stripping the
    'sphx_glr_' prefix and the trailing '_<number>.png' index. Used to
    group an example's numbered outputs together and to locate its
    source script.
    """
    name = image_name.removeprefix("sphx_glr_")
    return re.sub(r"_\d+\.png$", "", name)


_example_path_cache = {}


def find_example_rel_path(name):
    """Find the path of the script that generates a given example.

    Walks the 'examples/' directory tree. Results are memoized in
    `_example_path_cache` since the same example is looked up once per
    group of changed images.

    Returns the example script path, relative to the 'examples/'
    directory, or `None` if it could not be found.
    """
    if name in _example_path_cache:
        return _example_path_cache[name]

    examples_dir = ROOT / "examples"
    found = None
    for path in examples_dir.rglob(f"{name}.py"):
        found = path.relative_to(examples_dir)
        break

    _example_path_cache[name] = found
    return found


def dev_doc_url(rel_path):
    """Build the URL of the dev doc page for a given example.

    e.g. '01_plotting/plot_haxby_masks.py' ->
    https://nilearn.github.io/dev/auto_examples/01_plotting/plot_haxby_masks.html#sphx-glr-auto-examples-01-plotting-plot-haxby-masks-py
    """
    posix_path = rel_path.as_posix()
    without_ext = posix_path.removesuffix(".py")
    anchor = "sphx-glr-auto-examples-" + posix_path.replace("/", "-").replace(
        "_", "-"
    ).replace(".", "-")
    return f"https://nilearn.github.io/dev/auto_examples/{without_ext}.html#{anchor}"


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = ArgumentParser(
        description=(
            "Compare gallery example images between the 'dev' build of "
            "the doc and a baseline build, both published on "
            "nilearn.github.io."
        )
    )
    parser.add_argument(
        "--baseline",
        default="stable",
        help=(
            "Doc version to compare 'dev' against: the name of a "
            "directory at the root of the nilearn.github.io clone, "
            "e.g. 'stable' (default) or '0.10.0'."
        ),
    )
    return parser.parse_args(argv)


def check_clone(baseline, baseline_dir, dev_dir):
    """Exit with an error if the baseline/dev image directories are missing."""
    for directory in (baseline_dir, dev_dir):
        if not directory.exists():
            console_err.print(f"Expected directory not found: {directory}")
            console_err.print(
                "Clone nilearn/nilearn.github.io into "
                f"{CLONE_DIR} first "
                f"(with dev/_images and {baseline}/_images checked out)."
            )
            sys.exit(1)


def list_gallery_images(directory):
    """List gallery example images in a doc build's '_images' directory.

    Matches 'sphx_glr_plot_*.png', excluding thumbnails.
    """
    return [
        path.name
        for path in directory.iterdir()
        if path.name.startswith("sphx_glr_plot_")
        and path.name.endswith(".png")
        and not path.name.endswith("_thumb.png")
    ]


def compare_image(name, baseline_dir, dev_dir):
    """Compare one gallery image between the baseline and dev doc builds.

    Writes a pixelmatch diff image under `DIFF_DIR` whenever any pixel
    differs.

    Returns a dict with the image `name`, its `status`
    ('size-changed', 'changed' or 'unchanged') and its `diff_ratio`
    (`None` when the two images' dimensions differ, since pixelmatch
    cannot diff images of different sizes).
    """
    img_baseline = Image.open(baseline_dir / name)
    img_dev = Image.open(dev_dir / name)

    if img_baseline.size != img_dev.size:
        return {"name": name, "status": "size-changed", "diff_ratio": None}

    width, height = img_baseline.size
    img_diff = Image.new("RGBA", (width, height))
    num_diff_pixels = pixelmatch(
        img_baseline, img_dev, img_diff, threshold=PIXELMATCH_THRESHOLD
    )
    diff_ratio = num_diff_pixels / (width * height)

    if diff_ratio > 0:
        DIFF_DIR.mkdir(parents=True, exist_ok=True)
        img_diff.save(DIFF_DIR / name)

    status = "changed" if diff_ratio > DIFF_RATIO_TOLERANCE else "unchanged"
    return {"name": name, "status": status, "diff_ratio": diff_ratio}


def print_file_list(label, names):
    """Print a labeled, sorted list of image file names.

    e.g. images only found in one of the two doc builds, which can be a
    sign that an example failed to build, was renamed, or was
    added/removed.
    """
    if not names:
        return
    console.print(f"\n[yellow]{len(names)} image(s) {label}:[/yellow]")
    for name in sorted(names):
        console.print(f"[yellow]  {name}[/yellow]")


def main():
    """Compare the gallery images of the baseline and dev doc builds.

    Prints a report grouped by example, and exits with a non-zero
    status if any image's pixel diff exceeds `DIFF_RATIO_TOLERANCE`
    (dimension-only changes are reported but don't affect the exit
    status).
    """
    args = parse_args()
    baseline = args.baseline

    baseline_dir = CLONE_DIR / baseline / "_images"
    dev_dir = CLONE_DIR / "dev" / "_images"

    check_clone(baseline, baseline_dir, dev_dir)

    baseline_images = set(list_gallery_images(baseline_dir))
    dev_images = set(list_gallery_images(dev_dir))

    only_in_baseline = baseline_images - dev_images
    only_in_dev = dev_images - baseline_images
    shared = baseline_images & dev_images

    ignore_patterns = load_ignore_patterns(IGNORE_FILE)
    ignored = {n for n in shared if is_ignored(n, ignore_patterns)}
    to_compare = shared - ignored

    console.print(
        f"\nCompared [bold]{len(to_compare)}[/bold] gallery image(s) "
        f"present in both {baseline} and dev."
    )
    print_file_list(f"only in {baseline} (removed in dev)", only_in_baseline)
    print_file_list(f"only in dev (new since {baseline})", only_in_dev)
    if ignored:
        console.print(
            f"[dim]{len(ignored)} image(s) ignored per "
            f"{IGNORE_FILE.name}.[/dim]"
        )

    results = sorted(
        (compare_image(name, baseline_dir, dev_dir) for name in to_compare),
        key=lambda r: (r["name"], -(r["diff_ratio"] or 0)),
    )

    changed = [r for r in results if r["status"] != "unchanged"]
    size_changed = [r for r in results if r["status"] == "size-changed"]
    pixel_changed = [r for r in results if r["status"] == "changed"]

    changed_header = (
        f"{len(changed)} image(s) changed beyond tolerance "
        f"({DIFF_RATIO_TOLERANCE * 100}% of pixels):"
    )
    header_style = "bold" if changed else "green"
    console.print(f"\n[{header_style}]{changed_header}[/{header_style}]\n")

    previous_example_name = None
    for r in changed:
        name = example_name(r["name"])
        if name != previous_example_name:
            if previous_example_name is not None:
                console.print()
            rel_path = find_example_rel_path(name)
            if rel_path:
                url = f"[cyan]{dev_doc_url(rel_path)}[/cyan]"
            else:
                url = "[dim](example script not found)[/dim]"
            console.print(f"[bold]{name}[/bold]: {url}")
            previous_example_name = name

        status = r["status"]
        tag_style = "yellow" if status == "changed" else "magenta"
        tag = f"[{tag_style}]{escape(f'[{status}]')}[/{tag_style}]"
        if r["diff_ratio"] is None:
            pct = "[dim]n/a[/dim]"
        else:
            pct_style = "red" if r["diff_ratio"] > 0.05 else "yellow"
            pct = f"[{pct_style}]{r['diff_ratio'] * 100:.2f}%[/{pct_style}]"
        console.print(f"  {tag} {r['name']} - {pct} pixels differ")

    if changed:
        console.print(f"[dim]\nDiff images written to {DIFF_DIR}[/dim]")

    if size_changed:
        console.print(
            f"\n[magenta]{len(size_changed)} image(s) changed dimensions "
            "(ignored for pass/fail).[/magenta]"
        )

    if pixel_changed:
        console.print(
            f"\n[bold red]FAIL: {len(pixel_changed)} image(s) exceed the "
            "pixel diff tolerance.[/bold red]"
        )
        sys.exit(1)
    else:
        console.print(
            "[green]\nPASS: no images exceed the pixel diff tolerance[/green]"
        )


if __name__ == "__main__":
    main()
