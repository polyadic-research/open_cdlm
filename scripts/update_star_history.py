"""Record daily GitHub star counts and render self-contained README charts.

GitHub Actions supplies its repository token through GITHUB_TOKEN. Empty and
single-day histories render as cards; multiple dates render as a line chart.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SVG_STYLE = {
    "font.family": "DejaVu Sans",
    "svg.fonttype": "path",
    "svg.hashsalt": "condila-star-history",
}
THEMES = {
    "light": {
        "background": "#ffffff",
        "text": "#1f2328",
        "muted": "#59636e",
        "grid": "#d8dee4",
        "accent": "#7B4FE0",
        "border": "#d1d9e0",
    },
    "dark": {
        "background": "#0d1117",
        "text": "#f0f6fc",
        "muted": "#9198a1",
        "grid": "#30363d",
        "accent": "#a78bfa",
        "border": "#30363d",
    },
}


def validate_repository(repository):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("repository must use the owner/name format")
    return repository


def fetch_star_count(repository, token=None):
    """Read the current count from repository metadata, without third-party services."""
    validate_repository(repository)
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "condila-star-history"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(f"https://api.github.com/repos/{repository}", headers=headers)
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise TypeError("GitHub response must be a repository object")
    count = payload.get("stargazers_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("GitHub response has no valid stargazers_count")
    return count


def load_history(path, repository, repository_id=None):
    """Keep observations through a rename when the stable repository IDs match."""
    validate_repository(repository)
    history = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {"repository": repository, "observations": []}
    )
    if not isinstance(history, dict):
        raise TypeError("history must be an object")
    saved_id = history.get("repository_id")
    for value in (repository_id, saved_id):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError("repository_id must be a positive integer")
    if repository_id is not None and saved_id is not None and repository_id != saved_id:
        raise ValueError("history belongs to a different repository ID")
    same_id = repository_id is not None and repository_id == saved_id
    if history.get("repository", "").lower() != repository.lower() and not same_id:
        raise ValueError("history belongs to a different repository")
    observations = history.get("observations")
    if not isinstance(observations, list):
        raise TypeError("observations must be a list")
    seen = set()
    for item in observations:
        day = date.fromisoformat(item["date"])
        count = item.get("stars")
        if (
            item["date"] != day.isoformat()
            or day in seen
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise ValueError("history must contain unique dates and nonnegative integer counts")
        seen.add(day)
    result = {"repository": repository}
    resolved_id = repository_id if repository_id is not None else saved_id
    if resolved_id is not None:
        result["repository_id"] = resolved_id
    result["observations"] = sorted(observations, key=lambda item: item["date"])
    return result


def record_observation(history, day, count):
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("stars must be a nonnegative integer")
    observations = {item["date"]: dict(item) for item in history["observations"]}
    observations[day.isoformat()] = {"date": day.isoformat(), "stars": count}
    return {
        **history,
        "observations": [observations[key] for key in sorted(observations)],
    }


def render_card(history, theme):
    from matplotlib import rc_context
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyBboxPatch

    colors = THEMES[theme]
    observations = history["observations"]
    if observations:
        latest = observations[-1]
        count = latest["stars"]
        value = f"{count:,} {'star' if count == 1 else 'stars'}"
        message = "A trend line appears after another daily observation."
        footer = f"First observation: {latest['date']} UTC"
    else:
        value = "Awaiting GitHub data"
        message = "Star history appears after the first successful update."
        footer = "Daily observations"
    with rc_context(SVG_STYLE):
        figure = Figure(figsize=(10, 2.14), facecolor="none")
        figure.add_artist(
            FancyBboxPatch(
                (0.001, 0.005),
                0.998,
                0.99,
                boxstyle="round,pad=0,rounding_size=0.025",
                transform=figure.transFigure,
                facecolor=colors["background"],
                edgecolor=colors["border"],
                linewidth=0.7,
            )
        )
        figure.text(
            0.04,
            0.79,
            f"{history['repository']} / GitHub stars",
            color=colors["muted"],
            fontsize=10,
        )
        figure.text(0.04, 0.52, value, color=colors["text"], fontsize=22, fontweight="bold")
        figure.text(0.04, 0.34, message, color=colors["muted"], fontsize=10)
        figure.text(0.04, 0.13, "●", color=colors["accent"], fontsize=9)
        figure.text(0.06, 0.13, footer, color=colors["muted"], fontsize=9)
        return figure_svg(figure, history)


def figure_svg(figure, history):
    from matplotlib.backends.backend_svg import FigureCanvasSVG

    buffer = io.StringIO()
    FigureCanvasSVG(figure).print_svg(
        buffer,
        metadata={
            "Date": None,
            "Creator": "Condila",
            "Title": f"{history['repository']} star history",
        },
    )
    return "\n".join(line.rstrip() for line in buffer.getvalue().splitlines()) + "\n"


def render_chart(history, theme):
    """Draw only recorded dates and counts; no synthetic history or fractional dates."""
    if len(history["observations"]) < 2:
        return render_card(history, theme)

    from matplotlib import dates as mdates
    from matplotlib import rc_context
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    colors = THEMES[theme]
    observations = history["observations"]
    days = [date.fromisoformat(item["date"]) for item in observations]
    counts = [item["stars"] for item in observations]
    with rc_context(SVG_STYLE):
        figure = Figure(figsize=(10, 3.8), dpi=100, facecolor=colors["background"])
        axis = figure.add_axes((0.09, 0.22, 0.87, 0.51), facecolor=colors["background"])
        axis.plot(
            days,
            counts,
            color=colors["accent"],
            linewidth=2.6,
            marker="o" if len(days) <= 30 else None,
            markersize=4,
        )
        axis.fill_between(days, counts, 0, color=colors["accent"], alpha=0.08)
        axis.set_ylim(0, max(1, max(counts) * 1.15))
        span = (days[-1] - days[0]).days
        padding = max(0.15, span * 0.025)
        positions = mdates.date2num(days)
        axis.set_xlim(positions[0] - padding, positions[-1] + padding)
        tick_count = min(5, span + 1)
        ticks = [
            days[0] + timedelta(days=round(span * i / (tick_count - 1))) for i in range(tick_count)
        ]
        axis.set_xticks(mdates.date2num(ticks))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, min_n_ticks=2))
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axis.set_ylabel("GitHub stars", color=colors["muted"], labelpad=10)
        axis.tick_params(axis="both", colors=colors["muted"], labelsize=9, length=0, pad=8)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color=colors["grid"], linewidth=0.7, alpha=0.65)
        for side, spine in axis.spines.items():
            spine.set_visible(side == "bottom")
            spine.set_color(colors["grid"])
        figure.text(
            0.09, 0.91, "Star History", color=colors["text"], fontsize=16, fontweight="bold"
        )
        figure.text(0.09, 0.84, history["repository"], color=colors["muted"], fontsize=10)
        figure.text(
            0.96,
            0.90,
            f"{counts[-1]:,}",
            color=colors["accent"],
            fontsize=23,
            fontweight="bold",
            ha="right",
        )
        figure.text(0.96, 0.84, "stars", color=colors["muted"], fontsize=10, ha="right")
        figure.text(
            0.09,
            0.055,
            f"Daily observations / Updated {days[-1].isoformat()} UTC",
            color=colors["muted"],
            fontsize=9,
        )
        return figure_svg(figure, history)


def save_history(root, history):
    folder = root / "docs/assets"
    # Render both themes before replacing any files, preserving the last good
    # artifacts when rendering or fetching fails.
    outputs = {
        "star-history.json": json.dumps(history, indent=2) + "\n",
        "star-history.svg": render_chart(history, "light"),
        "star-history-dark.svg": render_chart(history, "dark"),
    }
    folder.mkdir(parents=True, exist_ok=True)
    pending = []
    for name, content in outputs.items():
        destination = folder / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        pending.append((temporary, destination))
    for temporary, destination in pending:
        temporary.replace(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="owner/name (defaults to GITHUB_REPOSITORY or the saved history)",
    )
    parser.add_argument(
        "--repo-id",
        type=int,
        default=os.environ.get("GITHUB_REPOSITORY_ID"),
        help="stable repository ID (defaults to GITHUB_REPOSITORY_ID)",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--render-only", action="store_true", help="render saved observations offline"
    )
    args = parser.parse_args(argv)
    try:
        path = args.root / "docs/assets/star-history.json"
        repository = args.repo
        if repository is None:
            repository = json.loads(path.read_text(encoding="utf-8"))["repository"]
        history = load_history(path, repository, args.repo_id)
        if not args.render_only:
            count = fetch_star_count(repository, os.environ.get("GITHUB_TOKEN"))
            history = record_observation(history, datetime.now(timezone.utc).date(), count)
        save_history(args.root, history)
    except HTTPError as exc:
        print(
            f"GitHub returned HTTP {exc.code}; existing star history was retained.", file=sys.stderr
        )
        return 1
    except (URLError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Could not update star history ({type(exc).__name__}).", file=sys.stderr)
        return 1
    print(f"Rendered star history from {len(history['observations'])} daily observations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
