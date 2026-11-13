"""Regression checks for README charts and their daily update workflow."""

import importlib.util
import io
import json
from datetime import date, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from xml.etree import ElementTree

import pytest

SPEC = importlib.util.spec_from_file_location(
    "update_star_history", Path(__file__).resolve().parents[1] / "scripts/update_star_history.py"
)
stars = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stars)


def history(*observations):
    return {"repository": "hojawn/Condila", "observations": list(observations)}


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_missing_data_is_a_card_without_a_fabricated_count(theme):
    svg = stars.render_chart(history(), theme)
    root = ElementTree.fromstring(svg)
    assert "Awaiting GitHub data" in svg
    _, _, width, height = map(float, root.get("viewBox").split())
    assert height / width < 0.3
    assert "0 stars" not in svg
    assert not root.findall(".//{http://www.w3.org/2000/svg}image")


@pytest.mark.parametrize("count,label", [(0, "0 stars"), (1, "1 star"), (1234, "1,234 stars")])
def test_first_observation_displays_the_actual_count(count, label):
    svg = stars.render_chart(history({"date": "2026-11-12", "stars": count}), "light")
    ElementTree.fromstring(svg)
    assert label in svg
    assert "First observation: 2026-11-12 UTC" in svg
    assert "another daily observation" in svg


def test_daily_updates_replace_same_day_and_preserve_decreases():
    saved = history({"date": "2026-11-11", "stars": 8})
    updated = stars.record_observation(saved, date(2026, 11, 11), 9)
    updated = stars.record_observation(updated, date(2026, 11, 12), 7)
    assert updated["observations"] == [
        {"date": "2026-11-11", "stars": 9},
        {"date": "2026-11-12", "stars": 7},
    ]
    assert saved["observations"][0]["stars"] == 8


@pytest.mark.parametrize("count", [-1, True, 1.5, None])
def test_invalid_counts_are_not_recorded(count):
    with pytest.raises(ValueError):
        stars.record_observation(history(), date(2026, 11, 12), count)


def test_corrupt_history_is_rejected(tmp_path):
    path = tmp_path / "history.json"
    duplicate = {"date": "2026-11-12", "stars": 3}
    for payload in [
        history(duplicate, duplicate),
        history({"date": "2026-99-12", "stars": 3}),
        {"repository": "another/repository", "observations": []},
    ]:
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            stars.load_history(path, "hojawn/Condila")


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("counts", [(0, 0), (1, 1), (5, 3)])
def test_two_dates_render_a_deterministic_chart_with_integer_ticks(theme, counts):
    saved = history(
        {"date": "2026-11-11", "stars": counts[0]}, {"date": "2026-11-12", "stars": counts[1]}
    )
    svg = stars.render_chart(saved, theme)
    root = ElementTree.fromstring(svg)
    assert "2026-11-11" in svg and "2026-11-12" in svg
    assert "GitHub stars" in svg
    assert "<!-- 0 -->" in svg
    assert "<!-- 0.5 -->" not in svg
    assert not root.findall(".//{http://www.w3.org/2000/svg}image")
    assert svg == stars.render_chart(saved, theme)


@pytest.mark.parametrize(
    "error",
    [
        HTTPError("https://api.github.com", 404, "Not Found", {}, None),
        URLError("network unavailable"),
        ValueError("invalid response"),
    ],
)
def test_failed_fetch_keeps_the_last_good_files(tmp_path, monkeypatch, error):
    stars.save_history(tmp_path, history({"date": "2026-11-11", "stars": 7}))
    folder = tmp_path / "docs/assets"
    before = {p.name: p.read_bytes() for p in folder.iterdir()}

    def fail(*args):
        raise error

    monkeypatch.setattr(stars, "fetch_star_count", fail)
    assert stars.main(["--root", str(tmp_path), "--repo", "hojawn/Condila"]) == 1
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before


def test_successful_update_stores_only_the_real_observation(tmp_path, monkeypatch):
    monkeypatch.setattr(stars, "fetch_star_count", lambda *args: 4)
    assert stars.main(["--root", str(tmp_path), "--repo", "hojawn/Condila"]) == 0
    saved = stars.load_history(tmp_path / "docs/assets/star-history.json", "hojawn/Condila")
    assert len(saved["observations"]) == 1
    assert saved["observations"][0]["stars"] == 4
    assert (tmp_path / "docs/assets/star-history.svg").exists()
    assert (tmp_path / "docs/assets/star-history-dark.svg").exists()


def test_failed_render_keeps_the_last_good_files(tmp_path, monkeypatch):
    stars.save_history(tmp_path, history({"date": "2026-11-11", "stars": 7}))
    folder = tmp_path / "docs/assets"
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    monkeypatch.setattr(stars, "fetch_star_count", lambda *args: 8)
    render = stars.render_chart

    def fail_on_dark(history, theme):
        if theme == "dark":
            raise ValueError("render failed")
        return render(history, theme)

    monkeypatch.setattr(stars, "render_chart", fail_on_dark)
    assert stars.main(["--root", str(tmp_path), "--repo", "hojawn/Condila"]) == 1
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before


def test_repository_metadata_request_uses_the_supplied_token(monkeypatch):
    def respond(request, timeout):
        assert request.full_url == "https://api.github.com/repos/hojawn/Condila"
        assert request.get_header("Authorization") == "Bearer test-token"
        assert timeout == 30
        return io.BytesIO(b'{"stargazers_count": 12}')

    monkeypatch.setattr(stars, "urlopen", respond)
    assert stars.fetch_star_count("hojawn/Condila", "test-token") == 12


def test_renamed_repository_keeps_history_and_renders_the_second_day(tmp_path, monkeypatch):
    saved = history({"date": "2026-11-12", "stars": 5})
    saved["repository_id"] = 1454437155
    stars.save_history(tmp_path, saved)

    class FixedDatetime:
        @staticmethod
        def now(tz):
            return datetime(2026, 11, 13, 8, tzinfo=tz)

    monkeypatch.setattr(stars, "datetime", FixedDatetime)
    monkeypatch.setenv("GITHUB_REPOSITORY", "hojawn/Open_CDLM")
    monkeypatch.setenv("GITHUB_REPOSITORY_ID", "1454437155")
    monkeypatch.setattr(stars, "fetch_star_count", lambda *args: 5)

    assert stars.main(["--root", str(tmp_path)]) == 0
    updated = stars.load_history(
        tmp_path / "docs/assets/star-history.json", "hojawn/Open_CDLM", 1454437155
    )
    assert updated["repository"] == "hojawn/Open_CDLM"
    assert updated["repository_id"] == 1454437155
    assert updated["observations"] == [
        {"date": "2026-11-12", "stars": 5},
        {"date": "2026-11-13", "stars": 5},
    ]
    for name in ["star-history.svg", "star-history-dark.svg"]:
        svg = (tmp_path / "docs/assets" / name).read_text(encoding="utf-8")
        assert "Star History" in svg
        assert "another daily observation" not in svg
        assert "2026-11-12" in svg and "2026-11-13" in svg


@pytest.mark.parametrize("name", ["hojawn/Condila", "hojawn/Open_CDLM"])
def test_different_repository_id_cannot_reuse_saved_history(tmp_path, name):
    saved = history({"date": "2026-11-12", "stars": 5})
    saved["repository_id"] = 1454437155
    stars.save_history(tmp_path, saved)
    with pytest.raises(ValueError, match="different repository ID"):
        stars.load_history(tmp_path / "docs/assets/star-history.json", name, 1454437156)


def test_offline_render_uses_the_saved_repository_name(tmp_path, monkeypatch):
    saved = {
        "repository": "hojawn/Open_CDLM",
        "repository_id": 1454437155,
        "observations": [{"date": "2026-11-12", "stars": 5}],
    }
    stars.save_history(tmp_path, saved)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY_ID", raising=False)
    assert stars.main(["--root", str(tmp_path), "--render-only"]) == 0
    assert json.loads((tmp_path / "docs/assets/star-history.json").read_text()) == saved
