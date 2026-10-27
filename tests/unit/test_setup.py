"""First-run setup: hardware -> model recommendation, config writing, downloads, `tx setup`."""

import threading

import pytest
from typer.testing import CliRunner

from transcribe.core import models
from transcribe.core import setup as core_setup
from transcribe.core.config import Config
from transcribe.core.setup import Hardware


@pytest.fixture
def cfg_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("TRANSCRIBE_MODELS_DIR", str(tmp_path / "models"))
    return tmp_path / "cfg" / "transcribe" / "config.toml"


@pytest.mark.parametrize("hw, key", [
    (Hardware(16), "balanced"),  # README default: small int8 on the CPU
    (Hardware(8), "balanced"),
    (Hardware(4), "fast"),  # few cores: base live, small afterwards
    (Hardware(4, cuda_devices=1), "cuda"),  # NVIDIA: large-v3-turbo
    (Hardware(16, gpu_build="vulkan"), "gpu"),  # other GPUs: whisper.cpp
    (Hardware(16, cuda_devices=1, gpu_build="vulkan"), "cuda"),
])
def test_recommendation(hw, key):
    rec = core_setup.recommend(hw)
    assert rec.key == key
    assert rec in core_setup.profiles_for(hw)


def test_choices_follow_hardware():
    keys = [p.key for p in core_setup.profiles_for(Hardware(16))]
    assert keys == ["fast", "balanced", "accurate"]  # no GPU options without a GPU
    assert "cuda" in [p.key for p in core_setup.profiles_for(Hardware(16, cuda_devices=2))]
    assert core_setup.PROFILES["gpu"].engine == "whisper.cpp"
    assert core_setup.PROFILES["cuda"].model == "large-v3-turbo"


def test_sizes():
    c = Config()
    # small + segmentation + TitaNet
    assert core_setup.profile_size_mb(core_setup.PROFILES["balanced"], c) == 485 + 7 + 40
    assert core_setup.size_mb("ggml:large-v3-turbo") == 1620
    assert core_setup.size_mb("whisper:org/custom-model") == 0
    assert core_setup.needed_models(core_setup.PROFILES["gpu"].applied(c))[:2] == [
        "ggml:small", "ggml:large-v3-turbo"]
    assert core_setup.format_size(1620) == "1.6 GB" and core_setup.format_size(0) == "?"


def test_apply_choices_writes_config(cfg_home):
    c = Config()
    c.asr.device = "cpu"
    core_setup.apply_choices(c, core_setup.PROFILES["accurate"], "de", ["en"],
                             store_voiceprints=False, identify_known=False)
    c.save()
    loaded = Config.load(cfg_home)
    assert (loaded.asr.model, loaded.live.refine_model) == ("small", "large-v3-turbo")
    assert loaded.asr.device == "cpu"  # the device is not the profile's business
    assert loaded.ui.language == "de" and loaded.asr.languages == ["en"]
    assert not loaded.speakers.store_voiceprints and not loaded.speakers.identify_known
    assert core_setup.matching_profile(loaded).key == "accurate"
    # None keeps values
    core_setup.apply_choices(loaded)
    assert loaded.asr.languages == ["en"] and loaded.ui.language == "de"


def test_needs_setup(cfg_home, monkeypatch):
    c = Config()
    monkeypatch.setattr(models, "missing_models", lambda *a: [])
    assert core_setup.needs_setup(c)  # no config file yet
    c.save()
    assert not core_setup.needs_setup(c)
    monkeypatch.setattr(models, "missing_models", lambda *a: ["whisper:small"])
    assert core_setup.needs_setup(c)


def test_download_all_reports_and_cancels(monkeypatch):
    calls = []

    def fake_download(name, progress):
        for done in (10, 20):
            progress(done, 20)
        calls.append(name)

    monkeypatch.setattr(models, "download", fake_download)
    seen = []
    models.download_all(["a", "b"], lambda *x: seen.append(x))
    assert calls == ["a", "b"]
    assert seen[-1] == ("b", 20, 20) and ("a", 10, 20) in seen

    cancel = threading.Event()

    def cancelling(name, done, total):
        cancel.set()

    calls.clear()
    with pytest.raises(models.Cancelled):
        models.download_all(["a", "b"], cancelling, cancel)
    assert calls == []  # aborted inside the first file


def test_hub_progress_counts_bytes():
    seen = []
    bar_cls = models.hub_progress(lambda done, total: seen.append((done, total)))
    files = bar_cls(total=3, unit="it")  # huggingface_hub's file counter: ignored
    files.update(1)
    data = bar_cls(total=0, unit="B", name="huggingface_hub.snapshot_download")
    data.total = 1000  # the hub grows the total as files start
    data.update(400)
    data.update(600)
    assert seen == [(400, 1000), (1000, 1000)]


def test_cli_setup_non_interactive(cfg_home, monkeypatch):
    from transcribe.cli.main import app

    monkeypatch.setattr(core_setup, "detect_hardware", lambda: Hardware(4))
    monkeypatch.setattr(models, "missing_models", lambda *a: ["whisper:base"])
    downloaded = []
    monkeypatch.setattr(models, "download_all",
                        lambda names, progress=None, cancel=None: downloaded.extend(names))
    result = CliRunner().invoke(app, ["setup", "-y", "--lang", "de", "--no-voiceprints"])
    assert result.exit_code == 0, result.output
    c = Config.load(cfg_home)
    assert (c.asr.model, c.live.refine_model) == ("base", "small")  # 4 cores: "fast"
    assert c.asr.languages == ["de"] and not c.speakers.store_voiceprints
    assert downloaded == ["whisper:base"]

    # an existing config keeps its model unless a profile is given
    result = CliRunner().invoke(app, ["setup", "-y", "--no-download"])
    assert result.exit_code == 0 and Config.load(cfg_home).asr.model == "base"
    result = CliRunner().invoke(app, ["setup", "-y", "--profile", "accurate",
                                      "--no-download"])
    assert Config.load(cfg_home).live.refine_model == "large-v3-turbo"
    assert CliRunner().invoke(app, ["setup", "-y", "--profile", "huge"]).exit_code == 1


def test_cli_setup_interactive(cfg_home, monkeypatch):
    from transcribe.cli import main

    monkeypatch.setattr(main, "_interactive", lambda: True)
    monkeypatch.setattr(core_setup, "detect_hardware", lambda: Hardware(16))
    monkeypatch.setattr(models, "missing_models", lambda *a: ["whisper:small"])
    downloaded = []
    monkeypatch.setattr(models, "download_all",
                        lambda names, progress=None, cancel=None: downloaded.extend(names))
    # model #3 (accurate), English only, German app, no voiceprints, no download now
    result = CliRunner().invoke(main.app, ["setup"], input="3\nen\nde\nn\nn\n")
    assert result.exit_code == 0, result.output
    assert "(recommended)" in result.output  # "balanced" for 16 cores
    c = Config.load(cfg_home)
    assert (c.asr.model, c.live.refine_model) == ("small", "large-v3-turbo")
    assert c.asr.languages == ["en"] and c.ui.language == "de"
    assert not c.speakers.store_voiceprints and not c.speakers.identify_known
    assert downloaded == []
    # second run: "keep the current settings" is offered; the defaults keep everything
    result = CliRunner().invoke(main.app, ["setup", "--no-download"], input="\n\n\n\n")
    assert result.exit_code == 0, result.output
    assert "Keep the current settings" in result.output
    assert Config.load(cfg_home).live.refine_model == "large-v3-turbo"
