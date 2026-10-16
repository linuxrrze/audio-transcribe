import hashlib
import io
import tarfile

import pytest

from transcribe.core import models


@pytest.fixture
def fake_assets(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    plain = src / "emb.onnx"
    plain.write_bytes(b"embedding")
    archive = src / "seg.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tar:
        data = b"segmentation"
        info = tarfile.TarInfo("seg-dir/model.onnx")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    assets = {
        "emb": models.ModelAsset("emb", "embedding", plain.as_uri(), sha(plain), "emb.onnx"),
        "seg": models.ModelAsset("seg", "segmentation", archive.as_uri(),
                                 sha(archive), "seg-dir/model.onnx"),
        "bad": models.ModelAsset("bad", "embedding", plain.as_uri(), "0" * 64, "emb.onnx"),
    }
    monkeypatch.setattr(models, "ASSETS", assets)
    monkeypatch.setenv("TRANSCRIBE_MODELS_DIR", str(tmp_path / "cache"))
    return assets


def test_ensure_file_and_archive(fake_assets):
    assert not models.is_present("emb")
    assert models.ensure("emb").read_bytes() == b"embedding"
    assert models.ensure("seg").read_bytes() == b"segmentation"
    assert models.is_present("emb") and models.is_present("seg")


def test_checksum_mismatch_leaves_nothing(fake_assets):
    with pytest.raises(RuntimeError, match="Checksum"):
        models.ensure("bad")
    assert not models.is_present("bad")
    assert not list(models.models_dir().glob(".bad*"))
