"""whisper.cpp engine against a fake whisper-server (no whisper.cpp build needed)."""

import json
import sys
import textwrap

import numpy as np
import pytest

from transcribe.asr import whisper_cpp as wc
from transcribe.audio.base import SAMPLE_RATE
from transcribe.core import models
from transcribe.core.config import Config, required_models
from transcribe.storage.transcript_model import Word

FAKE_SERVER = textwrap.dedent('''
    import email, json, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    args = sys.argv[1:]
    port = int(args[args.index("--port") + 1])
    with open(args[args.index("--record") + 1], "w") as f:
        json.dump(args, f)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"status": "ok"})

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            msg = email.message_from_bytes(
                b"Content-Type: " + self.headers["Content-Type"].encode() + b"\\r\\n\\r\\n" + body)
            fields = {part.get_param("name", header="content-disposition"):
                      part.get_payload(decode=True).decode()
                      for part in msg.get_payload() if not part.get_filename()}
            print("whisper_print_progress_callback: progress =  50%", flush=True)
            if fields.get("duration") == "50":  # language probabilities only
                return self._send({"language_probabilities": {"de": 0.6, "en": 0.2,
                                                              "fr": 0.2}})
            # times on the cut (speech-only) timeline, as whisper-server returns them
            words = [{"word": " Hal", "start": 0.0, "end": 0.2, "probability": 0.8},
                     {"word": "lo", "start": 0.2, "end": 0.4, "probability": 0.6},
                     {"word": ",", "start": 0.4, "end": 9.0, "probability": 0.9},
                     {"word": " Welt", "start": 1.2, "end": 1.6, "probability": 1.0},
                     {"word": ".", "start": 1.6, "end": 30.0, "probability": 1.0}]
            assert fields["language"] == "de" and fields["response_format"] == "verbose_json"
            self._send({"segments": [{"words": words}], "fields": fields})

    HTTPServer(("127.0.0.1", port), H).serve_forever()
''')


def tok(word, start, end, p=1.0, dtw=-1):
    return {"word": word, "start": start, "end": end, "probability": p, "t_dtw": dtw}


def test_tokens_merge_into_words_and_long_ends_are_capped():
    seg = {"words": [tok(" Guten", 0.0, 0.3), tok(" Mor", 0.3, 0.5, 0.5), tok("gen", 0.5, 0.7),
                     tok(".", 0.7, 25.0), tok(" [BLANK_AUDIO]", 1.0, 2.0),
                     tok(" Ja", 30.0, 30.2), tok("!", 30.2, 31.0)]}
    words = wc.tokens_to_words([seg])
    assert [w.text for w in words] == [" Guten", " Morgen.", " Ja!"]
    assert words[1].t0 == 0.3 and words[1].t1 == pytest.approx(0.7)  # "." did not extend it
    assert words[1].confidence == pytest.approx((0.5 + 1 + 1) / 3)
    assert all(w.t1 - w.t0 <= 0.5 + 1e-9 for w in words)


def test_dtw_times_mark_token_ends():
    # t_dtw marks where a token ends; a word starts where the previous one ended
    seg = {"words": [tok(" Okay", 0.0, 0.3, dtw=150), tok(",", 0.3, 0.4, dtw=160),
                     tok(" every", 0.4, 0.6, dtw=180), tok("body", 0.6, 0.9, dtw=210),
                     tok(" ready", 5.0, 5.4, dtw=900)]}
    words = wc.tokens_to_words([seg], use_dtw=True)
    assert [w.text for w in words] == [" Okay,", " everybody", " ready"]
    assert words[0].t1 == 1.5  # the comma's time is not the end of the spoken word
    assert (words[1].t0, words[1].t1) == (1.5, 2.1)
    # after a pause: no earlier than the longest plausible duration of "ready"
    assert words[2].t1 == 9.0 and words[2].t0 == pytest.approx(9.0 - (0.15 + 5 * 0.06))
    # without DTW data (e.g. a server built without it) token times are used
    plain = wc.tokens_to_words([{"words": [tok(" a", 0.0, 0.5), tok(" b", 0.5, 1.0)]}],
                               use_dtw=True)
    assert [w.t0 for w in plain] == [0.0, 0.5]


def test_cut_speech_and_time_map():
    audio = np.zeros(10 * SAMPLE_RATE, np.float32)
    speech, pieces = wc.cut_speech(audio, [(1.0, 2.0), (5.0, 7.5)])
    assert len(speech) == int(3.5 * SAMPLE_RATE)
    assert wc.to_original(0.5, pieces) == pytest.approx(1.5)
    assert wc.to_original(1.0, pieces) == pytest.approx(5.0)  # boundary -> next region
    assert wc.to_original(2.0, pieces) == pytest.approx(6.0)
    assert wc.to_original(99.0, pieces) == pytest.approx(7.5)
    assert wc.cut_speech(audio, [])[1] == []


def test_dtw_presets():
    assert wc.dtw_preset("large-v3-turbo") == "large.v3.turbo"
    assert wc.dtw_preset("small") == "small"
    assert wc.dtw_preset("/m/ggml-base.en-q5_1.bin") == "base.en"
    assert wc.dtw_preset("my-finetune") is None


def test_find_server_prefers_gpu_builds(tmp_path, monkeypatch):
    monkeypatch.setattr(wc, "build_dir", lambda: tmp_path)
    monkeypatch.setattr(wc.shutil, "which", lambda name: None)
    exe = wc.SERVER_NAME + (".exe" if sys.platform == "win32" else "")
    with pytest.raises(FileNotFoundError, match="build_whispercpp"):
        wc.find_server()
    for build in ("build-cpu", "build-vulkan"):
        (tmp_path / build / "bin").mkdir(parents=True)
        (tmp_path / build / "bin" / exe).write_text("")
    assert wc.find_server().parent.parent.name == "build-vulkan"
    assert wc.find_server(device="cpu").parent.parent.name == "build-cpu"
    with pytest.raises(FileNotFoundError):
        wc.find_server(str(tmp_path / "missing"))


def test_required_models_for_whisper_cpp(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_MODELS_DIR", str(tmp_path))
    c = Config()
    c.asr.engine, c.live.refine_model = "whisper.cpp", "large-v3-turbo"
    whisper, assets = required_models(c)
    assert whisper == ["ggml:small", "ggml:large-v3-turbo"]
    models.ggml_path("small").parent.mkdir(parents=True)
    models.ggml_path("small").write_bytes(b"x")
    assert models.missing_models(whisper, []) == ["ggml:large-v3-turbo"]


@pytest.fixture
def engine(tmp_path, monkeypatch):
    script, record = tmp_path / "fake_server.py", tmp_path / "args.json"
    script.write_text(FAKE_SERVER)
    model = tmp_path / "ggml-small.bin"
    model.write_bytes(b"")
    monkeypatch.setattr(wc, "_launcher", lambda binary: [sys.executable, "-W", "ignore",
                                                         str(script), "--record", str(record)])
    monkeypatch.setattr("transcribe.vad.silero.speech_regions",
                        lambda audio, **kw: [(1.0, 2.0), (5.0, 9.0)])
    e = wc.WhisperCppEngine(str(model), device="cpu", hotwords="Fentriss, DSGVO",
                            server=str(script))
    e.record = record
    yield e
    e.close()


def test_engine_end_to_end(engine):
    args = json.loads(engine.record.read_text())
    assert "-ng" in args and "-sns" in args
    assert args[args.index("-dtw") + 1] == "small" and "-nfa" in args
    audio = np.zeros(10 * SAMPLE_RATE, np.float32)
    seen = []
    words = engine.transcribe(audio, offset=100.0, language="de", progress=seen.append)
    assert [w.text for w in words] == [" Hallo,", " Welt."]
    # cut timeline 0.0 -> 1.0 s and 1.2 -> 5.2 s (second region starts at 1.0 on the cut)
    assert words[0].t0 == pytest.approx(101.0) and words[1].t0 == pytest.approx(105.2)
    assert all(w.t1 - w.t0 <= 0.8 for w in words)
    assert seen[-1] == pytest.approx(10.0) and any(s < 10.0 for s in seen[:-1])

    probs = engine.detect_language(audio, ["de", "en"])
    assert probs == pytest.approx({"de": 0.75, "en": 0.25})


def test_engine_request_options(engine, monkeypatch):
    sent = {}
    monkeypatch.setattr(engine._server, "post",
                        lambda fields, audio, progress=None: sent.update(fields) or {})
    audio = np.zeros(10 * SAMPLE_RATE, np.float32)
    engine.transcribe(audio, language="de")
    assert sent["prompt"] == "Fentriss, DSGVO Hallo zusammen. Also, dann fangen wir an."
    assert sent["carry_initial_prompt"] == "true" and sent["beam_size"] == "5"
    assert sent["max_context"] == str(len(sent["prompt"].encode()) + 1)  # no vocab in file
    assert sent["temperature_inc"] == "0.2"
    live = engine.with_options(beam_size=1, temperature_fallback=False)
    live.transcribe(audio, language="en", prompt="Earlier text.")
    assert sent["beam_size"] == "1" and sent["temperature_inc"] == "0.0"
    assert sent["prompt"] == "Fentriss, DSGVO Earlier text."
    assert live._server is engine._server


def test_tokenizer_matches_whisper_cpp_greedy_longest_match():
    vocab = {b"Hel", b"lo", b"Hello", b",", b" every", b" everyone", b" Gr", b"\xc3\xbc",
             b"\xc3\x9f", b"e", b".", b" 12", b" ", b"  "}
    assert wc.tokenize("Hello, everyone.", vocab) == [b"Hello", b",", b" everyone", b"."]
    assert wc.tokenize(" Grüße 12", vocab) == [b" Gr", b"\xc3\xbc", b"\xc3\x9f", b"e", b" 12"]
    assert wc.read_vocab(__file__) == set()  # not a ggml file


def test_server_startup_failure_reports_log(tmp_path, monkeypatch):
    script = tmp_path / "broken.py"
    script.write_text("import sys; print('error: failed to load model'); sys.exit(3)")
    monkeypatch.setattr(wc, "_launcher", lambda binary: [sys.executable, str(script)])
    model = tmp_path / "ggml-x.bin"
    model.write_bytes(b"")
    with pytest.raises(RuntimeError, match="failed to load model"):
        wc.WhisperCppEngine(str(model), server=str(script))


def test_window_bounds_cut_at_quietest_point():
    sr = SAMPLE_RATE
    speech = np.full(70 * sr, 0.5, np.float32)
    speech[int(24.0 * sr):int(24.3 * sr)] = 0.0  # pause inside the 20-28 s search range
    bounds = wc.window_bounds(speech)
    assert bounds[0] == 0 and bounds[-1] == len(speech)
    assert 24.0 * sr <= bounds[1] <= 24.3 * sr
    assert all(b - a <= 28 * sr for a, b in zip(bounds, bounds[1:], strict=False))
    assert wc.window_bounds(speech[: 10 * sr]) == [0, 10 * sr]


def test_previous_text_only_when_well_formed():
    def words(text):
        return [Word(" " + t, 0, 1) for t in text.split()]

    assert wc.previous_text(words("so we start now")) is None  # no punctuation
    assert wc.previous_text(words("it works. ok")) is None  # no capitals
    assert wc.previous_text(words("Okay. Let's go.")) == "Okay. Let's go."
    long = wc.previous_text(words("First sentence here. " * 20 + "Last one."))
    assert len(long) <= 200 and long.startswith("First") and long.endswith("Last one.")


def test_context_flows_between_windows(engine, monkeypatch):
    prompts = []
    replies = iter([" Hallo zusammen.", " und dann", " Weiter so."])

    def post(fields, audio, progress=None):
        prompts.append(fields.get("prompt"))
        return {"segments": [{"words": [tok(next(replies), 0.0, 0.5)]}]}

    monkeypatch.setattr(engine._server, "post", post)
    monkeypatch.setattr("transcribe.vad.silero.speech_regions",
                        lambda audio, **kw: [(0.0, len(audio) / SAMPLE_RATE)])
    engine.hotwords = None
    engine.transcribe(np.ones(56 * SAMPLE_RATE, np.float32) * 0.1, language="de")  # 3 windows
    style = wc.STYLE_PROMPTS["de"]
    assert prompts == [style, "Hallo zusammen.", style]  # "und dann" is not well-formed
