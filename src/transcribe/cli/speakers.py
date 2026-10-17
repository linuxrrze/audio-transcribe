"""`tx speakers ...`: known persons and their voiceprints."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from transcribe.core.config import Config
from transcribe.i18n import _, display_label
from transcribe.speakers.voiceprint_db import Person, VoiceprintDB
from transcribe.storage import session_store
from transcribe.storage.transcript_model import Transcript

speakers_app = typer.Typer(help=_("Manage known speakers (voiceprints)."),
                           no_args_is_help=True)
console = Console()
err = Console(stderr=True)

DbOption = Annotated[Path | None, typer.Option("--db", help=_("Voiceprint database path."))]
ConsentOpt = Annotated[bool, typer.Option(
    "--consent", help=_("The person agreed to storing their voiceprint (GDPR Art. 9)."))]
SessionArg = Annotated[Path, typer.Argument(exists=True, file_okay=False,
                                            help=_("Session folder."))]


def _open(path: Path | None) -> tuple[Config, VoiceprintDB]:
    from transcribe.speakers.identity import open_db

    config = Config.load()
    try:
        return config, open_db(config, path)
    except RuntimeError as e:
        err.print(f"[red]{e}[/]")
        raise typer.Exit(1) from e


def _require(store: VoiceprintDB, name: str) -> Person:
    p = store.find_person(name)
    if p is None:
        err.print(f"[red]{_('Unknown person: {name}').format(name=name)}[/]")
        raise typer.Exit(1)
    return p


def _fail(e: Exception) -> typer.Exit:
    err.print(f"[red]{e}[/]")
    return typer.Exit(1)


@speakers_app.command(
    "list", help=_("List known persons with their number of voiceprints."))
def speakers_list(db: DbOption = None) -> None:
    _config, store = _open(db)
    counts = store.voiceprint_counts()
    table = Table(_("ID"), _("Name"), _("Voiceprints"), _("Consent"), _("Created"))
    for p in store.persons():
        table.add_row(str(p.id), p.display_name, str(counts.get(p.id, 0)),
                      p.consent_at or "-", p.created_at)
    console.print(table)
    if store.cipher.encrypted:
        console.print(_("Voiceprints are encrypted at rest."))


@speakers_app.command(
    "add", help=_("Create a person entry (voiceprints are added with `enroll` or `name`)."))
def speakers_add(name: str, consent: ConsentOpt = False, db: DbOption = None) -> None:
    _config, store = _open(db)
    p = store.add_person(name, consent=consent)
    console.print(_("Added {name} (id {id})").format(name=p.display_name, id=p.id))


@speakers_app.command(
    "consent", help=_("Record (or withdraw) a person's consent to voiceprint storage."))
def speakers_consent(
    name: str,
    revoke: Annotated[bool, typer.Option(
        "--revoke",
        help=_("Withdraw consent: deletes all voiceprints of the person."))] = False,
    db: DbOption = None,
) -> None:
    _config, store = _open(db)
    p = _require(store, name)
    if revoke:
        store.delete_voiceprints_of(p.id)
        store.set_consent(p.id, False)
        console.print(_("Consent withdrawn; voiceprints of {name} deleted.").format(name=name))
    else:
        store.set_consent(p.id, True)
        console.print(_("Consent recorded for {name}.").format(name=name))


@speakers_app.command("rename", help=_("Rename a person."))
def speakers_rename(old: str, new: str, db: DbOption = None) -> None:
    _config, store = _open(db)
    store.rename_person(_require(store, old).id, new)
    console.print(_("Renamed {old} -> {new}").format(old=old, new=new))


@speakers_app.command(
    "merge", help=_("Move all voiceprints of DROP to KEEP and delete DROP."))
def speakers_merge(keep: str, drop: str, db: DbOption = None) -> None:
    _config, store = _open(db)
    store.merge_persons(_require(store, keep).id, _require(store, drop).id)
    console.print(_("Merged {drop} into {keep}").format(drop=drop, keep=keep))


@speakers_app.command(
    "delete", help=_("Delete a person and all of their voiceprints."))
def speakers_delete(
    name: str,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    db: DbOption = None,
) -> None:
    _config, store = _open(db)
    p = _require(store, name)
    if not yes:
        typer.confirm(_("Delete {name} and all voiceprints?").format(name=name), abort=True)
    store.delete_person(p.id)
    console.print(_("Deleted {name}").format(name=name))


def _record(seconds: float, device: str | None) -> np.ndarray:
    from transcribe.audio.mic import MicSource

    src = MicSource(device)
    src.open()
    parts, got = [], 0.0
    try:
        with err.status(_("Recording {seconds} s - please speak naturally...").format(
                seconds=f"{seconds:.0f}")):
            for frame in src.frames():
                parts.append(frame.samples)
                got += frame.duration
                if got >= seconds:
                    break
    finally:
        src.close()
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


@speakers_app.command(
    "enroll",
    help=_("Enroll a voiceprint from a file or a microphone recording (single speaker!)."))
def speakers_enroll(
    name: str,
    file: Annotated[Path | None, typer.Option(
        "--file", exists=True, dir_okay=False, help=_("Audio of the person speaking alone."))]
    = None,
    record: Annotated[float | None, typer.Option(
        "--record", min=5, help=_("Record this many seconds from the microphone."))] = None,
    input_device: Annotated[str | None, typer.Option("--input", "-i")] = None,
    consent: ConsentOpt = False,
    db: DbOption = None,
) -> None:
    from transcribe.audio.file import load_audio
    from transcribe.speakers.identity import enroll_audio

    if (file is None) == (record is None):
        err.print(f"[red]{_('Give exactly one of --file or --record.')}[/]")
        raise typer.Exit(2)
    config, store = _open(db)
    try:
        audio = load_audio(file) if file else _record(record or 20, input_device)
    except RuntimeError as e:
        raise _fail(e) from e
    try:
        n_chunks, added = enroll_audio(store, name, audio, consent, config,
                                       source=str(file) if file else "mic")
    except (ValueError, PermissionError) as e:
        raise _fail(e) from e
    console.print(_("Enrolled {name}: {added} new voiceprint(s) from {chunks} chunk(s).").format(
        name=name, added=added, chunks=n_chunks))


@speakers_app.command(
    "name", help=_("Name a speaker in a session (and learn their voice for future sessions)."))
def speakers_name(
    session: SessionArg,
    speaker_id: Annotated[str, typer.Argument(help=_("Session speaker id, e.g. S2."))],
    name: str,
    consent: ConsentOpt = False,
    enroll: Annotated[bool, typer.Option(
        help=_("Also store the voiceprint so the person is recognised next time."))] = True,
    db: DbOption = None,
) -> None:
    from transcribe.speakers.identity import name_speaker

    config, store = _open(db)
    try:
        added = name_speaker(session, session_store.resolve_transcript(session), speaker_id,
                             name, config, consent, enroll, db=store)
    except (KeyError, ValueError, PermissionError, FileNotFoundError, RuntimeError) as e:
        raise _fail(e) from e
    if enroll:
        console.print(_("{speaker} is now {name}; {added} voiceprint(s) stored.").format(
            speaker=speaker_id, name=name, added=added))
    else:
        console.print(_("{speaker} is now {name}.").format(speaker=speaker_id, name=name))


@speakers_app.command(
    "identify", help=_("Re-identify a session's speakers against the current voiceprints."))
def speakers_identify(session: SessionArg, db: DbOption = None) -> None:
    from transcribe.speakers.crypto import make_cipher
    from transcribe.speakers.identity import (
        apply_matches,
        group,
        load_exemplars,
        make_matcher,
        match_speakers,
        weak_speakers,
    )

    config, store = _open(db)
    path = session_store.resolve_transcript(session)
    transcript = Transcript.load(path)
    try:
        model_id, exemplars = load_exemplars(session,
                                             make_cipher(config.speakers.encrypt_voiceprints))
    except (FileNotFoundError, RuntimeError) as e:
        raise _fail(e) from e
    matches = match_speakers(make_matcher(store, model_id, config), group(exemplars),
                             weak_speakers(exemplars))
    apply_matches(transcript, matches)
    transcript.save(path)
    table = Table(_("Speaker"), _("Label"), _("Suggestion"), _("Score"))
    for sid, spk in transcript.speakers.items():
        table.add_row(sid, display_label(spk.label), spk.suggestion or "-",
                      f"{spk.score:.2f}" if spk.score is not None else "-")
    console.print(table)


@speakers_app.command(
    "export", help=_("Export persons and voiceprints to JSON (unencrypted - keep the file safe!)."))
def speakers_export(
    out: Path,
    names: Annotated[list[str] | None, typer.Option("--name", help=_("Only these persons."))]
    = None,
    db: DbOption = None,
) -> None:
    _config, store = _open(db)
    n = store.export_json(out, names)
    console.print(_("Exported {n} voiceprint(s) to {path}").format(n=n, path=out))


@speakers_app.command(
    "import", help=_("Merge an export into the local voiceprint DB (persons matched by name)."))
def speakers_import(file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
                    db: DbOption = None) -> None:
    _config, store = _open(db)
    try:
        persons, vps = store.import_json(file)
    except ValueError as e:
        raise _fail(e) from e
    console.print(_("Imported {n} voiceprint(s) for {persons} person(s).").format(
        n=vps, persons=persons))
