

def _candidate_splice_bases(target_path: Path) -> list[Path]:
    """Return ordered candidate splice-base files for a given target.

    Prefer migrated `*_reference` files. Several papers in the dataset have a
    primary legacy `_golden` file that's been
    overwritten or polluted (e.g. embedded `<think>` reasoning text, or only
    the call sites kept). The original implementation usually survives in a
    sibling named `_golden_backup` or with a trailing `~` (editor backup).
    """
    reference = _reference_sibling(target_path)
    masked = target_path.with_name(target_path.stem + "_masked" + target_path.suffix)
    golden = target_path.with_name(target_path.stem + "_golden" + target_path.suffix)
    golden_backup = target_path.with_name(target_path.stem + "_golden_backup" + target_path.suffix)
    tilde = target_path.with_name(target_path.name + "~")
    return [p for p in (reference, golden, golden_backup, tilde, masked, target_path) if p.exists()]


def _function_present(src: str, name: str, language: str) -> bool:
    """True if function `name` can be located in `src` for splicing."""
    if language == "python":
        return _locate_python(src, name) is not None
    if language in ("c", "r"):
        return _locate_braced(src, name, language) is not None
    return False


def _load_prompt_meta(infer_dir: Path) -> dict:
    p = infer_dir / "prompt.json"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    return json.loads(_read(p))


def _load_extraction(infer_dir: Path) -> dict:
    p = infer_dir / "extraction.json"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    return json.loads(_read(p))


def process(
    infer_dir: Path,
    data_root: Path,
    out_dir: Optional[Path],
    dry_run: bool,
) -> RunResult:
    parsed = _parse_key(infer_dir.name)
    if not parsed:
        raise ValueError(f"cannot parse infer dir name: {infer_dir.name}")
    lang_short, sample_id, func_part, mode = parsed

    meta = _load_prompt_meta(infer_dir)
    if meta.get("language") != lang_short:
        raise ValueError(f"language mismatch in {infer_dir.name}: {meta.get('language')!r}")
    extraction = _load_extraction(infer_dir)

    sample_dir = _sample_dir(data_root, _LANG_DIR[lang_short], sample_id)
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"sample dir not found: {sample_dir}")

    # Per-function flow: meta["target_functions"] has length 1.
    # Group by file (still a length-1 list) for symmetry with the old code.
    by_file: dict[str, list[str]] = {}
    for tf in meta["target_functions"]:
        by_file.setdefault(tf["file_path"], []).append(tf["name"])

    file_results: list[FilePatchResult] = []
    for rel_file, fnames in by_file.items():
        file_results.append(
            _patch_one_file(
                sample_dir=sample_dir,
                rel_file=rel_file,
                fnames=fnames,
                extraction=extraction,
                infer_dir=infer_dir,
                language=lang_short,
                out_dir=out_dir,
                run_key=infer_dir.name,
                dry_run=dry_run,
            )
        )

    # When using --out-dir, drop a copy of prompt.json next to the patched
    # files so downstream stages (runner, report) can recover the function
    # name without needing the infer-root path.
    if out_dir is not None and not dry_run:
        try:
            (out_dir / infer_dir.name).mkdir(parents=True, exist_ok=True)
            shutil.copy2(infer_dir / "prompt.json",
                         out_dir / infer_dir.name / "prompt.json")
        except OSError:
            pass

    return RunResult(
        key=infer_dir.name,
        sample_id=sample_id,
        language=lang_short,
        mode=mode,
        files=file_results,
    )


def _patch_one_file(
    *,
    sample_dir: Path,
    rel_file: str,
    fnames: list[str],
    extraction: dict,
    infer_dir: Path,
    language: str,
    out_dir: Optional[Path],
    run_key: str,
    dry_run: bool,
) -> FilePatchResult:
    notes: list[str] = []
    target_path = sample_dir / rel_file
    masked_snapshot = _masked_sibling(target_path)
    reference_path = _reference_sibling(target_path)
    golden_path = _golden_sibling(target_path)

    # Per-function flow: splice into the reference file, not the dataset's
    # shipped masked stub. This guarantees that all OTHER functions in the
    # file (sibling targets, helpers) keep their reference bodies — the model's
    # output for this single function is the only non-golden code in the
    # patched file.
    #
    # When the primary golden has been overwritten or polluted (some papers
    # ship a `_golden_backup` or `~` that retains the original — e.g.
    # 51468-6 varmodel, 52886-2 CS_Consensus), pick whichever candidate
    # actually contains the target function. Fall through to the original
    # golden / masked priority order if none match.
    base_src_path: Optional[Path] = None
    if fnames:
        ranked = _candidate_splice_bases(target_path)
        primary_fn = fnames[0]
        for cand in ranked:
            try:
                if _function_present(_read(cand), primary_fn, language):
                    base_src_path = cand
                    notes.append(
                        f"splice base: {cand.name} (verified contains `{primary_fn}`)"
                    )
                    break
            except Exception:
                continue
    if base_src_path is not None:
        pass
    elif reference_path.exists():
        base_src_path = reference_path
        notes.append(f"splice base: reference sibling {reference_path.name}")
    elif golden_path.exists():
        base_src_path = golden_path
        notes.append(f"splice base: legacy golden sibling {golden_path.name}")
    elif masked_snapshot.exists():
        base_src_path = masked_snapshot
        notes.append(
            f"no golden sibling; fallback to masked snapshot {masked_snapshot.name}"
        )
    elif target_path.exists():
        base_src_path = target_path
        notes.append(
            f"no golden sibling or snapshot; fallback to current {target_path.name}"
        )
    else:
        return FilePatchResult(
            file_path=rel_file, functions=fnames, success=False,
            output_path=str(target_path), masked_path=str(masked_snapshot),
            notes=notes, error=f"source file not found: {target_path}",
        )

    src_text = _read(base_src_path)
    # Some shipped goldens were generated by an earlier model run that leaked
    # `<think>...</think>` reasoning blocks into the file (49566-6 fam_chol,
    # 45674-5 calc_energy_Fs, etc.). Strip those before splicing so the
    # whole-file brace validator doesn't trip on prose.
    if language in ("c", "r"):
        cleaned = _strip_prose(src_text)
        if cleaned != src_text:
            notes.append("stripped <think>/fence prose from polluted splice base")
            src_text = cleaned

    # Splice each function. Each splice rewrites the in-memory text; later
    # functions are located against the partially patched buffer, which is
    # safe because ranges are recomputed from scratch.
    for fname in fnames:
        ext_meta = extraction.get(fname, {})
        if not ext_meta:
            return FilePatchResult(
                file_path=rel_file, functions=fnames, success=False,
                output_path=str(target_path), masked_path=str(masked_snapshot),
                notes=notes, error=f"no extraction entry for `{fname}`",
            )
        if not ext_meta.get("signature_matches"):
            notes.append(f"warning: signature_matches=False for `{fname}`")
        ext_path = infer_dir / f"{fname}{_LANG_EXT[language]}"
        if not ext_path.exists():
            return FilePatchResult(
                file_path=rel_file, functions=fnames, success=False,
                output_path=str(target_path), masked_path=str(masked_snapshot),
                notes=notes, error=f"extracted code missing: {ext_path}",
            )
        ext_code = _read(ext_path)
        # Pre-validate the extracted snippet on its own (signature only — full
        # parse may fail because Python snippets have free-floating defs which
        # are valid; C snippets are just a function body and are also valid as
        # an isolated translation unit fragment, so we skip strict parse here
        # and rely on whole-file validation after splicing).
        patched, splice_notes = _splice(src_text, fname, language, ext_code)
        notes.extend(splice_notes)
        if patched is None:
            return FilePatchResult(
                file_path=rel_file, functions=fnames, success=False,
                output_path=str(target_path), masked_path=str(masked_snapshot),
                notes=notes,
                error=f"splice failed for `{fname}`: {splice_notes[-1] if splice_notes else 'unknown'}",
            )
        src_text = patched

    # Strict whole-file validation
    ok, why = _validate(src_text, language)
    if not ok:
        return FilePatchResult(
            file_path=rel_file, functions=fnames, success=False,
            output_path=str(target_path), masked_path=str(masked_snapshot),
            notes=notes, error=f"validation failed: {why}",
        )

    # Write outputs
    if out_dir is None:
        # In-place: snapshot masked once, write patched to target
        if not masked_snapshot.exists():
            if not dry_run:
                shutil.copy2(target_path, masked_snapshot)
            notes.append(f"snapshot {target_path.name} -> {masked_snapshot.name}")
        if not dry_run:
            _write(target_path, src_text)
        out_path = target_path
        masked_out_path = masked_snapshot
    else:
        # Side-tree: <out_dir>/<run_key>/<rel_file> for the patched file,
        # plus <out_dir>/<run_key>/<rel_file_with_masked_suffix> for the snapshot.
        out_path = out_dir / run_key / rel_file
        masked_out_path = _masked_sibling(out_path)
        if not dry_run:
            _write(out_path, src_text)
            if not masked_out_path.exists():
                _write(masked_out_path, _read(base_src_path))

    return FilePatchResult(
        file_path=rel_file, functions=fnames, success=True,
        output_path=str(out_path), masked_path=str(masked_out_path),
        notes=notes,
    )
