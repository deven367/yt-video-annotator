# Agent guide

Keep README focused on installation and app usage. Contributor internals and deployment learnings belong here.

## Repository and runtime

- Python 3.11+; keep `requires-python`, classifiers, README, and CI aligned. Build-backend compatibility matters too.
- `annotator/app.py`: YouTube word search and SRT generation. `annotator/pages/wspy.py`: YouTube/upload transcription. `annotator/playback.py`: source-scoped seek state and callbacks. `annotator/utils.py`: downloads, conversion, inference, and subtitle utilities.
- `pyproject.toml` is the application dependency source of truth. Inference uses `pywhispercpp`; do not restore the removed fastcore, OpenAI Whisper, or Torch dependencies without a new requirement.
- Keep the UI and pages inside the installed package. `run_app` resolves the packaged entrypoint, invokes Streamlit through the active interpreter, forwards arguments, and returns its exit status. Test it outside the checkout.
- Inspect the worktree before editing. Local notebooks, models, media, build outputs, and an untracked `uv.lock` may belong to the user. Stage explicit paths; do not clean or publish those files incidentally.

## Local verification

Use an isolated environment without modifying the user's lockfile:

```sh
VENV="$(mktemp -d "${TMPDIR:-/tmp}/yt-annotator-dev.XXXXXX")"
uv venv --python 3.11 "$VENV"
uv pip install --python "$VENV/bin/python" -e '.[dev]'
"$VENV/bin/python" -m pytest tests -q
git ls-files -z '*.py' | xargs -0 "$VENV/bin/ruff" check --select E4,E7,E9,F
```

FFmpeg must be on PATH. Lint tracked files only: untracked scratch Python files may intentionally be incomplete. Remove only temporary environments/artifacts you created.

CI uses setup-uv on Python 3.11, builds sdist/wheel, tests the installed wheel outside the checkout, and smoke-checks source/editable installations. For UI behavior, also run the actual app and inspect the browser; passing AppTest alone does not prove media playback moved.

The workflow runs for both pushes and PRs. Duplicate push runs have stalled in `apt-get` while the same-SHA PR run passed. Check the event, SHA, and active step before treating that as an application failure. Only cancel a redundant infrastructure-stalled run after the same-SHA PR run succeeded, and record why; never bypass failing tests.

## Behavioral contracts

- Whisper segments use **centiseconds**, not milliseconds. `df_from_result` produces formatted timestamps; SRT uses comma-separated milliseconds. Native media seeks round down to whole seconds. Cover zero, hour boundaries, and duplicate timestamps.
- Store completed results in session state and render outside transcription button handlers. Associate results with the source and model; changing either must clear stale results and playback positions.
- Seeking must not call download/inference again, even after the shared transcript cache is cleared. Preserve edited text and download controls across reruns.
- The playback revision changes the native player's container key so clicking the same timestamp seeks again. Segment expanders use a stable key and `on_change="rerun"` so they stay open after seeking. Do not remove these behaviors as cosmetic cleanup.
- Use unique button keys for repeated words/segments. Verify actual `audio`/YouTube player time, not just timestamp labels or an iframe URL; keyboard activation should work too.
- Handle download, model-loading, and inference failures without caching a failed operation as a successful transcript. A later retry must work.

## Audio lifecycle and cache limits

- Whisper input is 16 kHz mono, 16-bit PCM WAV. Reuse the existing FFmpeg conversion and blank-audio filtering paths.
- `get_audio` is a context manager: keep inference inside `with get_audio(url) as (audio_path, was_cached):`. The working path is valid only until context exit; do not persist it for later use.
- Converted WAV caching is bounded by 32 files and 512 MiB, retaining the most recently used files. Cleanup runs during acquisition and after publication. Temporary hard links protect in-progress audio from eviction or replacement and are removed on context exit, including errors.
- Oversized files are processed without retaining a cache entry; evicted entries must download again successfully. Active processing can exceed the cache budget. Unrelated audio, models, and SRT outputs are outside this policy.
- Main-page transcript caching uses a one-hour TTL and at most 32 entries. Session results remain independent of that shared cache.

## GitHub and Hugging Face delivery

- One issue per feature/fix PR; link its issue and verify before merging. Delete merged feature branches. Preserve unique unmerged work with an archive tag before deleting a stale branch unless the user explicitly requests discarding it.
- GitHub defaults to `origin/master`; the Space deploys `hf/main`. The `hf` remote is `git@hf.co:spaces/deven367/yt-video-annotator-hf`. Keep the active local `deploy/hf-main` branch when cleaning stale branches.
- Keep the existing **native Streamlit SDK** deployment; do not switch to Docker as part of an ordinary sync. README frontmatter selects Python 3.11, the tested Streamlit SDK, and `app_file: annotator/app.py`. `packages.txt` installs FFmpeg.
- The native SDK installs requirements **before copying app source**, so a local `.` requirement fails. `requirements.txt` installs a pinned GitHub source archive, while dependencies remain in `pyproject.toml`. HF Git partial clones failed with `expected 'packfile'` / missing promisor blobs; do not replace the archive with a self-cloning Git requirement.
- For application/dependency changes, commit the verified implementation and update the archive revision to that exact GitHub commit. The changed URL invalidates HF's dependency build cache. If the pin references an implementation commit inside the PR, preserve it with a merge commit rather than squash; alternatively, pin an already-merged source commit in a deployment PR. Docs-only changes do not need a new application pin.
- Preserve the Space's history: fetch both remotes, use `deploy/hf-main` (or a worktree based on `hf/main`), merge the verified GitHub `master`, and push `HEAD:main` to `hf`. Do not force-push GitHub's branch over the Space. Confirm the resulting file trees match.
- After pushing, use `hf spaces wait deven367/yt-video-annotator-hf --timeout 10m`, inspect build/run logs on failure, and confirm the running revision. Smoke the actual surface at https://deven367-yt-video-annotator-hf.hf.space; a successful Git push or RUNNING server alone is not feature verification.

## SSH and observed deployment limits

- HF can accept an unrelated agent key yet identify the connection as **anonymous**, preventing pushes and stopping SSH from trying the configured HF key. The verified fix was host-specific `IdentitiesOnly yes` under `Host hf.co`, not replacing the key. `ssh -T -o BatchMode=yes git@hf.co` should greet `deven367`; inspect `ssh -G git@hf.co` and `ssh-add -l` when diagnosing selection. Never print private keys or access tokens.
- Observed on 2026-10-26: HF-to-YouTube extraction failed with `SSL: UNEXPECTED_EOF_WHILE_READING`; local YouTube extraction/seeking and live HF uploaded-audio seeking passed. Treat this as a separate verification limit, not proof that live HF YouTube transcription works. Keep TLS verification enabled and report the failure rather than hiding it.
