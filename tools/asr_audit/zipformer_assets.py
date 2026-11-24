"""Find the Urdu Zipformer INT8 release used for validation."""
from pathlib import Path

from common import ROOT
from fleurs_dev import generated_rows
from urdu_core import sha


MODEL_ID = "mahwizzzz/Urdu-Streaming-Zipformer-20M"
TAG = "v1.2.0-rc1"
GRAPHS = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx")
RELEASE_FILES = [f"onnx/{name}" for name in GRAPHS] + ["tokenizer/tokens.txt"]
INFERENCE = Path("src/urdu_streaming_asr/inference.py")


def usable(path):
    """A Git LFS pointer or an empty file cannot be passed to the recognizer."""
    if not path.is_file():
        return False
    with path.open("rb") as file:
        prefix = file.read(80)
    return bool(prefix) and not prefix.startswith(b"version https://git-lfs.github.com/spec/v1")


def local_pair(base, root, explicit_tokens):
    if not all(usable(base / name) for name in GRAPHS):
        return None
    if explicit_tokens:
        return base, explicit_tokens
    candidates = [base / "tokens.txt"]
    if base.name == "onnx":
        candidates.extend((base.parent / "tokenizer/tokens.txt", base.parent / "tokens.txt"))
    candidates.append(root / "tokenizer/tokens.txt")
    token_file = next((p for p in candidates if usable(p)), None)
    return (base, token_file) if token_file else None


def hub_release(explicit_tokens):
    try:
        from huggingface_hub import snapshot_download, try_to_load_from_cache
    except ImportError:
        raise SystemExit("The local Zipformer release is incomplete. To fetch it in this environment, run "
                         "python -m pip install huggingface_hub and rerun dev.py. You can also supply "
                         "--onnx-dir and --tokens for local release files.") from None

    cached = try_to_load_from_cache(MODEL_ID, RELEASE_FILES[0], revision=TAG)
    # Keep the snapshot path. Resolving a file symlink here would return its blob directory.
    snapshot = Path(cached).parent.parent if isinstance(cached, str) else None
    if snapshot is None or not all(usable(snapshot / name) for name in RELEASE_FILES):
        print(f"Fetching the four INT8 inference files from {MODEL_ID} at {TAG}.", flush=True)
        try:
            snapshot = Path(snapshot_download(repo_id=MODEL_ID, revision=TAG, allow_patterns=RELEASE_FILES))
        except Exception as exc:
            response = getattr(exc, "response", None)
            if getattr(response, "status_code", None) in (401, 403) or type(exc).__name__ == "GatedRepoError":
                raise SystemExit(f"Hugging Face requires release access. Open https://huggingface.co/{MODEL_ID} "
                                 "with your authorized account, run hf auth login in this environment, "
                                 "then rerun dev.py. Existing local files also work with --onnx-dir and --tokens.") from None
            raise SystemExit(f"Could not fetch {MODEL_ID} at {TAG} ({type(exc).__name__}). "
                             "Check the connection or supply the local release with --onnx-dir and --tokens.") from exc
        missing = [name for name in RELEASE_FILES if not usable(snapshot / name)]
        if missing:
            raise SystemExit("The release snapshot is incomplete. Missing or unmaterialized files: " + ", ".join(missing))

    token_file = snapshot / "tokenizer/tokens.txt"
    if explicit_tokens and sha(explicit_tokens) != sha(token_file):
        raise SystemExit(f"The supplied --tokens file differs from {MODEL_ID} at {TAG}. "
                         "To use a different local checkpoint, pass its matching --onnx-dir.")
    return snapshot / "onnx", explicit_tokens or token_file


def resolve_release(asr_repo, onnx_dir, tokens):
    if asr_repo:
        places = (asr_repo.expanduser(),)
    else:
        places = (ROOT.parent / "Urdu-Streaming-ASR", ROOT.parent / "stream-asr",
                  Path.home() / "PROXIMA/MAHWIZ/stream-asr")
    checkouts = [p.resolve() for p in places if (p / INFERENCE).is_file()]
    if not checkouts:
        raise SystemExit(f"Pass --asr-repo with your Urdu-Streaming-ASR checkout containing {INFERENCE}.")
    if tokens:
        tokens = tokens.expanduser().absolute()
        if not usable(tokens):
            raise SystemExit(f"Missing, empty or Git LFS pointer file for --tokens: {tokens}")

    if onnx_dir:
        root = onnx_dir.expanduser().absolute()
        for base in (root, root / "onnx"):
            pair = local_pair(base, root, tokens)
            if pair:
                return checkouts[0], *pair
        raise SystemExit(f"Could not find the complete INT8 release at {root}. "
                         "Set --onnx-dir to the directory with " + ", ".join(GRAPHS) +
                         " and --tokens to its matching tokens.txt.")

    for checkout in checkouts:
        for name in ("release/onnx", "models/onnx", "models", "onnx", ""):
            pair = local_pair(checkout / name, checkout, tokens)
            if pair:
                return checkout, *pair
    for root in (ROOT / "work" / f"zipformer-{TAG}", ROOT.parent / "Urdu-Streaming-Zipformer-20M"):
        pair = local_pair(root / "onnx", root, tokens)
        if pair:
            return checkouts[0], *pair
    return checkouts[0], *hub_release(tokens)


def validation_rows(work, asr_repo, onnx_dir, tokens, parquet):
    checkout, graphs, tokens = resolve_release(asr_repo, onnx_dir, tokens)
    print(f"Zipformer source: {checkout}\nINT8 graphs: {graphs}\nToken table: {tokens}", flush=True)
    return generated_rows(work, checkout, graphs, tokens, parquet)
