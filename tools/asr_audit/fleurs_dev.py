"""Build Urdu FLEURS validation predictions with the released Zipformer recognizer."""
import errno
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request
import numpy as np

from common import ROOT
from urdu_core import read_json, sha, write_json

REVISION = "d010211ed2d3d9281bcfed40dfdc5f57aceec472"
SHARD = "ur_pk/validation-00000-of-00001.parquet"
URL = f"https://huggingface.co/datasets/google/fleurs/resolve/{REVISION}/{SHARD}"
DIGEST = "71b968685451c0ac44435cb650ac67f30fc48ce5d78c7eb4484c03d23c61b316"
SIZE = 174091927
N_RECORDS = 267


def resolve_paths(repo, onnx_dir, tokens):
    if repo is None:
        places = (ROOT.parent / "Urdu-Streaming-ASR", ROOT.parent / "stream-asr",
                  Path.home() / "PROXIMA/MAHWIZ/stream-asr")
    else:
        places = (repo.expanduser(),)
    impl = Path("src/urdu_streaming_asr/inference.py")
    repo = next((p.resolve() for p in places if (p / impl).is_file()), None)
    if repo is None:
        raise ValueError("Pass --asr-repo with your Urdu-Streaming-ASR checkout containing " + str(impl))
    bases = ([onnx_dir.expanduser()] if onnx_dir else
             [repo / name for name in ("release/onnx", "models/onnx", "models", "onnx", "")])
    components = ("encoder", "decoder", "joiner")
    for base in bases:
        paths = {n: base / (n + ".int8.onnx") for n in components}
        if all(p.is_file() for p in paths.values()):
            paths["tokens"] = tokens.expanduser() if tokens else base / "tokens.txt"
            if not paths["tokens"].is_file():
                continue
            return repo, paths
    raise ValueError("Pass --onnx-dir with encoder.int8.onnx, decoder.int8.onnx, joiner.int8.onnx "
                     "and tokens.txt. Use --tokens if the matching release tokens.txt is elsewhere.")


def checked_shard(work, supplied):
    if supplied is not None:
        path = supplied.expanduser()
    else:
        path = work / "ur_pk-validation.parquet"
        if not path.is_file():
            print("Fetching the pinned Urdu FLEURS validation shard (174 MB).", flush=True)
            with tempfile.NamedTemporaryFile(dir=work, delete=False) as f:
                tmp = Path(f.name)
            try:
                request = urllib.request.Request(URL, headers={"User-Agent":"Kinnow-ASR-validation"})
                with urllib.request.urlopen(request, timeout=120) as src, tmp.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024*1024)
                if tmp.stat().st_size != SIZE or sha(tmp) != DIGEST:
                    raise ValueError("Downloaded validation shard failed its checksum.")
                os.replace(tmp, path)
                return path
            finally:
                tmp.unlink(missing_ok=True)
    if not path.is_file() or path.stat().st_size != SIZE or sha(path) != DIGEST:
        raise ValueError(f"Expected the official validation shard with SHA-256 {DIGEST} at {path}")
    return path


def recognize(recognizer, mono, sample_rate):
    if sample_rate != 16000 or mono.ndim != 1 or not len(mono):
        raise ValueError("Expected nonempty 16 kHz mono input.")
    recognizer.reset()
    for start in range(0, len(mono), 5120):
        recognizer.accept_chunk(mono[start:start + 5120], sample_rate)
    hypothesis = recognizer.finalize()
    if not isinstance(hypothesis, str):
        raise ValueError("Unexpected ASR final transcript type.")
    return hypothesis


def record_stream(parquet_path, recognizer):
    import pyarrow.parquet as pq
    import soundfile as sf
    pf = pq.ParquetFile(parquet_path)
    if pf.metadata.num_rows != N_RECORDS:
        raise ValueError("Unexpected number of Urdu validation recordings.")
    index = 0
    for batch in pf.iter_batches(batch_size=8, columns=["id","audio","raw_transcription"]):
        for r in batch.to_pylist():
            audio = r["audio"]
            if not isinstance(audio, dict) or not audio.get("bytes"):
                raise ValueError("Missing embedded audio in the pinned Parquet file.")
            payload = audio["bytes"]
            audio_sha = hashlib.sha256(payload).hexdigest()
            if r["id"] is None or not isinstance(r["raw_transcription"], str) or not r["raw_transcription"].strip():
                raise ValueError(f"Missing prompt ID or raw reference at {index}.")
            wave, sr = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
            hyp = recognize(recognizer, wave.mean(axis=1, dtype=np.float32), sr)
            yield {"id":f"fleurs-ur-validation/{index:04d}", "group":f"fleurs-prompt-{r['id']}",
                   "reference":r["raw_transcription"], "hypothesis":hyp, "audio_sha256":audio_sha}
            index += 1
            if index % 50 == 0 or index == N_RECORDS:
                print(f"Decoded {index}/{N_RECORDS} validation recordings.", flush=True)


def cached_rows(directory, inputs):
    manifest = read_json(directory / "provenance.json")
    rows = directory / "rows.jsonl"
    if (manifest.get("inputs") != inputs or manifest.get("recordings") != N_RECORDS
            or manifest.get("output_sha256") != sha(rows)):
        raise ValueError("Cached development inputs changed. Choose a fresh --work directory.")
    return rows


def generated_rows(work, repo, onnx_dir, tokens, parquet):
    repo, paths = resolve_paths(repo, onnx_dir, tokens)
    source = repo / "src/urdu_streaming_asr/inference.py"
    try:
        version = importlib.metadata.version("sherpa-onnx")
    except importlib.metadata.PackageNotFoundError:
        version = "unreported"
    signature = {"revision":REVISION, "shard_sha256":DIGEST, "split":"validation",
                 "precision":"int8","model_files":{k:sha(p) for k,p in paths.items()},
                 "recognizer_source_sha256":sha(source), "sherpa_onnx":version,
                 "decoder_script_sha256":sha(Path(__file__)),
                 "decoder":"modified_beam_search", "beam_paths":4, "blank_penalty":0.25,
                 "input_chunk_samples":5120,"sample_rate":16000,"provider":"cpu","threads":1,
                 "reference_column":"raw_transcription","group_unit":"FLEURS prompt"}
    dest = work / "fleurs-validation"
    if dest.exists():
        return cached_rows(dest, signature)
    missing = [package for module, package in (("pyarrow", "pyarrow"), ("soundfile", "soundfile"),
                                               ("sherpa_onnx", "sherpa-onnx"))
               if importlib.util.find_spec(module) is None]
    if missing:
        raise SystemExit("Audio decoding needs " + ", ".join(missing) +
                         " in the current Python environment. Use the sherpa-onnx version from your "
                         "Zipformer benchmark. Existing development predictions can be scored with --rows.")
    sys.path.insert(0, str(repo / "src"))
    from urdu_streaming_asr.inference import UrduStreamingRecognizer
    shard = checked_shard(work, parquet)
    rec = UrduStreamingRecognizer(encoder_path=paths["encoder"], decoder_path=paths["decoder"],
                                  joiner_path=paths["joiner"], tokens_path=paths["tokens"],
                                  num_threads=1, provider="cpu", decoding_method="modified_beam_search",
                                  max_active_paths=4, blank_penalty=0.25)
    stage = Path(tempfile.mkdtemp(dir=work, prefix="validation-"))
    try:
        out = stage / "rows.jsonl"
        n = 0
        with out.open("w", encoding="utf-8", newline="\n") as f:
            for row in record_stream(shard, rec):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        if n != N_RECORDS:
            raise ValueError("Incomplete validation split.")
        write_json(stage / "provenance.json", {"inputs":signature, "recordings":n, "output_sha256":sha(out)})
        try:
            stage.rename(dest)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not dest.is_dir():
                raise
            return cached_rows(dest, signature)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return dest / "rows.jsonl"
