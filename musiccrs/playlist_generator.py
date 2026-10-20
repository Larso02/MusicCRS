
"""Automatic playlist generation for MusicCRS (R8)."""
import json
import logging
import math
from quopri import encode
import re
import sys
from pathlib import Path
import numpy as np
import torch
import pyarrow as pa
import pyarrow.compute as pc
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Allow imports from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.bm25 import BM25Retriever
from retrieval.data_loader import MusicCatalogLoader

log = logging.getLogger(__name__)
DEFAULT_MODEL = "gorina10.gemma-4-31b"
CORPUS_TYPES = ["track_name", "artist_name", "album_name", "tag_list"]

EMBEDDING_DATASET = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"
META_COLUMN = "metadata-qwen3_embedding_0.6b"
QUERY_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_CACHE = PROJECT_ROOT / "cache" / "emb" / "meta.npy"

POOL = 200
RRF_K = 60

def rrf(rankings: list[list[str]], k:int = RRF_K, weights: list[float] | None = None) -> list[str]:
    weights = weights or [1.0] * len(rankings)
    score: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, track_id in enumerate(ranking, start=1):
            score[track_id] = score.get(track_id, 0.0) + weight / (k + rank)

    return sorted(score, key=lambda t: score[t], reverse=True)

def load_meta_embeddings(track_ids: list[str], path: Path = EMBEDDING_CACHE) -> np.ndarray:
    if path.exists():
        matrix = np.load(path)
        if matrix.shape[0] == len(track_ids):
            return matrix

    ds = load_dataset(EMBEDDING_DATASET, split="all_tracks")
    row = {t: i for i, t in enumerate(ds.data.column("track_id").to_pylist())}
    order = np.array([row[t] for t in track_ids])

    column = pa.concat_arrays(ds.data.column(META_COLUMN).chunks)
    lengths = pc.fill_null(pc.list_value_length(column), 0).to_numpy()
    dim = int(lengths.max())
    flat = column.flatten().to_numpy(zero_copy_only=False)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    full = np.where(lengths == dim)[0]
    matrix = np.zeros((len(ds), dim), dtype=np.float32)
    matrix[full] = flat[offsets[full][:, None] + np.arange(dim)[None, :]]
    matrix = matrix[order]
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)
    return matrix

class DenseMetaIndex:
    def __init__(self, track_ids: list[str], matrix: np.ndarray | None = None, encoder=None):
        self.track_ids = track_ids
        self._matrix = matrix
        self._encoder = encoder
        self.failed = False

    def _load_encoder(self):
        model = SentenceTransformer(QUERY_MODEL, model_kwargs={"dtype": torch.float32})

        def encode(text: str) -> np.ndarray:
            return model.encode([text], prompt_name="query", normalize_embeddings=True, show_progress_bar=False)[0]

        return encode

    def warm_up(self) -> None:
        if self._matrix is None:
            self._matrix = load_meta_embeddings(self.track_ids)

        if self._encoder is None:
            self._encoder = self._load_encoder()

    def search(self, query: str, topk: int = POOL) -> list[str]:
        self.warm_up()

        assert self._matrix is not None and self._encoder is not None
        scores = self._matrix @ np.asarray(self._encoder(query), dtype=np.float32)
        top = np.argpartition(-scores, topk)[:topk]
        top = top[np.argsort(-scores[top])]
        return [self.track_ids[i] for i in top]

_SHARED: dict = {}

def get_shared() -> dict:
    if not _SHARED:
        retriever = BM25Retriever(corpus_types=CORPUS_TYPES)
        _SHARED["retriever"] = retriever
        _SHARED["metadata"] = MusicCatalogLoader().metadata_dict
        _SHARED["dense"] = DenseMetaIndex(retriever.track_ids)
    return _SHARED

def warm_up() -> None:
    shared = get_shared()
    try:
        shared["dense"].warm_up()
    except Exception:
        shared["dense"].failed = True
        log.warning("Dense retrieval unavailable; playlists use BM25 only.", exc_info=True)
class PlaylistGenerator:
    """Generate playlists from natural-language descriptions."""

    def __init__(self, llm, model: str = DEFAULT_MODEL):
        self.llm = llm
        self.model = model
        shared = get_shared()

        # Index music-related metadata, including descriptive tags.
        self.retriever = shared["retriever"]
        self.metadata = shared["metadata"]
        self.dense = shared["dense"]

    def interpret_request(self, description: str) -> dict:
        """Use the LLM to interpret the playlist request."""

        prompt = f"""
You are a music playlist planning assistant.

Analyze the user's request and return ONLY a JSON object.

Required JSON format:
{{
    "search_query": "descriptive music keywords",
    "duration_minutes": 60,
    "explanation": "brief reason for the duration"
}}

Rules:
- Search query should contain relevant genres, moods,
  instruments, or music styles.
- Estimate an appropriate playlist duration based on context.
- Do NOT ask the user for the number of songs.
- Short activities: approximately 15-30 minutes.
- General listening: approximately 40-60 minutes.
- Gym sessions: approximately 45-75 minutes.
- Studying: approximately 60-120 minutes.
- Road trips: approximately 90-180 minutes.
- Describe the MUSIC in the search query: genres, moods, tempo, instruments, and the themes
  the user names (for example sad, love, rain, summer). Do NOT put the place or activity itself
  in the query (gym, workout, studying, party, road trip, cooking, sleep): song and artist names
  containing those words are misleading. Translate it into music words instead, e.g.
  "music for the gym" -> "high energy hip hop electronic dance upbeat".
- Always keep artist or song names the user mentions.
- Return valid JSON only.

User request: {description}
"""

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )

        content = response.choices[0].message.content.strip()

        # Allow JSON enclosed in a Markdown code block.
        content = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            content,
        ).strip()

        return json.loads(content)

    @staticmethod
    def duration_ms(track: dict) -> int:
        """Extract track duration in milliseconds."""
        value = track.get("duration", 0)

        if isinstance(value, list):
            value = value[0] if value else 0

        try:
            return int(float(value))
        except (ValueError, TypeError):
            return 0

    def candidates(self, search_query: str) -> tuple[list[str], str]:
        rankings = [self.retriever.text_to_item_retrieval(search_query, topk=POOL)]
        if not self.dense.failed:
            try:
                rankings.append(self.dense.search(search_query, topk=POOL))
            except Exception:
                self.dense.failed = True
                log.warning("Dense retrieval failed; using BM25 only.", exc_info=True)

        method = "bm25+dense" if len(rankings) == 2 else "bm25"
        return rrf(rankings)[:POOL], method

    def generate(self, description: str) -> dict:
        """Generate a complete playlist based on a description."""

        if not description.strip():
            raise ValueError("Please provide a playlist description.")

        plan = self.interpret_request(description)

        search_query = str(
            plan.get("search_query", description)
        ).strip()

        try:
            target_minutes = float(
                plan.get("duration_minutes", 45)
            )
        except (ValueError, TypeError):
            target_minutes = 45

        if not math.isfinite(target_minutes):
            target_minutes = 45

        # Prevent excessively short or long playlists.
        target_minutes = max(15, min(target_minutes, 120))
        target_ms = target_minutes * 60 * 1000

        # Retrieve a reasonably large candidate pool.
        candidate_ids, method = self.candidates(search_query)

        selected = []
        selected_ids = set()
        artist_counts = {}
        total_ms = 0

        for track_id in candidate_ids:
            track = self.metadata.get(track_id)

            if not track or track_id in selected_ids:
                continue

            duration = self.duration_ms(track)

            # Exclude missing or implausible durations.
            if not 60_000 <= duration <= 900_000:
                continue

            artists = track.get("artist_name", [])
            artist = (
                str(artists[0]).casefold()
                if artists
                else "unknown"
            )

            # Allow at most 3 tracks by the same primary artist.
            if artist_counts.get(artist, 0) >= 3:
                continue

            selected.append(track_id)
            selected_ids.add(track_id)

            artist_counts[artist] = (
                artist_counts.get(artist, 0) + 1
            )

            total_ms += duration

            if total_ms >= target_ms or len(selected) >= 40:
                break

        return {
            "track_ids": selected,
            "description": description,
            "search_query": search_query,
            "retrieval": method,
            "target_minutes": target_minutes,
            "actual_minutes": round(total_ms / 60_000, 1),
            "explanation": plan.get("explanation", ""),
        }
