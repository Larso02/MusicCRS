"""Neural reranking: score candidate tracks by similarity in learned embedding spaces.

The challenge data ships precomputed neural embeddings for every track (Hugging Face dataset
talkpl-ai/TalkPlayData-Challenge-Track-Embeddings). Two are used here:

  * metadata: a Qwen3-Embedding-0.6B vector (1024-d) of the track's metadata text. Text typed by the
    user (a request, a wish, liked styles) is embedded with the SAME model, so words and tracks live
    in one space.
  * cf: a BPR collaborative-filtering vector (128-d) trained on listening behaviour: tracks that the
    same listeners play are close together.

The main scorer is a TRAINED two-tower network (reranker_model.py, trained by
scripts/train_reranker.py on the TalkPlay training sessions) that takes both embeddings. The plain
cosine similarities in the raw embedding spaces are available as extra, zero-shot components.

A candidate is scored by similarity to anchors built from the session profile:

    + the songs on the playlist and the songs the user liked   (mean of their embeddings)
    + liked styles: the mean embedding of the tracks that carry a liked tag (a style prototype)
    + wishes that match no tag, as text                        (embedded by the encoder)
    - the songs and styles the user rejected                   (only above-average similarity is penalised)

Each component is z-scored over the candidate set so that the weights are comparable. There is no
BM25 and no tag counting in this stage; a tag only selects the tracks whose embeddings form the style.
"""

from pathlib import Path

import numpy as np

try:
    import reranker_model
except ImportError:                              # when imported as musiccrs.neural
    from musiccrs import reranker_model

ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_DATASET = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"
COLUMNS = {"meta": "metadata-qwen3_embedding_0.6b", "cf": "cf-bpr"}
CACHE_DIR = ROOT / "cache" / "emb"


def load_embeddings(track_ids: list[str], name: str, path: Path | None = None) -> np.ndarray:
    """Unit-length embeddings, one row per track id in the order of track_ids (zeros if missing).

    Built once from the Hugging Face dataset (a few hundred MB) and cached next to the other
    embedding caches.
    """
    path = path or CACHE_DIR / f"{name}.npy"
    if path.exists():
        matrix = np.load(path)
        if matrix.shape[0] == len(track_ids):
            return matrix

    import pyarrow as pa
    import pyarrow.compute as pc
    from datasets import load_dataset

    ds = load_dataset(EMBEDDING_DATASET, split="all_tracks")
    row = {t: i for i, t in enumerate(ds.data.column("track_id").to_pylist())}
    order = np.array([row[t] for t in track_ids])

    column = pa.concat_arrays(ds.data.column(COLUMNS[name]).chunks)
    lengths = pc.fill_null(pc.list_value_length(column), 0).to_numpy()  # pyright: ignore[reportAttributeAccessIssue]
    dim = int(lengths.max())
    flat = column.flatten().to_numpy(zero_copy_only=False)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    full = np.where(lengths == dim)[0]
    matrix = np.zeros((len(ds), dim), dtype=np.float32)               # tracks without an embedding stay zero
    matrix[full] = flat[offsets[full][:, None] + np.arange(dim)[None, :]]
    matrix = matrix[order]
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)
    return matrix


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


class NeuralScorer:
    """Scoring of candidates against the profile anchors in the embedding spaces."""

    def __init__(self, meta: np.ndarray, cf: np.ndarray | None = None, model=None):
        self.meta = meta
        self.cf = cf
        self.model = model                      # the trained two-tower network, or None
        self.tracks = None                      # its track tower applied to every track (N x d)
        if model is not None:
            import torch
            features = torch.tensor(np.concatenate([meta, cf if cf is not None else np.zeros((len(meta), 0), np.float32)], axis=1))
            with torch.no_grad():
                self.tracks = model.encode_tracks(features).numpy()

    @classmethod
    def load(cls, track_ids: list[str]) -> "NeuralScorer":
        meta = load_embeddings(track_ids, "meta")
        try:
            cf = load_embeddings(track_ids, "cf")
        except Exception:                      # the metadata space alone still works
            cf = None
        return cls(meta, cf, reranker_model.load())

    def similar(self, track: int, k: int = 5) -> list[int]:
        """The k nearest tracks in the metadata space (for inspection and tests)."""
        s = self.meta @ self.meta[track]
        s[track] = -1
        return [int(i) for i in np.argsort(-s)[:k]]

    def user_vector(self, positives: list[int]) -> np.ndarray:
        """The trained user tower applied to the songs the user chose (oldest first, last one counts extra)."""
        import torch
        history = positives[-reranker_model.MAX_HISTORY:]
        tracks = torch.tensor(self.tracks[history])[None]
        with torch.no_grad():
            return self.model.encode_user(tracks, torch.ones(1, len(history))).numpy()[0]

    def style_prototype(self, groups: list[np.ndarray], space: str = "meta") -> np.ndarray | None:
        """A style as the mean embedding of the tracks that carry its tag (one group of track indices per tag)."""
        matrix = self.tracks if space == "learned" and self.tracks is not None else self.meta
        protos = [unit(matrix[g[:3000]].mean(axis=0)) for g in groups if len(g)]
        return unit(np.mean(protos, axis=0)) if protos else None

    def components(self, cand: np.ndarray, positives: list[int], negatives: list[int],
                   text: str = "", encode=None, liked_styles: list[np.ndarray] | None = None,
                   rejected_styles: list[np.ndarray] | None = None, style_space: str = "meta") -> dict[str, np.ndarray]:
        """z-scored similarity components for the candidate rows (positive = good, negative = penalty).

        positives (oldest first) and negatives are track indices. text (a wish with no matching tag) is
        embedded with encode (str -> vector). liked_styles / rejected_styles hold, per tag, the indices of
        the tracks with that tag; a style is represented by their mean embedding. Components: learned (the
        trained network), tracks / cf (raw cosine to the mean of the positives), text, styles, and the
        penalties disliked_tracks / disliked_styles.
        """
        out: dict[str, np.ndarray] = {}
        rows = self.meta[cand]
        if positives:
            if self.model is not None and self.tracks is not None:
                out["learned"] = zscore(self.tracks[cand] @ self.user_vector(positives))
            out["tracks"] = zscore(rows @ unit(self.meta[positives].mean(axis=0)))
            if self.cf is not None:
                anchor = self.cf[positives].mean(axis=0)
                if np.linalg.norm(anchor) > 0:
                    out["cf"] = zscore(self.cf[cand] @ unit(anchor))
        vector = encode(text) if text and encode is not None else None
        if vector is not None:
            out["text"] = zscore(rows @ unit(np.asarray(vector, dtype=np.float32)))
        space_rows = self.tracks[cand] if style_space == "learned" and self.tracks is not None else rows
        liked = self.style_prototype(liked_styles or [], style_space)
        if liked is not None:
            out["styles"] = zscore(space_rows @ liked)
        if negatives:
            if self.tracks is not None:
                bad = unit(self.tracks[negatives].mean(axis=0))
                out["disliked_tracks"] = -np.maximum(zscore(self.tracks[cand] @ bad), 0)
            else:
                out["disliked_tracks"] = -np.maximum(zscore(rows @ unit(self.meta[negatives].mean(axis=0))), 0)
        rejected = self.style_prototype(rejected_styles or [], style_space)
        if rejected is not None:
            out["disliked_styles"] = -np.maximum(zscore(space_rows @ rejected), 0)
        return out
