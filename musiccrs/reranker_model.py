"""A trained neural reranker: a two-tower network over the challenge's track embeddings.

    track tower:  [Qwen3 metadata embedding (1024) | BPR collaborative-filtering embedding (128)] -> d
    user tower:   [mean of the tower outputs of the tracks the user chose | the last one]          -> d
    score(user, track) = cosine(user tower, track tower)

It is trained on the TalkPlay training sessions (scripts/train_reranker.py): given the tracks played
so far in a conversation, the track played next must score higher than random other tracks (sampled
softmax). Some sessions are held out for the evaluation (see is_heldout). At run time the profile's
liked songs and the playlist form the "history"; rejected songs are scored the same way and penalised.
"""

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "reranker.pt"
MAX_HISTORY = 8


def is_heldout(session_id: str, every: int = 10) -> bool:
    """Every 10th session is kept out of training and used for evaluation."""
    return int(hashlib.md5(str(session_id).encode()).hexdigest()[:8], 16) % every == 0


class TwoTower(nn.Module):
    def __init__(self, dim_in: int = 1152, dim: int = 256, hidden: int = 512):
        super().__init__()
        self.dim_in, self.dim, self.hidden = dim_in, dim, hidden
        self.track = nn.Sequential(nn.Linear(dim_in, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.user = nn.Sequential(nn.Linear(2 * dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.log_scale = nn.Parameter(torch.tensor(3.0))        # learned inverse temperature

    def encode_tracks(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.track(x), dim=-1)

    def encode_user(self, tracks: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """tracks: (B, L, d) encoded history, oldest first; mask: (B, L) 1 for real entries."""
        mask = mask.float()
        mean = (tracks * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        last_pos = (mask * torch.arange(mask.shape[1], device=mask.device)).argmax(1)
        last = tracks[torch.arange(tracks.shape[0]), last_pos]
        return F.normalize(self.user(torch.cat([mean, last], dim=-1)), dim=-1)

    def scale(self) -> torch.Tensor:
        return self.log_scale.exp().clamp(max=100)


def save(model: TwoTower, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "dim_in": model.dim_in, "dim": model.dim, "hidden": model.hidden}, path)


def load(path: Path = MODEL_PATH) -> TwoTower | None:
    """The trained model, or None if it has not been trained (python -m scripts.train_reranker)."""
    if not path.exists():
        return None
    blob = torch.load(path, map_location="cpu", weights_only=True)
    model = TwoTower(blob["dim_in"], blob["dim"], blob.get("hidden", 512))
    model.load_state_dict(blob["state"])
    return model.eval()
