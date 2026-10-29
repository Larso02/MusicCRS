"""Part 4: the neural reranker (trained two-tower network + embedding similarities)."""

import numpy as np
import pytest
import torch

neural = pytest.importorskip("musiccrs.neural")
reranker_model = pytest.importorskip("musiccrs.reranker_model")
from musiccrs.track_database import TrackDatabase, artist_key  # noqa: E402
from musiccrs.user_profile import Profile  # noqa: E402

refine_module = pytest.importorskip("musiccrs.refine")
Config = refine_module.Config


# ---------- the building blocks (synthetic data, no downloads) ----------
def test_zscore_and_unit():
    z = neural.zscore(np.array([1.0, 2.0, 3.0]))
    assert abs(z.mean()) < 1e-6 and abs(z.std() - 1) < 1e-3
    assert np.allclose(neural.zscore(np.ones(4)), 0)                 # a constant column adds nothing
    assert abs(np.linalg.norm(neural.unit(np.array([3.0, 4.0]))) - 1) < 1e-6


def tiny_scorer(model=None, with_cf=True):
    rng = np.random.default_rng(0)
    meta = rng.normal(size=(60, 8)).astype(np.float32)
    meta /= np.linalg.norm(meta, axis=1, keepdims=True)
    cf = rng.normal(size=(60, 4)).astype(np.float32) if with_cf else None
    if cf is not None:
        cf /= np.linalg.norm(cf, axis=1, keepdims=True)
    return neural.NeuralScorer(meta, cf, model)


def test_raw_components_follow_the_anchors():
    s = tiny_scorer()
    cand = np.arange(10, 40)
    parts = s.components(cand, positives=[1, 2], negatives=[3])
    assert set(parts) == {"tracks", "cf", "disliked_tracks"}
    assert all(v.shape == (30,) for v in parts.values())
    assert (parts["disliked_tracks"] <= 0).all()                     # only penalties
    near = int(np.argmax(s.meta[cand] @ neural.unit(s.meta[[1, 2]].mean(axis=0))))
    assert int(np.argmax(parts["tracks"])) == near                   # the most similar candidate scores highest


def test_no_anchors_means_no_components():
    assert tiny_scorer().components(np.arange(5), [], []) == {}


def test_text_wishes_use_the_encoder_and_survive_a_missing_encoder():
    s = tiny_scorer()
    cand = np.arange(20)
    parts = s.components(cand, [], [], text="something calm", encode=lambda text: s.meta[7])
    assert int(np.argmax(parts["text"])) == 7
    assert "text" not in s.components(cand, [], [], text="something calm", encode=lambda text: None)
    assert "text" not in s.components(cand, [], [], text="something calm", encode=None)


def test_a_style_is_the_mean_embedding_of_the_tracks_with_the_tag():
    s = tiny_scorer()
    group = np.array([5, 6, 7, 8])
    proto = s.style_prototype([group])
    assert abs(np.linalg.norm(proto) - 1) < 1e-6
    assert np.allclose(proto, neural.unit(s.meta[group].mean(axis=0)), atol=1e-5)
    parts = s.components(np.arange(30), [], [], liked_styles=[group], rejected_styles=[np.array([20, 21])])
    assert set(parts) == {"styles", "disliked_styles"} and (parts["disliked_styles"] <= 0).all()
    assert s.style_prototype([]) is None and s.style_prototype([np.array([], dtype=int)]) is None


def tiny_model(dim_in=12):
    torch.manual_seed(0)
    return reranker_model.TwoTower(dim_in, dim=16, hidden=32).eval()


def test_the_two_tower_network_scores_by_cosine_and_ignores_padding():
    model = tiny_model()
    x = torch.randn(5, 12)
    tracks = model.encode_tracks(x)
    assert torch.allclose(tracks.norm(dim=-1), torch.ones(5), atol=1e-5)
    padded = torch.cat([tracks[:3], torch.zeros(2, 16)])[None]
    a = model.encode_user(padded, torch.tensor([[1, 1, 1, 0, 0]]))
    b = model.encode_user(tracks[:3][None], torch.ones(1, 3))
    assert torch.allclose(a, b, atol=1e-5)                           # masked entries change nothing


def test_the_network_is_used_when_present_and_saved_models_load(tmp_path):
    model = tiny_model(12)
    s = tiny_scorer(model)                                           # 8 meta + 4 cf = 12 features
    assert s.tracks is not None and s.tracks.shape == (60, 16)
    parts = s.components(np.arange(30), positives=[1, 2, 3], negatives=[4])
    assert "learned" in parts and (parts["disliked_tracks"] <= 0).all()
    path = tmp_path / "model.pt"
    reranker_model.save(model, path)
    again = reranker_model.load(path)
    x = torch.randn(3, 12)
    assert torch.allclose(model.encode_tracks(x), again.encode_tracks(x), atol=1e-6)
    assert reranker_model.load(tmp_path / "missing.pt") is None


def test_heldout_sessions_are_stable_and_about_a_tenth():
    ids = [f"session-{i}" for i in range(3000)]
    flags = [reranker_model.is_heldout(i) for i in ids]
    assert flags == [reranker_model.is_heldout(i) for i in ids]
    assert 0.07 < sum(flags) / len(flags) < 0.13


# ---------- the trained network and the real embeddings ----------
real = pytest.mark.skipif(
    not reranker_model.MODEL_PATH.exists() or not (neural.CACHE_DIR / "meta.npy").exists(),
    reason="train the network first: python -m scripts.train_reranker",
)


@pytest.fixture(scope="module")
def ix(agent_module):
    return agent_module.get_term_index(TrackDatabase())


@pytest.fixture(scope="module")
def scorer(agent_module, ix):
    scorer = agent_module.get_neural(ix)
    assert scorer is not None
    return scorer


def find(ix, title, artist):
    return next(i for i in range(ix.n) if ix.titles[i] == title and artist in ix.artist_names[i])


@real
def test_the_real_scorer_has_the_trained_network(scorer, ix):
    assert scorer.model is not None and scorer.tracks.shape == (ix.n, 256)
    assert scorer.meta.shape == (ix.n, 1024) and scorer.cf is not None and scorer.cf.shape == (ix.n, 128)


@real
def test_metadata_embeddings_put_similar_songs_together(scorer, ix):
    yellow = find(ix, "Yellow", "Coldplay")
    near = scorer.similar(yellow, 10)
    assert sum(bool(ix.desc_tags[i] & ix.desc_tags[yellow]) for i in near) >= 7     # same styles
    far = [int(i) for i in np.argsort(scorer.meta @ scorer.meta[yellow])[:10]]
    assert sum(bool(ix.desc_tags[i] & ix.desc_tags[yellow]) for i in far) < sum(bool(ix.desc_tags[i] & ix.desc_tags[yellow]) for i in near)


@real
def test_the_trained_network_continues_a_listening_history(scorer, ix):
    history = [find(ix, "Lithium", "Nirvana"), find(ix, "Come As You Are", "Nirvana")]
    scores = scorer.tracks @ scorer.user_vector(history)
    scores[history] = -9
    top = [int(i) for i in np.argsort(-scores)[:20]]
    assert sum("grunge" in ix.desc_tags[i] for i in top) >= 10       # it learned what comes next


@real
def test_a_liked_style_pulls_its_tracks_up(scorer, ix):
    candidates = np.random.default_rng(1).choice(ix.n, 2000, replace=False)
    parts = scorer.components(candidates, [], [], liked_styles=[ix.tag_docs["grunge"]], style_space="learned")
    top = candidates[np.argsort(-parts["styles"])[:30]]
    base = np.mean(["grunge" in ix.desc_tags[i] for i in candidates])
    assert np.mean(["grunge" in ix.desc_tags[i] for i in top]) > 5 * base


@real
def test_neural_reranking_changes_the_order_but_not_the_rules(scorer, ix):
    played = [find(ix, "Lithium", "Nirvana")]
    plain = refine_module.refine(ix, "rock", played, Profile(), Config(neural=False), k=20)
    ranked = refine_module.refine(ix, "rock", played, Profile(), Config(), k=20, neural=scorer)
    assert [i.idx for i in ranked] != [i.idx for i in plain]
    assert played[0] not in {i.idx for i in ranked}
    assert all("learned" in i.parts for i in ranked)                 # the trained network scored them
    assert "bm25" not in ranked[0].parts                             # no BM25 in the reranking stage
    assert not any(k.startswith("tag") for k in ranked[0].parts)     # and no tag-overlap scoring


@real
def test_rejections_hold_with_the_neural_reranker(scorer, ix):
    played = [find(ix, "Lithium", "Nirvana")]
    p = Profile(); p.dislike_artist(artist_key("Nirvana")); p.dislike_tag("punk")
    items = refine_module.refine(ix, "grunge", played, p, Config(), k=30, neural=scorer)
    assert not any("Nirvana" in ix.artist_names[i.idx] for i in items)
    assert not any("punk" in ix.desc_tags[i.idx] for i in items)


@real
def test_a_disliked_song_is_pushed_away(scorer, ix):
    played = [find(ix, "Yellow", "Coldplay")]
    base = refine_module.refine(ix, "", played, Profile(), Config(), k=10, neural=scorer)
    p = Profile(); p.dislike_track(ix.ids[base[0].idx])
    after = refine_module.refine(ix, "", played, p, Config(), k=10, neural=scorer)
    assert base[0].idx not in {i.idx for i in after}


@real
def test_liked_style_in_chat_conditions_raises_its_share(scorer, ix):
    played = [find(ix, "Yellow", "Coldplay")]
    p = Profile(); p.like_track(ix.ids[played[0]])
    plain = refine_module.refine(ix, "", played, p, Config(), k=10, neural=scorer)
    p.like_tag("grunge")
    styled = refine_module.refine(ix, "", played, p, Config(), k=10, neural=scorer)
    share = lambda items: np.mean(["grunge" in ix.desc_tags[i.idx] for i in items[:5]])
    assert share(styled) > share(plain)


@real
def test_mmr_uses_the_neural_space(scorer, ix):
    played = [find(ix, "Lithium", "Nirvana")]
    plain = refine_module.refine(ix, "rock", played, Profile(), Config(), k=5, neural=scorer)
    varied = refine_module.refine(ix, "rock", played, Profile(), Config(mmr_lambda=0.3), k=5, neural=scorer)
    similarity = lambda items: (lambda m: (m @ m.T).sum() - len(items))(scorer.tracks[[i.idx for i in items]])
    assert similarity(varied) < similarity(plain)


@real
def test_explanations_name_the_neural_reasons(scorer, ix):
    played = [find(ix, "Lithium", "Nirvana")]
    items = refine_module.refine(ix, "rock", played, Profile(), Config(), k=20, neural=scorer)
    texts = [refine_module.explain(ix, i) for i in items]
    assert all(texts) and any("fits the songs you chose" in t for t in texts)


@real
def test_the_agent_uses_the_neural_reranker_after_feedback(agent, say, agent_module):
    say(agent, "add Smells Like Teen Spirit by Nirvana")
    agent._profile.dislike_artist(artist_key("Nirvana"))
    agent._profile.like_tag("indie rock")
    text = agent._refined_recommendation("Based on what you told me:")
    assert "Here is the new list" in text
    assert not any("Nirvana" in t["artist_name"] for t in agent._last_recs)
    assert agent_module.get_neural(agent_module.get_term_index(TrackDatabase())) is not None
