"""Part 4: the session preference profile built from the user's feedback.

The profile holds what the user liked and rejected during this conversation: tracks, artists and
tags. It is updated by every feedback sentence (explicit) and by playlist actions (implicit: adding
a recommended track counts as liking it, removing one as rejecting it). The newest opinion wins:
liking something removes it from the rejected set and the other way round.
"""

from dataclasses import dataclass, field


@dataclass
class Profile:
    liked_tracks: list[str] = field(default_factory=list)        # track ids, oldest first
    disliked_tracks: list[str] = field(default_factory=list)
    liked_artists: set[str] = field(default_factory=set)         # artist keys (see track_database.artist_key)
    disliked_artists: set[str] = field(default_factory=set)
    liked_tags: set[str] = field(default_factory=set)            # descriptive tags (see tag_labels.json)
    disliked_tags: set[str] = field(default_factory=set)
    wishes: list[str] = field(default_factory=list)              # free text that matched no tag

    def is_empty(self) -> bool:
        return not (self.liked_tracks or self.disliked_tracks or self.liked_artists or self.disliked_artists
                    or self.liked_tags or self.disliked_tags or self.wishes)

    def like_track(self, track_id: str) -> None:
        _move(track_id, self.liked_tracks, self.disliked_tracks)

    def dislike_track(self, track_id: str) -> None:
        _move(track_id, self.disliked_tracks, self.liked_tracks)

    def like_artist(self, key: str) -> None:
        self.liked_artists.add(key)
        self.disliked_artists.discard(key)

    def dislike_artist(self, key: str) -> None:
        self.disliked_artists.add(key)
        self.liked_artists.discard(key)

    def like_tag(self, tag: str) -> None:
        self.liked_tags.add(tag)
        self.disliked_tags.discard(tag)

    def dislike_tag(self, tag: str) -> None:
        self.disliked_tags.add(tag)
        self.liked_tags.discard(tag)

    def add_wish(self, text: str) -> None:
        text = text.strip()
        if text and text not in self.wishes:
            self.wishes.append(text)

    def clear(self) -> None:
        self.__init__()                                          # type: ignore[misc]

    def summary(self) -> dict:
        """A plain dict for logs, the explanation and the LLM prompt."""
        return {
            "liked_tracks": len(self.liked_tracks), "disliked_tracks": len(self.disliked_tracks),
            "liked_artists": sorted(self.liked_artists), "disliked_artists": sorted(self.disliked_artists),
            "liked_tags": sorted(self.liked_tags), "disliked_tags": sorted(self.disliked_tags),
            "wishes": list(self.wishes),
        }


def _move(track_id: str, into: list[str], out_of: list[str]) -> None:
    if track_id in out_of:
        out_of.remove(track_id)
    if track_id not in into:
        into.append(track_id)
