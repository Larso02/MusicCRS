import "./SidePanel.css";

import { useEffect, useMemo, useState } from "react";
import { PlaylistTrack, TasteChip } from "../../types";

type Props = {
  playlist: PlaylistTrack[];
  taste: TasteChip[];
  send: (message: string) => void;   // sends a chat command such as "/remove <id>"
  serverUrl: string;                 // where the /search route lives
};

type SearchState = "idle" | "loading" | "done" | "error";

const SEARCH_DELAY_MS = 250;

/** Every word of the filter must occur in the title, the artists or the album. */
export function matchesFilter(track: PlaylistTrack, filter: string): boolean {
  const words = filter.toLowerCase().split(/\s+/).filter(Boolean);
  const text = `${track.track_name} ${track.artist_name.join(" ")} ${track.album_name.join(" ")}`.toLowerCase();
  return words.every((word) => text.includes(word));
}

function SongLine({ track, index }: { track: PlaylistTrack; index?: number }) {
  return (
    <span className="song-text">
      <span className="song-title">
        {index !== undefined && <span className="song-number">{index}.</span>}
        {track.track_name}
      </span>
      <small className="song-artist">
        {track.artist_name.join(", ")}
        {track.album_name.length > 0 && <span className="song-album"> · {track.album_name[0]}</span>}
      </small>
    </span>
  );
}

export default function SidePanel({ playlist, taste, send, serverUrl }: Props) {
  const [filter, setFilter] = useState("");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<PlaylistTrack[]>([]);
  const [status, setStatus] = useState<SearchState>("idle");

  const onList = useMemo(() => new Set(playlist.map((t) => t.track_id)), [playlist]);
  const shown = useMemo(
    () => playlist.map((track, index) => ({ track, number: index + 1 })).filter(({ track }) => matchesFilter(track, filter)),
    [playlist, filter]
  );

  // search the whole catalog while typing: wait a moment, and drop answers to outdated queries
  useEffect(() => {
    const text = query.trim();
    if (!text) {
      setResults([]);
      setStatus("idle");
      return;
    }
    setStatus("loading");
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const base = serverUrl.replace(/\/+$/, "");
        const response = await fetch(`${base}/search?q=${encodeURIComponent(text)}&limit=15`, {
          signal: controller.signal,
        });
        if (!response.ok) throw new Error(`search failed: ${response.status}`);
        const data = await response.json();
        setResults(data.results as PlaylistTrack[]);
        setStatus("done");
      } catch (error) {
        if ((error as Error).name !== "AbortError") setStatus("error");
      }
    }, SEARCH_DELAY_MS);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [query, serverUrl]);

  const clearPlaylist = () => {
    if (playlist.length < 2 || window.confirm(`Remove all ${playlist.length} songs from the playlist?`)) {
      send("/clear");
    }
  };

  return (
    <aside className="side-panel" aria-label="Playlist and song search">
      {taste.length > 0 && (
        <section className="side-card taste-card" aria-label="What I have learned about your taste">
          <div className="side-card-header">
            <strong>Your taste</strong>
            <button type="button" className="btn btn-sm btn-outline-secondary" onClick={() => send("/forget")}>
              Forget
            </button>
          </div>
          <div className="taste-chips">
            {taste.map((chip, index) => (
              <span key={index} className={`taste-chip taste-${chip.slot}`}>
                {chip.slot === "dislike" ? "− " : "+ "}
                {chip.value}
              </span>
            ))}
          </div>
        </section>
      )}

      <section className="side-card playlist-card" aria-label="Current playlist">
        <div className="side-card-header">
          <strong>
            My playlist ({filter.trim() ? `${shown.length} of ${playlist.length}` : playlist.length})
          </strong>
          <button type="button" className="btn btn-sm btn-outline-danger" disabled={!playlist.length} onClick={clearPlaylist}>
            Clear
          </button>
        </div>
        <input
          type="search"
          className="form-control form-control-sm side-search"
          placeholder="Filter the playlist"
          aria-label="Filter the playlist"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          disabled={playlist.length === 0}
        />
        <div className="side-list">
          {playlist.length === 0 && <p className="side-empty">No songs yet. Search below, or ask in the chat.</p>}
          {playlist.length > 0 && shown.length === 0 && <p className="side-empty">No song in the playlist matches “{filter}”.</p>}
          {shown.map(({ track, number }) => (
            <div className="song-row" key={track.track_id}>
              <SongLine track={track} index={number} />
              <button
                type="button"
                className="btn btn-sm btn-outline-secondary"
                onClick={() => send(`/remove ${track.track_id}`)}
                aria-label={`Remove ${track.track_name}`}
              >
                Remove
              </button>
            </div>
          ))}
        </div>
      </section>

      <section className="side-card search-card" aria-label="Search all songs">
        <div className="side-card-header">
          <strong>Find songs</strong>
          {status === "loading" && <small className="text-muted">searching…</small>}
        </div>
        <input
          type="search"
          className="form-control form-control-sm side-search"
          placeholder="Search all songs by title or artist"
          aria-label="Search all songs by title or artist"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <div className="side-list">
          {status === "idle" && <p className="side-empty">Type a title, an artist, or both (for example “yellow coldplay”).</p>}
          {status === "error" && <p className="side-empty">The search is not available right now. Is the server running?</p>}
          {status === "done" && results.length === 0 && (
            <div className="side-empty">
              <p className="mb-1">No song found for “{query.trim()}”.</p>
              <button type="button" className="btn btn-sm btn-outline-primary" onClick={() => send(`/add ${query.trim()}`)}>
                Try adding it by name
              </button>
            </div>
          )}
          {results.map((track) => (
            <div className="song-row" key={track.track_id}>
              <SongLine track={track} />
              {onList.has(track.track_id) ? (
                <span className="in-playlist">In playlist</span>
              ) : (
                <button
                  type="button"
                  className="btn btn-sm btn-primary"
                  onClick={() => send(`/add ${track.track_id}`)}
                  aria-label={`Add ${track.track_name}`}
                >
                  Add
                </button>
              )}
            </div>
          ))}
        </div>
      </section>
    </aside>
  );
}
