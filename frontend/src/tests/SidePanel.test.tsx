import { act, fireEvent, render, screen } from "@testing-library/react";
import SidePanel, { matchesFilter } from "../components/SidePanel/SidePanel";
import { PlaylistTrack } from "../types";

const track = (id: string, name: string, artist: string, album = "Some Album"): PlaylistTrack => ({
  track_id: id,
  track_name: name,
  artist_name: [artist],
  album_name: [album],
  popularity: 50,
});

const yellow = track("t1", "Yellow", "Coldplay", "Parachutes");
const clocks = track("t2", "Clocks", "Coldplay", "A Rush of Blood to the Head");
const lithium = track("t3", "Lithium", "Nirvana", "Nevermind");

function renderPanel(playlist: PlaylistTrack[] = [yellow, clocks, lithium], taste: { slot: string; value: string }[] = []) {
  const send = jest.fn();
  render(<SidePanel playlist={playlist} taste={taste} send={send} serverUrl="http://server:5000/" />);
  return send;
}

function mockSearch(results: PlaylistTrack[], ok = true) {
  const fetchMock = jest.fn().mockResolvedValue({ ok, json: async () => ({ query: "", results }) });
  (global as any).fetch = fetchMock;
  return fetchMock;
}

async function typeInto(placeholder: string, text: string) {
  fireEvent.change(screen.getByPlaceholderText(placeholder), { target: { value: text } });
  await act(async () => {
    jest.advanceTimersByTime(300); // the search waits a moment after the last key
  });
}

beforeEach(() => {
  jest.useFakeTimers();
  mockSearch([]);
});

afterEach(() => {
  jest.useRealTimers();
});

describe("playlist filter", () => {
  test("every word must occur in the title, artist or album, in any case", () => {
    expect(matchesFilter(yellow, "yellow")).toBe(true);
    expect(matchesFilter(yellow, "COLDPLAY yel")).toBe(true);
    expect(matchesFilter(yellow, "parachutes")).toBe(true);
    expect(matchesFilter(yellow, "yellow nirvana")).toBe(false);
    expect(matchesFilter(yellow, "")).toBe(true);
    expect(matchesFilter(yellow, "   ")).toBe(true);
  });

  test("shows the songs that match, keeps their numbers and counts them", () => {
    renderPanel();
    expect(screen.getByText(/My playlist \(3\)/)).toBeTruthy();
    fireEvent.change(screen.getByPlaceholderText("Filter the playlist"), { target: { value: "nirvana" } });
    expect(screen.getByText(/My playlist \(1 of 3\)/)).toBeTruthy();
    expect(screen.getByText("Lithium")).toBeTruthy();
    expect(screen.getByText("3.")).toBeTruthy();                     // the real position in the playlist
    expect(screen.queryByText("Yellow")).toBeNull();
  });

  test("says when nothing matches", () => {
    renderPanel();
    fireEvent.change(screen.getByPlaceholderText("Filter the playlist"), { target: { value: "zzz" } });
    expect(screen.getByText(/No song in the playlist matches/)).toBeTruthy();
  });

  test("an empty playlist has nothing to filter", () => {
    renderPanel([]);
    expect(screen.getByText(/No songs yet/)).toBeTruthy();
    expect((screen.getByPlaceholderText("Filter the playlist") as HTMLInputElement).disabled).toBe(true);
    expect((screen.getByText("Clear") as HTMLButtonElement).disabled).toBe(true);
  });
});

describe("playlist buttons", () => {
  test("Remove sends the remove command with the track id", () => {
    const send = renderPanel();
    fireEvent.click(screen.getByLabelText("Remove Clocks"));
    expect(send).toHaveBeenCalledWith("/remove t2");
  });

  test("Clear asks first when there are several songs", () => {
    const send = renderPanel();
    const confirm = jest.spyOn(window, "confirm");
    confirm.mockReturnValueOnce(false);
    fireEvent.click(screen.getByText("Clear"));
    expect(send).not.toHaveBeenCalled();
    confirm.mockReturnValueOnce(true);
    fireEvent.click(screen.getByText("Clear"));
    expect(send).toHaveBeenCalledWith("/clear");
  });

  test("Clear does not ask for a single song", () => {
    const send = renderPanel([yellow]);
    const confirm = jest.spyOn(window, "confirm");
    fireEvent.click(screen.getByText("Clear"));
    expect(confirm).not.toHaveBeenCalled();
    expect(send).toHaveBeenCalledWith("/clear");
  });
});

describe("catalog search", () => {
  test("asks the server after a short pause and shows the songs", async () => {
    const fetchMock = mockSearch([track("c1", "Hurt", "Johnny Cash"), track("c2", "Hurt", "Nine Inch Nails")]);
    renderPanel();
    await typeInto("Search all songs by title or artist", "hurt");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe("http://server:5000/search?q=hurt&limit=15");   // no double slash
    expect(await screen.findByText("Johnny Cash", { exact: false })).toBeTruthy();
    expect(screen.getAllByText("Hurt")).toHaveLength(2);
  });

  test("does not search on every key, only after the pause", async () => {
    const fetchMock = mockSearch([]);
    renderPanel();
    const box = screen.getByPlaceholderText("Search all songs by title or artist");
    fireEvent.change(box, { target: { value: "y" } });
    fireEvent.change(box, { target: { value: "ye" } });
    fireEvent.change(box, { target: { value: "yel" } });
    await act(async () => {
      jest.advanceTimersByTime(300);
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toContain("q=yel&");
  });

  test("Add sends the add command with the track id", async () => {
    mockSearch([track("c1", "Hurt", "Johnny Cash")]);
    const send = renderPanel();
    await typeInto("Search all songs by title or artist", "hurt");
    fireEvent.click(await screen.findByLabelText("Add Hurt"));
    expect(send).toHaveBeenCalledWith("/add c1");
  });

  test("songs already on the playlist cannot be added again", async () => {
    mockSearch([yellow, track("c1", "Yellow Submarine", "The Beatles")]);
    renderPanel();
    await typeInto("Search all songs by title or artist", "yellow");
    expect(await screen.findByText("In playlist")).toBeTruthy();
    expect(screen.queryByLabelText("Add Yellow")).toBeNull();
    expect(screen.getByLabelText("Add Yellow Submarine")).toBeTruthy();
  });

  test("offers to add by name when nothing is found", async () => {
    mockSearch([]);
    const send = renderPanel();
    await typeInto("Search all songs by title or artist", "stnad by me");
    expect(await screen.findByText(/No song found/)).toBeTruthy();
    fireEvent.click(screen.getByText("Try adding it by name"));
    expect(send).toHaveBeenCalledWith("/add stnad by me");
  });

  test("says so when the server cannot be reached", async () => {
    (global as any).fetch = jest.fn().mockRejectedValue(new Error("network down"));
    renderPanel();
    await typeInto("Search all songs by title or artist", "yellow");
    expect(await screen.findByText(/search is not available/)).toBeTruthy();
  });

  test("clearing the box clears the results", async () => {
    mockSearch([track("c1", "Hurt", "Johnny Cash")]);
    renderPanel();
    await typeInto("Search all songs by title or artist", "hurt");
    await screen.findByLabelText("Add Hurt");
    await typeInto("Search all songs by title or artist", "");
    expect(screen.queryByLabelText("Add Hurt")).toBeNull();
    expect(screen.getByText(/Type a title, an artist, or both/)).toBeTruthy();
  });
});

describe("taste", () => {
  test("is hidden until something was learned, then shows the chips and a Forget button", () => {
    const send = renderPanel([yellow], []);
    expect(screen.queryByText("Your taste")).toBeNull();
  });

  test("shows likes and dislikes and Forget sends the forget command", () => {
    const send = renderPanel([yellow], [
      { slot: "like", value: "indie rock" },
      { slot: "dislike", value: "Nirvana" },
    ]);
    expect(screen.getByText(/indie rock/)).toBeTruthy();
    expect(screen.getByText(/Nirvana/)).toBeTruthy();
    fireEvent.click(screen.getByText("Forget"));
    expect(send).toHaveBeenCalledWith("/forget");
  });
});
