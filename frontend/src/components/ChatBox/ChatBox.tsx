import "./ChatBox.css";

import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  useContext,
} from "react";
import QuickReplyButton from "../QuickReply";
import { useSocket } from "../../contexts/SocketContext";
import { UserContext } from "../../contexts/UserContext";
import {
  MDBCard,
  MDBCardHeader,
  MDBCardBody,
  MDBIcon,
  MDBCardFooter,
} from "mdb-react-ui-kit";
import { AgentChatMessage, UserChatMessage } from "../ChatMessage";
import { ChatMessage } from "../../types";
import { ConfigContext } from "../../contexts/ConfigContext";


interface PlaylistTrack {
  track_id: string;
  track_name: string;
  artist_name: string[];
  album_name: string[];
  popularity: number;
}


export default function ChatBox() {
  const { config } = useContext(ConfigContext);
  const { user } = useContext(UserContext);
  const {
    startConversation,
    sendMessage,
    quickReply,
    onMessage,
    onRestart,
    giveFeedback,
  } = useSocket();
  const [chatMessages, setChatMessages] = useState<JSX.Element[]>([]);
  const [chatButtons, setChatButtons] = useState<JSX.Element[]>([]);
  const [inputValue, setInputValue] = useState<string>("");
  const [playlist, setPlaylist] = useState<PlaylistTrack[]>([]);
  const [newTrack, setNewTrack] = useState("");
  const chatMessagesRef = useRef(chatMessages);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    startConversation();
  }, [startConversation]);

  const updateMessages = (message: JSX.Element) => {
    chatMessagesRef.current = [...chatMessagesRef.current, message];
    setChatMessages(chatMessagesRef.current);
  };

  const handleInput = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (inputValue.trim() === "") return;
    updateMessages(
      <UserChatMessage
        key={chatMessagesRef.current.length}
        message={inputValue}
      />
    );
    sendMessage({ message: inputValue });
    setInputValue("");
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  const handleQuickReply = useCallback(
    (message: string) => {
      updateMessages(
        <UserChatMessage
          key={chatMessagesRef.current.length}
          message={message}
        />
      );
      quickReply({ message: message });
    },
    [chatMessagesRef, quickReply]
  );

  const handelMessage = useCallback(
    (message: ChatMessage) => {
      if (!!message.text) {
        let displayText = message.text;
        if (displayText.startsWith("__PLAYLIST_STATE__")) {
          try {
            const state = JSON.parse(displayText.slice("__PLAYLIST_STATE__".length));
            if (Array.isArray(state.tracks)) setPlaylist(state.tracks);
            displayText = state.message || "Playlist updated.";
          } catch (error) {
            console.error("Invalid playlist state:", error);
          }
        }
        const image_url = message.attachments?.find(
          (attachment) => attachment.type === "images"
        )?.payload.images?.[0];
        updateMessages(
          <AgentChatMessage
            key={chatMessagesRef.current.length}
            feedback={config.useFeedback ? giveFeedback : null}
            message={displayText}
            image_url={image_url}
          />
        );
      }
    },
    [giveFeedback, chatMessagesRef, config]
  );

  const handleButtons = useCallback(
    (message: ChatMessage) => {
      const buttons = message.attachments?.find(
        (attachment) => attachment.type === "buttons"
      )?.payload.buttons;
      if (!!buttons && buttons.length > 0) {
        setChatButtons(
          buttons.map((button, index) => {
            return (
              <QuickReplyButton
                key={index}
                text={button.title}
                message={button.payload}
                click={handleQuickReply}
              />
            );
          })
        );
      } else {
        setChatButtons([]);
      }
    },
    [handleQuickReply]
  );

  useEffect(() => {
    onMessage((message: ChatMessage) => {
      handelMessage(message);
      handleButtons(message);
    });
  }, [onMessage, handleButtons, handelMessage]);

  useEffect(() => {
    onRestart(() => {
      setChatMessages([]);
      setChatButtons([]);
      setPlaylist([]);
    });
  }, [onRestart]);

  return (
    <div className="chat-widget-content">
      <MDBCard
        id="chatBox"
        className="chat-widget-card"
        style={{ borderRadius: "15px" }}
      >
        <MDBCardHeader
          className="d-flex justify-content-between align-items-center p-3 bg-info text-white border-bottom-0"
          style={{
            borderTopLeftRadius: "15px",
            borderTopRightRadius: "15px",
          }}
        >
          <p className="mb-0 fw-bold">{config.name}</p>
          <p className="mb-0 fw-bold">{user?.username}</p>
        </MDBCardHeader>

        <MDBCardBody>
          <div className="card-body-messages">
            {chatMessages}
            <div className="d-flex flex-wrap justify-content-between">
              {chatButtons}
            </div>
          </div>
        </MDBCardBody>
        <section className="playlist-panel" aria-label="Current playlist">
          <div className="playlist-header">
            <strong>My Playlist ({playlist.length})</strong>
            <button type="button" className="btn btn-sm btn-outline-danger" disabled={!playlist.length}
              onClick={() => sendMessage({ message: "/clear" })}>Clear</button>
          </div>
          <div className="playlist-items">
            {playlist.length === 0 ? (
              <p className="text-muted mb-1">No tracks yet.</p>
            ) : (
              playlist.map((track, index) => (
                <div className="playlist-item" key={track.track_id}>
                  <span>
                    {index + 1}. {track.track_name}
                    <small className="text-muted">
                      {" — "}{track.artist_name.join(", ")}
                    </small>
                  </span>

                  <button
                    type="button"
                    className="btn btn-sm btn-outline-secondary"
                    onClick={() =>
                      sendMessage({ message: `/remove ${track.track_id}` })
                    }
                    aria-label={`Remove ${track.track_name}`}
                  >
                    Remove
                  </button>
                </div>
              ))
        )}

          </div>
          <form className="playlist-add" onSubmit={(e) => {
            e.preventDefault();
            if (newTrack.trim()) sendMessage({ message: `/add ${newTrack.trim()}` });
            setNewTrack("");
          }}>
            <input className="form-control form-control-sm" value={newTrack}
              onChange={(e) => setNewTrack(e.target.value)} placeholder="Song title or title by artist" />
            <button className="btn btn-sm btn-primary" type="submit">Add</button>
          </form>
        </section>
        <MDBCardFooter className="text-muted d-flex justify-content-start align-items-center p-2">
          <form className="d-flex flex-grow-1" onSubmit={handleInput}>
            <input
              type="text"
              className="form-control form-control-lg"
              id="ChatInput"
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Type message"
              ref={inputRef}
            ></input>
            <button type="submit" className="btn btn-link text-muted">
              <MDBIcon fas size="2x" icon="paper-plane" />
            </button>
          </form>
        </MDBCardFooter>
      </MDBCard>
    </div>
  );
}
