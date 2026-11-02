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
import { ChatMessage, PlaylistTrack, TasteChip } from "../../types";
import SidePanel from "../SidePanel/SidePanel";
import { ConfigContext } from "../../contexts/ConfigContext";


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
  const [taste, setTaste] = useState<TasteChip[]>([]);
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
      const playlistAct = message.dialogue_acts?.find((act) => act.intent === "PLAYLIST_STATE");
      if (playlistAct){
        try {
          setPlaylist(playlistAct.annotations.map((a) => JSON.parse(a.value)));
        } catch (error) {
          console.error("Invalid playlist state:", error);
        }
      }
      const tasteAct = message.dialogue_acts?.find((act) => act.intent === "PROFILE_STATE");
      if (tasteAct) {
        setTaste(tasteAct.annotations.map((a) => ({ slot: a.slot, value: a.value })));
      }
      if (!!message.text) {
        const image_url = message.attachments?.find((attachment) => attachment.type === "image")?.payload.images?.[0];
        updateMessages(
          <AgentChatMessage
            key={chatMessagesRef.current.length}
            feedback={config.useFeedback ? giveFeedback : null}
            message={message.text}
            image_url={image_url}
          />
        );
      }
    },
    [giveFeedback, chatMessagesRef, config]
  );

  const handleButtons = useCallback(
    (message: ChatMessage) => {
      const optionsAct = message.dialogue_acts?.find(
        (act) => act.intent === "OPTIONS"
      );
      if (optionsAct && optionsAct.annotations.length > 0) {
        setChatButtons(
          optionsAct.annotations.map((annotation, index) => {
            let title = annotation.value;
            let payload = annotation.value;
            try {
              const option = JSON.parse(annotation.value);
              title = option.title ?? title;
              payload = option.payload ?? payload;
            } catch (error) {
              // plain-text option (e.g. the /options demo): text is title and payload
            }
            return (
              <QuickReplyButton
                key={index}
                text={title}
                message={payload}
                click={handleQuickReply}
              />
            );
          })
        );
        return;
      }
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
      setTaste([]);
    });
  }, [onRestart]);

  return (
    <div className={`musiccrs-layout${config.useWidget ? " compact" : ""}`}>
      <div className="chat-widget-content chat-column">
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
      <SidePanel
        playlist={playlist}
        taste={taste}
        send={(message) => sendMessage({ message })}
        serverUrl={config.serverUrl ?? "http://127.0.0.1:5000"}
      />
    </div>
  );
}
