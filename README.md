# DAT640-2026 MusicCRS starting package

This repository contains a starting package for the MusicCRS group project.

It has some basic functionality to provide static responses (type `/info`), display options to the user to choose from (`/option`), prompt a large language model (`/ask_llm [prompt]`) and terminate the conversation (`/quit`).

![Example](musiccrs_example.png)

## Backend (Conversational Agent)

The code for the conversational agent is found under the [musiccrs](musiccrs/) folder. It is based on the [DialogueKit](https://github.com/iai-group/DialogueKit) toolkit and assumes **Python 3.11**.

Install dependencies by running:

```bash
pip install -r requirements.txt
```

Then, start the server by running:

```bash
python musiccrs/musiccrs.py
```

The server will be running on <http://127.0.0.1:5000>.

Leave the terminal open to keep the server running. Open a new terminal window to set up and start the frontend.

## Frontend

The [frontend](frontend/) is based on [ChatWidget](https://github.com/iai-group/ChatWidget).

**You are allowed to make any changes and customizations to the frontend.**

Install Node if you don't have it from [here](https://nodejs.org/en/download/).

Then install all necessary packages:

```bash
frontend/$ npm install
```

This should generate `node_modules` folder.

To run the app in the development mode, use the following command:

```bash
npm start
```

Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

This should automatically launch the client in your browser, and the client should be able to communicate with the server.

The page will reload if you make edits.
You will also see any lint errors in the console.

## Dataset

All data is part of the [TalkPlay Data Challenge](https://huggingface.co/collections/talkpl-ai/talkplay-data-challenge)
collection on Hugging Face — the same challenge dataset used in the RecSys 2026 MusicCRS challenge.

| Dataset | Size | Description |
| --- | --- | --- |
| [`TalkPlayData-Challenge-Dataset`](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-Dataset) | 15,199 train sessions, 1,000 test sessions | Multi-turn (8-turn) music conversations with user profiles, conversation goals, and goal-progress assessments. Ground truth (gold track + response per turn) is embedded in each session's `conversations` field. |
| [`TalkPlayData-Challenge-Track-Metadata`](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-Track-Metadata) | 47,071 tracks | Track metadata: name, artist, album, tags, popularity, release date. |
| [`TalkPlayData-Challenge-User-Metadata`](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-User-Metadata) | 9,090 users | User demographics: age, gender, country. |
| [`TalkPlayData-Challenge-Track-Embeddings`](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-Track-Embeddings) | 47,071 tracks | Pre-computed track embeddings: audio (LAION-CLAP), cover image (SigLIP2), collaborative filtering (BPR), and text (Qwen3-Embedding-0.6B) over attributes, lyrics, and metadata. |
| [`TalkPlayData-Challenge-User-Embeddings`](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-User-Embeddings) | 9,090 users | Pre-computed collaborative filtering (BPR) user embeddings. |

See [retrieval/README.md](retrieval/README.md) for how these are loaded and used.

## Retrieval

The [retrieval](retrieval/) folder contains a text-to-item retrieval component over
the track metadata catalog above — a BM25 baseline, a data loader, and evaluation
utilities (nDCG, item/catalog diversity, and a combined final score) for scoring
predictions against the challenge dataset's ground truth. New retrieval methods
(e.g. dense, hybrid) plug into the same `RetrievalModule` interface and can be run
and evaluated the same way — see [retrieval/README.md](retrieval/README.md) for
the interface, setup, usage, and baseline results.

Minimal example of running and scoring a single query:

```python
from retrieval import BM25Retriever
from retrieval.evaluation import compute_ndcg_metrics

retriever = BM25Retriever()  # downloads + indexes the track catalog on first run
track_ids = retriever.text_to_item_retrieval("upbeat pop song", topk=5)

compute_ndcg_metrics(preds=track_ids, gold=["some_gold_track_id"], k_values=[1, 5])
```

For evaluating a full baseline against the challenge devset (ground truth extraction,
prediction generation, and scoring), see [retrieval/README.md](retrieval/README.md).

## Structuring responses

Use the `AnnotatedUtterance` class in Python to when sending responses from the agent.
Minimally, it contains the text of the message and specifies the sender:

```python
utterance = AnnotatedUtterance(
            "This is something the agent is saying",
            participant=DialogueParticipant.AGENT,
        )
```

Optionally, it can also provide a dialogue act with annotations. For example, there is a built-in dialogue act in DialogueKit for indicating that the chat is terminated:

```python
utterance = AnnotatedUtterance(
            "It was nice talking to you. Bye",
            dialogue_acts=[DialogueAct(intent=self.stop_intent)],
            participant=DialogueParticipant.AGENT,
        )
```

On the client side, annotated utterances are received as JSON dictionaries. E.g., the message with the stop intent is encoded as:

```json
{
  "sender": "MusicCRS",
  "message": {
    "text": "It was nice talking to you. Bye",
    "dialogue_acts": [
      {
        "intent": "EXIT",
        "annotations": []
      }
    ]
  }
}
```

You may introduce specific intents for certain functionality, which may require special handling on the frontend (or help the simulator interact automatically with your agent). In our example, we use a specific intent when presenting the user with a set of options to choose from. The specific options can also be marked up as annotations.

```python
utterance = AnnotatedUtterance(
                "Here are some options: [...]",
                participant=DialogueParticipant.AGENT,
                dialogue_acts=[
                    DialogueAct(
                        intent=_INTENT_OPTIONS,
                        annotations=[
                            SlotValueAnnotation("option", option) for option in options
                        ],
                    )
                ],
            )
```

Then, the specific options will also be passed on as annotations:

```json
{
  "sender": "MusicCRS",
  "message": {
    "text": "Here are some options: [...]",
    "dialogue_acts": [
      {
        "intent": "OPTIONS",
        "annotations": [
          {"slot": "option", "value": "Play some jazz music"},
          {"slot": "option", "value": "Recommend me some pop songs"},
          {"slot": "option", "value": "Create a workout playlist"}
        ]
      }
    ]
  }
}
```

Similarly, you may mark up certain entities in the text, such as artists or tracks as annotations, and display them differently (e.g., link them) in the chat.

```python
utterance = AnnotatedUtterance(
                "How about adding TRACK_NAME to the playlist?",
                participant=DialogueParticipant.AGENT,
                dialogue_acts=[
                    DialogueAct(
                        intent=_INTENT_RECOMMEND_TRACK,
                        annotations=[
                            SlotValueAnnotation(
                                "track",
                                value="https://api.spotify.com/v1/tracks/...",
                                start=17,
                                end=27,
                            )
                        ],
                ]
            )
```

## Using an LLM

You may utilize an LLM for implementing the desired functionality. You may make several LLM calls per conversation turn, but the agent's response must still be returned within a reasonable time (3-5 seconds).

All teams need to use the same LLM, [Gemma 4 31B](https://ai.google.dev/gemma), which is hosted on the Unix network at UiS on a dedicated GPU server and served through Open WebUI.

  - To get access, first register at <https://openwebui.ux.uis.no>, using your student number and password for login
  - Once approved, generate an API key by going to `Settings` -> `Account` and generate a key

You can test the model using `curl` to send a prompt:

```bash
curl https://openwebui.ux.uis.no/api/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gorina10.gemma-4-31b",
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'
```

The endpoint is OpenAI-compatible, so you can use the [openai](https://pypi.org/project/openai/) Python library (`pip install openai`) to access it programmatically:

```python
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY", base_url="https://openwebui.ux.uis.no/api")

response = client.chat.completions.create(
    model="gorina10.gemma-4-31b", messages=[{"role": "user", "content": "Hello"}]
)

print(response.choices[0].message.content)
```
