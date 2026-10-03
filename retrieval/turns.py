from datasets import load_dataset

DIALOGUE_DATASET = "talkpl-ai/TalkPlayData-Challenge-Dataset"
NUM_TURNS = 8

def load_turns(split: str) -> list[dict]:
    dataset = load_dataset(DIALOGUE_DATASET, split=split)
    records = []
    for item in dataset:
        by_turn = {}
        for m in item["conversations"]: # type: ignore
            by_turn.setdefault(int(m["turn_number"]), {})[m["role"]] = m["content"]
        for t in range(1, NUM_TURNS + 1):
            records.append({
                "session_id": item["session_id"], # type: ignore
                "user_id": item["user_id"], # type: ignore
                "turn_number": t,
                "user_message": by_turn[t]["user"],
                "history_user": [by_turn[k]["user"] for k in range(1, t)],
                "history_assistant": [by_turn[k]["assistant"] for k in range(1, t)],
                "history_tracks": [by_turn[k]["music"] for k in range(1, t)],
                "gold": by_turn[t]["music"],
            })
    return records