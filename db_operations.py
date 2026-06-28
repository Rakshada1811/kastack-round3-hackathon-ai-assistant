from database import supabase

def save_all(payload):

    if payload.get("topic_checkpoints"):
        supabase.table("topic_checkpoints").insert(
            payload["topic_checkpoints"]
        ).execute()

    if payload.get("message_checkpoints"):
        supabase.table("message_checkpoints").insert(
            payload["message_checkpoints"]
        ).execute()

    if payload.get("mood_checkpoints"):
        supabase.table("mood_checkpoints").insert(
            payload["mood_checkpoints"]
        ).execute()

    if payload.get("day_checkpoint_metadata"):
        supabase.table("day_checkpoint_metadata").insert(
            payload["day_checkpoint_metadata"]
        ).execute()

    if payload.get("events_memory"):
        supabase.table("events_memory").insert(
            payload["events_memory"]
        ).execute()