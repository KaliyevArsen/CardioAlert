"""Optional private-chat demonstration. No LLM impersonation or clinical advice."""

from dataclasses import dataclass, field
import os
from pathlib import Path
import threading
import time

from .predict import load_model, predict_record
from .schema import DISCLAIMER, FIELDS, validate_value


@dataclass
class Session:
    updated_at: float
    values: dict = field(default_factory=dict)


class Conversation:
    """Transport-independent state machine, also exercised by offline tests."""

    def __init__(self, artifact, ttl_seconds=900, clock=time.monotonic):
        self.artifact = artifact
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self.sessions: dict[int, Session] = {}

    def expire(self):
        now = self.clock()
        for key, session in list(self.sessions.items()):
            if now - session.updated_at >= self.ttl_seconds:
                del self.sessions[key]

    def message(self, key: int, text: str) -> str:
        self.expire()
        command = text.strip().lower()
        if command in {"/cancel", "/break"}:
            self.sessions.pop(key, None)
            return "Session cleared. Use /start to begin a fictional example."
        if command == "/privacy":
            return ("Use fictional data only. Inputs stay in server memory and are cleared after "
                    "completion, /cancel, or 15 minutes of inactivity. Telegram itself processes "
                    "messages under its own policies; this bot cannot delete Telegram's copies.")
        if command in {"/help", "/about"}:
            return DISCLAIMER + "\n/start: fictional example; /cancel: clear session; /privacy: data handling."
        if command == "/start":
            self.sessions[key] = Session(self.clock())
            return DISCLAIMER + "\nUse fictional values only. Enter '?' for an unknown value (up to three).\n" + FIELDS[0].prompt
        session = self.sessions.get(key)
        if session is None:
            return "No active session (it may have expired). Use /start."
        current = FIELDS[len(session.values)]
        try:
            value = None if text.strip() == "?" else float(text.strip())
            validate_value(current.name, value)
            if value is None and sum(v is None for v in session.values.values()) >= 3:
                raise ValueError("At most three unknown values are allowed")
        except ValueError as error:
            return f"Invalid input: {error}\n{current.prompt}"
        session.values[current.name] = value
        session.updated_at = self.clock()
        if len(session.values) < len(FIELDS):
            return FIELDS[len(session.values)].prompt
        values = self.sessions.pop(key).values  # Clear even when inference fails.
        result = predict_record(self.artifact, values)
        return (f"Research model score: {result['model_score']:.3f}\n"
                f"Dataset class at fixed threshold 0.5: {result['classification_at_threshold']}\n"
                "This score does not establish whether a person is healthy or ill.\n" + DISCLAIMER)


def run():
    try:
        import telebot
    except ImportError as error:
        raise RuntimeError('Install the optional bot dependency: pip install -e ".[bot]"') from error
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN in the environment")
    model_path = Path(os.environ.get("CARDIOALERT_MODEL", "artifacts/cleveland/model.joblib"))
    conversation = Conversation(load_model(model_path))
    bot = telebot.TeleBot(token, threaded=False)
    lock = threading.Lock()
    stop = threading.Event()

    def cleanup():
        while not stop.wait(30):
            with lock:
                conversation.expire()

    @bot.message_handler(content_types=["text"])
    def handle(message):
        if message.chat.type != "private":
            bot.reply_to(message, "This demonstration only accepts private chats with fictional data.")
            return
        try:
            with lock:
                answer = conversation.message(message.chat.id, message.text)
        except Exception:
            with lock:
                conversation.sessions.pop(message.chat.id, None)
            answer = "The demonstration could not complete. Session cleared. Use /start to retry."
        bot.send_message(message.chat.id, answer)

    thread = threading.Thread(target=cleanup, daemon=True)
    thread.start()
    print("Starting the educational Telegram demo. Do not submit personal health records.", flush=True)
    try:
        bot.infinity_polling(skip_pending=True, timeout=20, long_polling_timeout=20)
    finally:
        stop.set()
        with lock:
            conversation.sessions.clear()
