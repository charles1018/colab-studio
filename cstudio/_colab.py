"""Run pinned Colab CLI commands with a fixed OAuth grant and Session endpoint."""
from __future__ import annotations

import json
import os
import sys

from colab_cli import cli
from colab_cli.auth import PUBLIC_SCOPES
from colab_cli.client import Client, Prod
from colab_cli.state import SessionState, StateStore
from google.auth.transport.requests import AuthorizedSession
from google.oauth2.credentials import Credentials

from .colab_login import TOKEN, credential_scope


class SessionStore(StateStore):
    def __init__(self, name: str | None, endpoint: str | None, path: str):
        super().__init__(path)
        self.name = name
        self.endpoint = endpoint
        self.created_endpoint = None

    def _load_raw(self, stream):
        # The upstream store folds malformed data into an empty store.
        stream.seek(0)
        content = stream.read().strip()
        if not content:
            return {}
        return {name: SessionState.model_validate(value) for name, value in json.loads(content).items()}

    def get(self, name):
        session = super().get(name)
        if name != self.name or (session.endpoint if session else None) != self.endpoint:
            raise RuntimeError("Colab Session alias changed")
        return session

    def add(self, session):
        with self._lock_exclusive() as stream:
            sessions = self._load_raw(stream)
            previous = sessions.get(self.name)
            expected = self.endpoint or self.created_endpoint
            if (session.name != self.name or (previous.endpoint if previous else None) != expected
                    or (expected is not None and session.endpoint != expected)):
                raise RuntimeError("Colab Session alias changed")
            sessions[session.name] = session
            self._save_raw(stream, sessions)
            if self.endpoint is None:
                self.created_endpoint = session.endpoint

    def remove(self, name):
        with self._lock_exclusive() as stream:
            sessions = self._load_raw(stream)
            session = sessions.get(name)
            if name == self.name and session and session.endpoint == self.endpoint:
                del sessions[name]
                self._save_raw(stream, sessions)


def main():
    expected = os.environ["CSTUDIO_COLAB_ACCOUNT_SCOPE"]
    try:
        data = json.loads(TOKEN.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or credential_scope(data) != expected:
            raise ValueError
        credentials = Credentials.from_authorized_user_info(data, PUBLIC_SCOPES)
    except (OSError, ValueError, RecursionError):
        sys.exit("Colab credentials are unavailable or have changed")

    # A single credential snapshot backs the command, including token refreshes.
    cli.state._client = Client(Prod(), AuthorizedSession(credentials))
    cli.setup_logging = lambda _: None
    cli.auto_update.run_background_check = lambda: None
    command = sys.argv[1]
    store = None
    try:
        if command == "sessions":
            # Bypass sync_sessions' empty-store auth fallback and cross-account pruning.
            cli.state._sessions = SessionStore(None, None, cli.state.store.path).list()
        elif command != "usage":
            name = sys.argv[sys.argv.index("-s") + 1]
            store = SessionStore(name, os.environ.get("CSTUDIO_COLAB_ENDPOINT"), cli.state.store.path)
            if command != "new" and store.endpoint is None:
                sys.exit("Colab endpoint is required")
            store.get(name)
            cli.state._store = store
        code = cli.app(standalone_mode=False)
    except Exception as exc:
        sys.exit(f"Colab {command} failed ({type(exc).__name__})")
    if not code and store and store.created_endpoint:
        print("CSTUDIO_ASSIGNED " + store.created_endpoint, flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
