from __future__ import annotations as _annotations

import asyncio
import os
import sqlite3
from collections.abc import AsyncIterator
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, TypeVar

import httpx
import logfire
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import Field, TypeAdapter, BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from sqlalchemy.orm import Session
from typing_extensions import LiteralString, ParamSpec, TypedDict

from models import User, Portfolio, Orders, get_db

THIS_DIR = Path(__file__).parent
FRONTEND_BASE_URL = "http://localhost:5173"
agent = Agent("openai:gpt-4o")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://192.168.1.6:5173",
    ],  # Replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MessageTypeAdapter: TypeAdapter[ModelMessage] = TypeAdapter(
    Annotated[ModelMessage, Field(discriminator="kind")]
)
P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class Database:
    """Rudimentary database to store chat messages in SQLite.

    The SQLite standard library package is synchronous, so we
    use a thread pool executor to run queries asynchronously.
    """

    con: sqlite3.Connection
    _loop: asyncio.AbstractEventLoop
    _executor: ThreadPoolExecutor

    @classmethod
    @asynccontextmanager
    async def connect(
        cls, file: Path = THIS_DIR / ".chat_app_messages.sqlite"
    ) -> AsyncIterator[Database]:
        with logfire.span("connect to DB"):
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            con = await loop.run_in_executor(executor, cls._connect, file)
            slf = cls(con, loop, executor)
        try:
            yield slf
        finally:
            await slf._asyncify(con.close)

    @staticmethod
    def _connect(file: Path) -> sqlite3.Connection:
        con = sqlite3.connect(str(file))
        con = logfire.instrument_sqlite3(con)
        cur = con.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS messages (id INT PRIMARY KEY, message_list TEXT);"
        )
        con.commit()
        return con

    async def add_messages(self, messages: bytes):
        await self._asyncify(
            self._execute,
            "INSERT INTO messages (message_list) VALUES (?);",
            messages,
            commit=True,
        )
        await self._asyncify(self.con.commit)

    async def get_messages(self) -> list[ModelMessage]:
        c = await self._asyncify(
            self._execute, "SELECT message_list FROM messages order by id desc"
        )
        rows = await self._asyncify(c.fetchall)
        messages: list[ModelMessage] = []
        for row in rows:
            messages.extend(ModelMessagesTypeAdapter.validate_json(row[0]))
        return messages

    def _execute(
        self, sql: LiteralString, *args: Any, commit: bool = False
    ) -> sqlite3.Cursor:
        cur = self.con.cursor()
        cur.execute(sql, args)
        if commit:
            self.con.commit()
        return cur

    async def _asyncify(
        self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        return await self._loop.run_in_executor(  # type: ignore
            self._executor,
            partial(func, **kwargs),
            *args,  # type: ignore
        )


load_dotenv()

# Google OAuth2 setup
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = "http://localhost:8000/auth/google/callback"


@app.get("/")
async def root(request: Request, db: Session = Depends(get_db)):
    user_email = request.cookies.get("user_email")
    if not user_email or not db.query(User).filter(User.email == user_email).first():
        return RedirectResponse("/login")
    return RedirectResponse("/chat_app")


@app.get("/auth/google")
async def google_login():
    google_oauth_url = (
        f"https://accounts.google.com/o/oauth2/auth"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_type=code"
        f"&scope=openid email profile"
    )
    return JSONResponse(content={"redirect_url": google_oauth_url})


@app.get("/auth/google/callback")
async def google_callback(request: Request, db: Session = Depends(get_db)):
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Code not found")

    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    }

    async with httpx.AsyncClient() as client:
        token_response = await client.post(token_url, data=data)
        token_response.raise_for_status()
        tokens = token_response.json()

    async with httpx.AsyncClient() as client:
        user_info = await client.get(
            "https://www.googleapis.com/oauth2/v1/userinfo",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        user_info.raise_for_status()
        user = user_info.json()

    existing_user = db.query(User).filter(User.email == user["email"]).first()
    if not existing_user:
        new_user = User(email=user["email"], name=user.get("name", "Unknown"))
        db.add(new_user)
        db.commit()

    response = RedirectResponse(f"{FRONTEND_BASE_URL}/chat")
    response.set_cookie("user_email", user["email"])
    return response


class ChatMessage(TypedDict):
    """Format of messages sent to the browser."""

    role: Literal["user", "model"]
    timestamp: str
    content: str


def to_chat_message(m: ModelMessage) -> ChatMessage:
    first_part = m.parts[0]
    if isinstance(m, ModelRequest):
        if isinstance(first_part, UserPromptPart):
            return {
                "role": "user",
                "timestamp": first_part.timestamp.isoformat(),
                "content": first_part.content,
            }
    elif isinstance(m, ModelResponse):
        if isinstance(first_part, TextPart):
            return {
                "role": "model",
                "timestamp": m.timestamp.isoformat(),
                "content": first_part.content,
            }
    raise UnexpectedModelBehavior(f"Unexpected message type for chat app: {m}")


class PromptRequest(BaseModel):
    prompt: str


@app.post("/chat")
async def chat_prompt(request: PromptRequest):
    result = await agent.run(f'"{request.prompt}"')

    return {"response": f"Agent answer: {result.data}"}


@app.post("/order")
async def order(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    user_email = request.cookies.get("user_email")
    user = db.query(User).filter(User.email == user_email).first()
    if user:
        new_order = Orders(symbol=data["symbol"], quantity=data["quantity"], user_id=user.id)
        db.add(new_order)

        portfolio = (
            db.query(Portfolio)
            .filter(Portfolio.user_id == user.id, Portfolio.symbol == data["symbol"])
            .first()
        )
        if not portfolio:
            portfolio = Portfolio(symbol=data["symbol"], quantity=0, user_id=user.id)
            db.add(portfolio)
        portfolio.quantity += data["quantity"]
        db.commit()

        return {"status": "ok"}
    return {"status": "error", "message": "User not found"}


@app.get("/auth/status")
async def auth_status(request: Request):
    user = request.cookies.get("user_email")  # Or use session data
    return {"authenticated": user is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
