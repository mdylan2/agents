import json
import logging
from pathlib import Path

import voices
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
    cli,
    inference,
)

logger = logging.getLogger("expressive-agent")

load_dotenv()

AGENT_NAME = "expressive_agent"
INSTRUCTIONS = (Path(__file__).parent / "prompt.md").read_text()

GREETING = (
    "Open the call the way you'd answer the phone to someone you know well. "
    "Short and warm, and leave them room to say what's going on."
)


class Friend(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS)

    async def on_enter(self) -> None:
        await self.session.generate_reply(instructions=GREETING)


server = AgentServer()


@server.rtc_session(agent_name=AGENT_NAME)
async def expressive_agent(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    meta = json.loads(ctx.job.metadata) if ctx.job.metadata else {}
    expressive = bool(meta.get("expressive", True))
    voice = voices.resolve(meta.get("tts"))
    logger.info("starting session", extra={"expressive": expressive, "voice": voice.label})

    session = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("google/gemma-4-31b-it"),
        tts=inference.TTS(voice.model, voice=voice.voice),
        turn_handling=TurnHandlingOptions(turn_detection=inference.TurnDetector()),
        expressive=expressive,
        preemptive_generation=True,
    )

    await session.start(agent=Friend(), room=ctx.room)
    await ctx.connect()

    # the frontend renders the active config, which only the agent knows once
    # the dispatch metadata has been resolved
    await ctx.room.local_participant.set_attributes(
        {
            "expressive": "true" if expressive else "false",
            "tts_provider": voice.provider,
            "tts_label": voice.label,
        }
    )


if __name__ == "__main__":
    cli.run_app(server)
