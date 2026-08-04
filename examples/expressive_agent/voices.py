"""The expressive-capable TTS voices this demo can switch between.

Expressive mode needs a LiveKit Inference TTS whose model declares a markup
dialect. Providers without one (deepgram, rime) synthesize fine but have no
tags to render, so they are deliberately absent here.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Voice:
    provider: str
    model: str
    voice: str
    label: str


VOICES: dict[str, Voice] = {
    "fishaudio": Voice(
        provider="fishaudio",
        model="fishaudio/s2.1-pro",
        voice="9a9cf47702da476aa4629e2506d4a857",
        label="Fish Audio S2.1 Pro (Hannah)",
    ),
    "inworld": Voice(
        provider="inworld",
        model="inworld/inworld-tts-2",
        voice="Ashley",
        label="Inworld TTS 2 (Ashley)",
    ),
    "cartesia": Voice(
        provider="cartesia",
        model="cartesia/sonic-3",
        voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        label="Cartesia Sonic 3 (Jacqueline)",
    ),
    "xai": Voice(
        provider="xai",
        model="xai/tts-1",
        voice="eve",
        label="xAI TTS 1 (Eve)",
    ),
}

DEFAULT_VOICE = "fishaudio"


def resolve(provider: str | None) -> Voice:
    """Pick a voice by provider name, falling back to the default."""
    return VOICES.get(provider or "", VOICES[DEFAULT_VOICE])
