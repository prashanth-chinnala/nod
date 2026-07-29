"""
Voice for the assistant: speech in, speech out, one request each.

**Why not the interview's adapters.** `avatar.audio.stt` holds a WebSocket open to Deepgram's
streaming endpoint and `tts_deepgram` yields PCM in chunks, because an interview needs partial
transcripts and audio it can start playing before generation finishes. Neither property is worth
anything here: a question is spoken once and complete before it is sent, and an answer is short.
So this uses Deepgram's *prerecorded* listen endpoint and asks speak for one finished file — the
same vendor and the same credential, in the shape that matches the workload.

The point of reusing the vendor rather than the browser's `SpeechRecognition` is narrower than
convenience. Chrome's implementation ships audio to Google, which is a third processor nobody
consented to and a different privacy story from the rest of this product; and it does not exist
in Firefox or Safari, so a "voice assistant" would silently be a Chrome feature.

**Why the assistant's voice differs from the interviewer's by default.** They speak to the same
person about the same interviews, and if they sound identical then a recruiter listening to a
summary of a session hears the same voice that conducted it. Different voice, different speaker
— configurable, but not the same by accident.

**What is deliberately absent: turn detection.** No VAD, no end-of-turn policy, none of the
machinery `avatar.audio.turn_detection` exists for. The user presses a button to start and stop,
and that is the correct answer for a screen where they are also reading and typing — guessing
when someone has finished dictating into a text box produces a transcript that submits itself
mid sentence.
"""

from __future__ import annotations

import os

LISTEN_URL = "https://api.deepgram.com/v1/listen"
"""
The prerecorded endpoint, not the `wss://` one `avatar.audio.stt` uses.

Same path, different scheme, and the difference is the whole reason this module exists:
streaming returns interim results that get revised, which an interview needs and a dictated
question does not.
"""

SPEAK_URL = "https://api.deepgram.com/v1/speak"

DEFAULT_MODEL = "nova-3"

DEFAULT_VOICE = "aura-2-orion-en"
"""
The assistant's voice, deliberately not the interviewer's `aura-2-thalia-en` default.

Overridable with `AVATAR_ASSISTANT_VOICE`. Different by default so a summary of an interview
does not arrive in the voice that conducted it.
"""

TIMEOUT_SECONDS = 30.0
"""
Generous, because neither call is on anyone's critical path.

A dictated question is a second or two of audio and an answer is a few sentences; if either
takes longer than this, something is wrong in a way a retry will not fix.
"""

MAX_AUDIO_BYTES = 8 * 1024 * 1024
"""
Ceiling on an upload. About ten minutes of Opus, far past any dictated question.

Present because this endpoint takes a raw body from a browser and forwards it to a paid API:
without a bound, one page with a stuck recorder is an unbounded bill.
"""

MAX_SPEAK_CHARS = 2000
"""
Ceiling on synthesis. Long answers are for reading, not listening.

Truncated rather than refused -- hearing the first part of an answer is more useful than an
error, and the full text is on screen either way.
"""


class VoiceUnavailable(RuntimeError):
    """
    No Deepgram credential. Its own type so the caller can answer 404 rather than 500.

    Voice is optional: the assistant works entirely by typing, and a deployment with no speech
    credential should present a console with no microphone button rather than one that fails
    when pressed.
    """


def available() -> bool:
    return bool(os.environ.get("DEEPGRAM_API_KEY"))


def _key() -> str:
    key = os.environ.get("DEEPGRAM_API_KEY", "")
    if not key:
        raise VoiceUnavailable(
            "voice needs DEEPGRAM_API_KEY. Without it the assistant is text-only, which is a "
            "supported configuration -- the console hides the microphone rather than failing."
        )
    return key


def voice_name() -> str:
    return os.environ.get("AVATAR_ASSISTANT_VOICE", DEFAULT_VOICE)


async def transcribe(audio: bytes, content_type: str) -> str:
    """
    One audio blob to text.

    `content_type` is forwarded from the browser rather than assumed. MediaRecorder produces
    whatever the platform supports -- webm/opus on Chrome, mp4 on Safari -- and Deepgram sniffs
    the container, so passing the recorder's own type through is what makes this work on both.
    Guessing a format here would produce an empty transcript on one browser and no error on
    either.

    Returns the transcript, or an empty string when nothing was said. Empty is a legitimate
    outcome -- a user who pressed record and then changed their mind -- and must not raise.
    """
    if not audio:
        return ""
    if len(audio) > MAX_AUDIO_BYTES:
        raise ValueError(
            f"audio is {len(audio)} bytes, over the {MAX_AUDIO_BYTES} ceiling; a dictated "
            "question is seconds long, so this is a stuck recorder rather than a long one"
        )

    import httpx

    params = {"model": DEFAULT_MODEL, "smart_format": "true", "punctuate": "true"}
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        response = await client.post(
            LISTEN_URL,
            params=params,
            headers={"Authorization": f"Token {_key()}", "Content-Type": content_type},
            content=audio,
        )
        response.raise_for_status()
        body = response.json()

    # Deepgram nests the transcript four levels deep and returns the structure with empty lists
    # rather than omitting them when nothing was heard, so this walks it defensively rather than
    # indexing -- an IndexError here would present as a broken microphone.
    channels = (body.get("results") or {}).get("channels") or []
    if not channels:
        return ""
    alternatives = channels[0].get("alternatives") or []
    if not alternatives:
        return ""
    return str(alternatives[0].get("transcript") or "").strip()


async def speak(text: str) -> tuple[bytes, str]:
    """
    Text to one finished audio file. Returns the bytes and their content type.

    MP3 rather than raw PCM: the browser plays it from a Blob URL with no Web Audio scheduling,
    no jitter buffer, and no sample-rate agreement. All of that machinery exists on the
    interview path because audio arrives there while it is still being generated; here the file
    is complete before it is sent, and reaching for the same apparatus would be copying the
    shape of a solution without its problem.
    """
    trimmed = text.strip()[:MAX_SPEAK_CHARS]
    if not trimmed:
        return b"", ""

    import httpx

    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        response = await client.post(
            SPEAK_URL,
            params={"model": voice_name(), "encoding": "mp3"},
            headers={"Authorization": f"Token {_key()}", "Content-Type": "application/json"},
            json={"text": trimmed},
        )
        response.raise_for_status()
        return response.content, response.headers.get("content-type", "audio/mpeg")
