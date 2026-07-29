"""
Recording an interview, as a property of the room rather than a service this code runs.

**Why auto-egress on room creation, and not a start/stop call.** LiveKit will record a room
either way, but the two shapes fail differently. A manual `start_egress` after the participants
join has to be issued by something, retried if it fails, and stopped when the call ends -- and
every one of those steps is a chance to produce an interview with no recording and nobody
noticing until a reviewer asks for it. `RoomEgress` is set once, in the request that creates the
room, and the SFU starts and stops the recording itself around the room's own lifetime. So "the
session produces a reviewable artefact" becomes a room field rather than a state machine, which
is the claim `transport/livekit.py` makes about why WebRTC was worth the move.

**What that costs: the room must now be created explicitly.** LiveKit auto-creates a room when
the first participant joins, and an auto-created room carries no egress config. So `ensure_room`
runs before the agent connects. It is idempotent -- creating a room that exists returns the
existing one -- but the ordering is load-bearing and stated here because the failure is silent:
join first and the interview records nothing, with no error anywhere.

**What is honestly not verified.** Egress is a separate service from the SFU. It needs the
`livekit-egress` binary and a Redis instance, and neither is present in this development
environment -- `livekit-server --dev` runs the SFU alone. So the request below has never been
observed producing a file. Every part of it that can be checked without one has been: the
request is constructed and asserted against in tests, the room is created with the config
attached, and a missing egress service is reported as a stored reason rather than an exception.
The line between those two states is the whole point of `recording_status`, and the honest
reading of this module today is "wired, unverified end to end" -- see PROCESS.md rather than
assuming a recording exists.

Nothing here is on the conversation's critical path. A room is created once per session, before
the first turn.
"""

from __future__ import annotations

import os
from typing import Any

RECORDINGS_DIR = os.environ.get("AVATAR_RECORDINGS_DIR", "recordings")
"""
Where a local recording lands, relative to the egress service's own working directory.

Local files by default rather than S3, so the credential-free path stays credential-free. That
is also the configuration least likely to be right in production: a file on the egress
container's disk disappears when it restarts, and the fix is one of the `s3`/`gcp`/`azure`
fields on the same output object rather than a change here.
"""


def recording_enabled() -> bool:
    """
    Whether to ask for a recording at all.

    Off unless explicitly enabled. Recording an interview is a decision with consent and
    retention consequences, and it must not switch itself on because a deployment happened to
    configure an SFU -- the candidate's pre-join screen promises a recording, and that promise
    should follow the setting rather than the other way round.
    """
    return os.environ.get("AVATAR_RECORD", "").strip().lower() in ("1", "true", "yes", "on")


def egress_config(room: str) -> Any:
    """
    The `RoomEgress` for one interview: a composite of every participant, to one MP4.

    Room-composite rather than per-track, because a reviewer wants the interview, not four files
    to line up by hand. It captures both sides -- a recording with only the avatar in it is not
    a reviewable artefact of a conversation.

    The filename is derived from the room, which is derived from the session id, so a recording
    can be matched back to its record without a lookup table. `{time}` is expanded by the egress
    service, and it is included because a re-run of the same session must not silently overwrite
    the first recording.
    """
    from livekit import api
    from livekit.protocol import egress as proto

    return api.RoomEgress(
        room=api.RoomCompositeEgressRequest(
            room_name=room,
            # Both halves of the call. `audio_only` would be cheaper and is the wrong artefact.
            file_outputs=[
                proto.EncodedFileOutput(
                    # The enum rather than its integer value: mypy rejects the int, and it would
                    # be a silent mis-encode if the ordering ever changed upstream.
                    file_type=proto.EncodedFileType.MP4,
                    filepath=f"{RECORDINGS_DIR}/{room}-{{time}}.mp4",
                )
            ],
        )
    )


async def ensure_room(room: str) -> dict[str, Any]:
    """
    Create the room with recording attached, before anyone joins. Idempotent.

    Returns a status dict for the session record rather than raising, and that shape is the
    point: a deployment with no egress service is a normal state, not an outage, and it must be
    distinguishable from a deployment where recording was never asked for. Three outcomes, each
    stored with its reason:

      `off`          recording was not requested
      `unavailable`  requested, but the room could not be created with it -- reason attached
      `requested`    the room exists with an egress config, and the SFU owns it from here

    `requested` is deliberately not called `recording`. Nothing here observes a file being
    written; claiming a recording exists because a request succeeded is exactly the kind of
    unverified assertion this project's first rule prohibits.
    """
    if not recording_enabled():
        return {
            "status": "off",
            "reason": "AVATAR_RECORD is not set, so no recording was requested",
        }

    try:
        from livekit import api

        from avatar.transport.livekit import credentials

        url, key, secret = credentials()
    except Exception as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}

    client = api.LiveKitAPI(url, key, secret)
    try:
        from livekit.protocol import room as proto

        created = await client.room.create_room(
            proto.CreateRoomRequest(name=room, egress=egress_config(room))
        )
        return {
            "status": "requested",
            "room": room,
            "filepath": f"{RECORDINGS_DIR}/{room}-<time>.mp4",
            # `sid` proves the request reached the SFU rather than merely being constructed,
            # which is the strongest claim available with no egress service to observe.
            "room_sid": getattr(created, "sid", ""),
            "reason": (
                "the room was created with an egress config. Whether a file is produced "
                "depends on an egress service being registered with this SFU, which this "
                "code does not and cannot confirm from here."
            ),
        }
    except Exception as exc:
        # The expected failure when no egress service is registered, and it must not take the
        # interview down with it: a candidate should be interviewed even if the recording cannot
        # be set up. The reason is stored so the gap is visible on the record instead of being
        # discovered when someone asks for the video.
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        await client.aclose()
