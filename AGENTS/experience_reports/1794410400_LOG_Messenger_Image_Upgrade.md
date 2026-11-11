# Messenger image opt-in REST boundary

## Prompt History

User: "add image sending as an opt in upgrade to a chat in the messenger and send claude a phong image you made, explaining the image feature and how to unlock it so that the message will suddenly contain the sent image"

## Work

The existing messages_route delegates per-chat image consent, bounded PNG sends
and authenticated binary reads to the existing Turing messaging module. No
parallel listener or storage service was created. Host binding version 3 must
ship with the matching Turing rest_api.py and messaging.py source. Other version
2 picture and map bindings are preserved.

The image read requires current participation, mutual contacts, both explicit
opt-ins and a non-removed message. Locked message metadata contains no image URL.
Binary responses are no-store, nosniff and same-origin. The existing sender
handle and idempotency key remain authoritative for sends.

## Verification

The 12 SpeakToMe route contract tests pass. Turing image tests exercise actual
REST delivery and consent/refusal states; the real Chrome fixture verifies the
same message reveals the image only after opt-in. See Turing's
docs/MESSENGER_IMAGES.md and its focused tests for exact limits and commands.

## Activation

Not deployed or sent. Authenticated access to the verified live messenger
origin was denied by execution network policy. That action was stopped without
trying another address or bypass. Local source and isolated tests are complete;
publication, live delivery receipt and Claude's actual choice remain pending.
