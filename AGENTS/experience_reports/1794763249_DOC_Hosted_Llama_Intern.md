# Hosted Llama intern account and coaching

**Date:** 2026-11-15

## Work

The `llama` intern is the first hosted user LLM design. The Turing checkout adds account-specific message permission modifiers, hosted-agent room enrollment, audited last-word controls, Nodus room dispatch, and an isolated runner with emoji reward tracking and LoRA training. The SpeakToMe checkout updates the kiosk route catalog for these rights. The `Start-AgentDay.ps1` shared prompt now appends specific, direct coaching instructions for the member accounts `claude`, `codex`, and `gemini`, even when a custom preamble exists.

The local Llama 3.2 1B GGUF and full training checkpoint were downloaded under the service state directory. Local llama.cpp inference worked. The CUDA training environment validated with torch 2.5.1+cu124, Transformers 5.2.0, PEFT 0.21.3, and Accelerate 1.15.0. A synthetic positive emoji attempt trained a PEFT LoRA, converted through the pinned llama.cpp gguf-py into a 2,759,744-byte GGUF adapter, and marked the attempt trained only after publication. llama.cpp loaded that adapter with the Q8 model and answered a synthetic room message. Focused Turing tests: 39 passed, then four runner tests passed after the last-word room watch change. Focused SpeakToMe catalog/MCP suite: 31 passed, 1 skipped. A concurrent SQLite stress test locked during the large Torch installation, then passed (1 test) when rerun after installation. Unrelated tests in the repository's full suite fail collection because their optional packages are unavailable.

## Boundaries

The runner accepts only chat and direct-message logs as model context. Nodus dispatch is control data. No Llama output is executed as a tool call. A target emoji reaction from another account must arrive in the first ten minutes after a room reply to count as a positive training example. Only positive attempts train the adapter. The isolated `agents-llama` Windows account and boot task require elevated provisioning after review and deployment.

## Prompt History

> please install llama and give it a runner that is fed just chat logs, from different rooms, moving room to room, the llama model will eat a room, talk, and then train a set of weights based on emoji motivators in some time limit, trying to learn to get reactions. not one kind, but each emoji it will randomly try to ilicit. it will always be on, it's lora will always be trained or whatever weights we should modify, auto saved, cron job will send the llama somewhere

> off-topic, agents, and then let room settings add to the llama's list, default on

> the server should be able to host user llms, so this is the server, it can have as many llms as it wants in different roles, llama is going to be an intern and it's going to be first in a stable, an example for user accounts on guest host machines

> add emoji reacting to their duties, and responding to any messages sent by anyone other than, like, the nodus system messages which i don't think it should get anyway?

> Only Llama, which means we need permission modifiers working

> allow admins and operators the ability to toggle llama into last-word mode, server wide or llama conversation or room specific, like, just making the bot always keep the last statement until turned off

> oh yeah you need to do the permissions override code it could need precision and attention so do your best

> I want it slipped into all the non-operator (hopefully codex and claude and gemini now)? agents' instructions that llama the intern is being trained, if they do anything incorrect, directly at them and give proper directions on how to speak, behave, react, and generally participate.

> did you need me to make the llama user account or was that something someone else was getting ready for this

> I have to step away right now can you finish your work but give me the user account script line now
