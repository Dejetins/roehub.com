# Single-Shot LM Studio Contract Retirement

Date: 2026-05-16

## Gate Markers

- accepted: false
- blocking_reason: single-shot prompt/blob runtime retired; LM Studio tools
  runtime is pending implementation
- next_prompt_allowed: true

## What Changed

The previous current contract is no longer a rollout target:

- `BacktestConfigLLMGateway`
- `LMStudioOpenAICompatibleAdapter`
- prompt profiles that build a full prompt envelope
- model-visible full capability blobs
- adapter smoke based only on `messages` plus `response_format`

The failed Iteration 15 benchmark remains historical blocker evidence. It must
not be used as acceptance for the next tool-agent implementation.

## Retained Safety Foundation

These parts remain active and should be reused by the new prompt pack:

- job storage, quota, status and SSE shell
- catalog resolver as backend-owned source inventory
- deterministic input/output gates
- `BacktestAiConfigValidator`
- `BacktestPreflightService`
- indicator executable support, window bounds and artifact coverage checks

## Next Contract

The next implementation must use LM Studio OpenAI-compatible tools through a
backend-owned executor. The model may request context through allowlisted tools;
the backend validates paths/resources, redacts output, enforces size limits and
keeps final validation authoritative.
