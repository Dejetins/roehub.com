# Stage transition receipt

<!-- prompt-pack-receipt:v1 -->
```json
{
  "schema_version": "prompt-pack-receipt/v1",
  "stage_id": "S05",
  "stage_contract_sha256": "ce603fcef26b8689bafec6deb2d9dcf192dfa733042ad24baa8ad8376617b24e",
  "created_at": "2026-10-07T22:45:30.920784Z",
  "status": "ready",
  "plan": {
    "path": "implementation-plan.md",
    "sha256": "6b647b5b2a3d2abf65b8ea5dd14cad6875ef8b01207ed0a0f7b9b55c0b61e3c9"
  },
  "prompt": {
    "path": "05-durable-replay-and-lazy-details.md",
    "sha256": "fa8e2d07f4220204b05ecfa6429bfa5bab84353034a0d71680df89f534db566c"
  },
  "report": {
    "path": "reports/S05.md",
    "sha256": "efbec5bacf0db550093fe6db710b96c2e8ed0396f1c012dd1cfd11679a032c9e"
  },
  "validation": {
    "profile": "roehub-focused-gates-and-prompt-pack-artifacts/v1",
    "result": "pass",
    "evidence": [
      {
        "path": "reports/S05.md",
        "sha256": "efbec5bacf0db550093fe6db710b96c2e8ed0396f1c012dd1cfd11679a032c9e"
      },
      {
        "path": "reports/S05-evidence.json",
        "sha256": "9eedcd46dcae49f55334afcc6327ee692fbf436e61e436dbb5cd5ebab40dd054"
      }
    ]
  },
  "user_acceptance": null,
  "next_stage": {
    "id": "S06",
    "prompt": {
      "path": "06-policy-readiness-and-rollout.md",
      "sha256": "ab7f6f4c5e1d5067b61657fe3baf5deb9db9c1b52d951ad606c290035d44a5fc"
    },
    "stage_contract_sha256": "7961d84ab1f1e783167293b6dbb910814bdb02bee029294f19985e1a1734facf"
  },
  "next_stage_allowed": true,
  "handoff_reason": "Required checks passed; next prompt may execute when submitted."
}
```
