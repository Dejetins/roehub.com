# Stage transition receipt

<!-- prompt-pack-receipt:v1 -->
```json
{
  "schema_version": "prompt-pack-receipt/v1",
  "stage_id": "S05",
  "stage_contract_sha256": "7847196aa9651fb62867b113ab917dd5d2f9874290d42d1d055263a8fab4a9a4",
  "created_at": "2026-10-08T18:09:51.758746Z",
  "status": "ready",
  "plan": {
    "path": "implementation-plan.md",
    "sha256": "70dff4403743130c26e84dd907f779b2d4f6f9e91ff7b1f11eab70e83cc83a12"
  },
  "prompt": {
    "path": "05-scheduling-and-progress.md",
    "sha256": "93a5f13e669fe7d6fac589e96c09c2c52bbea51aca778d96c91c0084561d31c2"
  },
  "report": {
    "path": "reports/S05.md",
    "sha256": "6d50d9371d48c0170691a2bb771d144faadc43f6bffe8a1aed2dafcb54d8dde0"
  },
  "validation": {
    "profile": "roehub-focused-gates-and-prompt-pack-artifacts/v1",
    "result": "pass",
    "evidence": [
      {
        "path": "reports/S05.md",
        "sha256": "6d50d9371d48c0170691a2bb771d144faadc43f6bffe8a1aed2dafcb54d8dde0"
      },
      {
        "path": "reports/S05-review.md",
        "sha256": "749343eac7f3e1cdeb55b71288c7099cd9e64bac4386fa40ba5e3ca195c3ebb3"
      },
      {
        "path": "reports/S05-source-bindings.json",
        "sha256": "8fab669cc45266fa56247aef9307a8910e17fb7d1d0182937727d6262a886087"
      },
      {
        "path": "reports/S05-ownership.json",
        "sha256": "a94928c48b195de75fc9e642dfeaf88ae7795ad5423c209f4dffab6a1f053a0f"
      },
      {
        "path": "reports/S05-unit.json",
        "sha256": "5468ec3519beae18b811d1016df4dc15f523538a91a964673063aeddf5995afc"
      },
      {
        "path": "reports/S05-postgres.json",
        "sha256": "e94f72c32c8b756a01a10815ce358a6d714cf4140d6b3b7e94fb80d240d2e310"
      },
      {
        "path": "reports/S05-physical.json",
        "sha256": "762d0b452d97195abff4cbbeef13beaa276908be61b790fd86fa2f7e4695d4a8"
      },
      {
        "path": "reports/S05-migration.json",
        "sha256": "b88cc2329548b6d14a43d475da41d9c484b901422dc26d2981ec7b10b1a9706e"
      },
      {
        "path": "reports/S05-lint.json",
        "sha256": "6046c9f576dbbfe19419a0f8d7d49164841fd4823759428d03e67098c8a268ef"
      },
      {
        "path": "reports/S05-types.json",
        "sha256": "b7bfb004e0bf1e6c8667586699dd9216ae05d6131cc3471dcd6ed393a967da8f"
      },
      {
        "path": "reports/S05-diff-check.json",
        "sha256": "7afcad36ff954c3f386c75ae3d13a05cac96bfd1c95c5fe7a5f0d0b8d66cf785"
      }
    ]
  },
  "user_acceptance": null,
  "next_stage": {
    "id": "S06",
    "prompt": {
      "path": "06-api-and-storage-boundary-r2.md",
      "sha256": "1d11375c6aaf953cb6dbb9fcb9e2cd3cb6624026446aa450169224d2b79dd6c6"
    },
    "stage_contract_sha256": "eef3382672f5af25feb66b5c921cb0082fd15ec7056127e24075fc5fd937678d"
  },
  "next_stage_allowed": true,
  "handoff_reason": "Required checks passed; next prompt may execute when submitted."
}
```
