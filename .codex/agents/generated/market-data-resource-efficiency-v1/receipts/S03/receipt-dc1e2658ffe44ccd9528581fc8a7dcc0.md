# Stage transition receipt

<!-- prompt-pack-receipt:v1 -->
```json
{
  "schema_version": "prompt-pack-receipt/v1",
  "stage_id": "S03",
  "stage_contract_sha256": "72070f0dfda93740c9ec2e745568520dfdaf96a9348d17f7076aec65fc9d796d",
  "created_at": "2026-10-10T21:33:29.070811Z",
  "status": "ready",
  "plan": {
    "path": "implementation-plan.md",
    "sha256": "5890e10ea64f08addb89fe679c37aa3d56c8ab11f9354106d874309391606b4e"
  },
  "prompt": {
    "path": "03-rolling-ingestion-and-provider-limits.md",
    "sha256": "680359d2829f0a98f5cd574426a98110f1cf6e45bd77bdb4c1a6776da3b72ef0"
  },
  "report": {
    "path": "reports/S03.md",
    "sha256": "caa882db36fa6b7e2a858fd344a1033e923f76d479d414a58fa21998148986f9"
  },
  "validation": {
    "profile": "roehub-focused-gates-and-prompt-pack-artifacts/v1",
    "result": "pass",
    "evidence": [
      {
        "path": "reports/S03.md",
        "sha256": "caa882db36fa6b7e2a858fd344a1033e923f76d479d414a58fa21998148986f9"
      },
      {
        "path": "reports/S03/evidence-sha256-final-01.json",
        "sha256": "de752f8631c9e529924c7bc670675de6918706a2a3ea25ac7cd74b0153fe8689"
      },
      {
        "path": "reports/S03/source-final-06.json",
        "sha256": "aa5e4810278847b609503723d86bee07142e73765f3c7776db48d707745be0a3"
      },
      {
        "path": "reports/S03/independent-review-final-05.json",
        "sha256": "c663e03a3400089607424d67594e1e47c4a397dcb541c34d82f69043652236af"
      },
      {
        "path": "reports/S03/map-final-05.txt",
        "sha256": "1e0c92e18b1237c70fd573bea97188d646d279b61962e17f64dfe72d5f11e4d3"
      }
    ]
  },
  "user_acceptance": null,
  "next_stage": {
    "id": "S04",
    "prompt": {
      "path": "04-async-batched-recovery.md",
      "sha256": "960449b7124012d3c4f7558f092edbd37730d47bfef18ca33e4bd29bdb53f92e"
    },
    "stage_contract_sha256": "413b635c2bbbfcdb2c4bd36b8176426014a9ff5e1c52c9b790a9f7d2035f876b"
  },
  "next_stage_allowed": true,
  "handoff_reason": "Required checks passed; next prompt may execute when submitted."
}
```
