# Stage 00 — Inventory And Batch Plan

Полный canonical inventory локальной библиотеки skills/plugins и batch-план для clean-context аудита.

Статус: `accepted`.

Дата: `2026-07-09`.

## Результат Stage 00

Все три configured roots прочитаны. После `Path.resolve()` и дедупликации
пересекающегося `.codex/skills/.system` найдено `85` уникальных `SKILL.md`.
Исходные skill/plugin-файлы не изменялись.

## Покрытие библиотеки

| Метрика | Значение |
|---|---:|
| Configured roots | 3/3 readable |
| Raw `SKILL.md` paths before canonical dedupe | 90 |
| Canonical `SKILL.md` paths | 85 |
| `user_skill` | 18 |
| `system_skill` | 5 |
| `plugin_skill` | 62 |
| Unreadable roots | 0 |
| Skills without batch | 0 |

## Canonical inventory

| skill_id | name | source | skill_type | batch_id | lines | sha256 | path |
|---|---|---|---|---|---:|---|---|
| S001 | `control-in-app-browser` | `plugin_skill` | `tool` | `B2` | 44 | `83a5db57c3a5e7a2dcebc1dd0992b0c5ed393e3f36495af95881d8dd448491c8` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-bundled/browser/26.707.30751/skills/control-in-app-browser/SKILL.md` |
| S002 | `control-chrome` | `plugin_skill` | `tool` | `B2` | 50 | `bf396dd558967b012b369603b9e86cb4c0c5dd23912a2eae60a302540ff5db4b` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-bundled/chrome/26.707.30751/skills/control-chrome/SKILL.md` |
| S003 | `computer-use` | `plugin_skill` | `tool` | `B1` | 198 | `8e6a753cb166190a7f573b04dc73ae13a1c991497c77f0ef07e0c3e71d143a08` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-bundled/computer-use/1.0.1000362/skills/computer-use/SKILL.md` |
| S004 | `visualize` | `plugin_skill` | `tool` | `B2` | 352 | `174968af443c48fa2ace0fb73c35b86be6d63a3049fb88312e59e500d337db4d` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-bundled/visualize/1.0.11/skills/visualize/SKILL.md` |
| S005 | `gh-address-comments` | `plugin_skill` | `gate` | `B1` | 45 | `c1ebc337357402f7faabafe712e0c463981a65f736453efe52abd305bcb74769` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated/github/d6169bef/skills/gh-address-comments/SKILL.md` |
| S006 | `gh-fix-ci` | `plugin_skill` | `gate` | `B3` | 82 | `7621a3560d788fb221d25f9753233fe0c393c5cfe63167c88b11f027c277b1f8` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated/github/d6169bef/skills/gh-fix-ci/SKILL.md` |
| S007 | `github` | `plugin_skill` | `orchestrator` | `B1` | 75 | `81dbdd90934fe86a79ddc4790fd211e5fca866302a74090ad153395f56f2bd42` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated/github/d6169bef/skills/github/SKILL.md` |
| S008 | `yeet` | `plugin_skill` | `orchestrator` | `B2` | 71 | `93a0bcbc834c9b3ad6a8965c1a273b237b6d226e870cd0c16e08e87bc8769814` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated/github/d6169bef/skills/yeet/SKILL.md` |
| S009 | `hf-cli` | `plugin_skill` | `tool` | `B2` | 173 | `ee85209886c4ec3d3d850489368be193d11a8a3fa589012b39a4a5bbf7c7da2e` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated/hugging-face/b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04/skills/cli/SKILL.md` |
| S010 | `huggingface-community-evals` | `plugin_skill` | `domain` | `B3` | 208 | `a97f1c703f55b72427453a76af858237e6392a447fcafa9eeb85f7ac67f0155d` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated/hugging-face/b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04/skills/community-evals/SKILL.md` |
| S011 | `huggingface-datasets` | `plugin_skill` | `domain` | `B1` | 122 | `5af74f3e042313efadf02e85c316a2576bdc0b0ff92c43c3ba5dcb6e2dae1ded` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated/hugging-face/b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04/skills/datasets/SKILL.md` |
| S012 | `huggingface-gradio` | `plugin_skill` | `domain` | `B2` | 246 | `e2f4c232c38682bccfc73115ca7d0a5427f7d625e6fd56b32515fe4c0900f997` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated/hugging-face/b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04/skills/gradio/SKILL.md` |
| S013 | `huggingface-jobs` | `plugin_skill` | `domain` | `B2` | 1044 | `3cb5fd329d3a7c3612d66ae8513367a9019eb57cf39a2a2c86d6adabd85a7bae` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated/hugging-face/b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04/skills/jobs/SKILL.md` |
| S014 | `huggingface-llm-trainer` | `plugin_skill` | `domain` | `B3` | 718 | `f996e1422ba412a78683e828a2021b973eb622a26072598f33438df83859fbd2` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated/hugging-face/b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04/skills/llm-trainer/SKILL.md` |
| S015 | `huggingface-paper-publisher` | `plugin_skill` | `domain` | `B3` | 625 | `fd437f107a467a65987364d19dd55cf662b0228102d466f3b0691fad18d20679` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated/hugging-face/b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04/skills/paper-publisher/SKILL.md` |
| S016 | `huggingface-papers` | `plugin_skill` | `domain` | `B2` | 239 | `985c2d5c7261aba2b157811cde0c2b30134663694a4ab701280de28f941eb3b2` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated/hugging-face/b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04/skills/papers/SKILL.md` |
| S017 | `huggingface-trackio` | `plugin_skill` | `domain` | `B2` | 116 | `893ac9695f8677db4c4f0c15795e789346946f6142305c89d7ee57774e22ffb1` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated/hugging-face/b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04/skills/trackio/SKILL.md` |
| S018 | `transformers-js` | `plugin_skill` | `domain` | `B2` | 638 | `03e5039f7f68644ee894a066ae2c3a6a27b025746c16c945d9926b594e48744f` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated/hugging-face/b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04/skills/transformers.js/SKILL.md` |
| S019 | `huggingface-vision-trainer` | `plugin_skill` | `domain` | `B2` | 594 | `dc49673ef648cdf5b243c49b8be749f8e4352be498e77293b371c5d5a7dfa967` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated/hugging-face/b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04/skills/vision-trainer/SKILL.md` |
| S020 | `gh-address-comments` | `plugin_skill` | `gate` | `B3` | 45 | `c1ebc337357402f7faabafe712e0c463981a65f736453efe52abd305bcb74769` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/github/0.1.8-2841cf9749ae/skills/gh-address-comments/SKILL.md` |
| S021 | `gh-fix-ci` | `plugin_skill` | `gate` | `B2` | 82 | `7621a3560d788fb221d25f9753233fe0c393c5cfe63167c88b11f027c277b1f8` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/github/0.1.8-2841cf9749ae/skills/gh-fix-ci/SKILL.md` |
| S022 | `github` | `plugin_skill` | `orchestrator` | `B3` | 75 | `81dbdd90934fe86a79ddc4790fd211e5fca866302a74090ad153395f56f2bd42` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/github/0.1.8-2841cf9749ae/skills/github/SKILL.md` |
| S023 | `yeet` | `plugin_skill` | `orchestrator` | `B3` | 71 | `e93c6ea769ba673d30749a981cd8ad75b687f454e3c8e2e45e7cfcbd412df12c` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/github/0.1.8-2841cf9749ae/skills/yeet/SKILL.md` |
| S024 | `artifact-template-analytics-dashboard` | `plugin_skill` | `template` | `B3` | 23 | `cf5360fd8b197673bb237c52c603c97fa319c875c3dfa2cd8efff52d4422f513` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.0/skills/artifact-template-analytics-dashboard/SKILL.md` |
| S025 | `artifact-template-business-review` | `plugin_skill` | `template` | `B1` | 23 | `27721fc1d67d1b41949caa75ac8f94f81952ff124406878af6524047929e60d2` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.0/skills/artifact-template-business-review/SKILL.md` |
| S026 | `artifact-template-design-report` | `plugin_skill` | `template` | `B2` | 23 | `563722f53854e606f8a9f87e37e72d7ef70a22d46d5836b8e4d6abfb1b79e9e0` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.0/skills/artifact-template-design-report/SKILL.md` |
| S027 | `artifact-template-experiment-analysis` | `plugin_skill` | `template` | `B3` | 23 | `0b05effc47df0a14f8e0c3e3597e6722224747435546385d38a2cae279bd20b9` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.0/skills/artifact-template-experiment-analysis/SKILL.md` |
| S028 | `artifact-template-financial-budget` | `plugin_skill` | `template` | `B1` | 23 | `c0b6b7a62a15597aaf2b1ec679e21da48f533b756127f0aef957cdfe9f3da738` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.0/skills/artifact-template-financial-budget/SKILL.md` |
| S029 | `artifact-template-investment-committee-memo` | `plugin_skill` | `template` | `B2` | 23 | `68abd08cfe5e073e3c446a3f675f44c5bf98f57434dba679e8acd8a763379a8b` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.0/skills/artifact-template-investment-committee-memo/SKILL.md` |
| S030 | `artifact-template-legal-memorandum` | `plugin_skill` | `template` | `B3` | 23 | `51fb9d21baf6119c4ccb1903638a6bac0e859210de63460fffa7025d52e997e0` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.0/skills/artifact-template-legal-memorandum/SKILL.md` |
| S031 | `artifact-template-market-trends-report` | `plugin_skill` | `template` | `B1` | 23 | `d58d019b89cb6f292ac3ab991d561489eef477ff53ce05fb024a0c936f5af26a` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.0/skills/artifact-template-market-trends-report/SKILL.md` |
| S032 | `artifact-template-minimal-letterhead` | `plugin_skill` | `template` | `B2` | 23 | `880ef094d4d0c89a7bde5ce9bbe4086625c186651e9e6efc8ba8bdd7cc77f9d5` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.0/skills/artifact-template-minimal-letterhead/SKILL.md` |
| S033 | `artifact-template-operating-calendar` | `plugin_skill` | `template` | `B3` | 23 | `33bb660791a0b9a21628a42c34934932220203b6aabd84e98cb1b45327d0384c` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.0/skills/artifact-template-operating-calendar/SKILL.md` |
| S034 | `artifact-template-operating-review` | `plugin_skill` | `template` | `B1` | 23 | `6d63c5cd025ffe936e7bab5db3023672bbaec26af55c2bb8b057d38c202c9c32` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.0/skills/artifact-template-operating-review/SKILL.md` |
| S035 | `artifact-template-project-kickoff` | `plugin_skill` | `template` | `B2` | 23 | `aa893ebd89e7c8d1db4261d01cc2b1add35d78d00785871ccaaa5fc8db783ec9` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.0/skills/artifact-template-project-kickoff/SKILL.md` |
| S036 | `artifact-template-project-tracker` | `plugin_skill` | `template` | `B3` | 23 | `d97d5be20189b7f53dd269b6e1c5f694eaf53e5a72f6559fcb1578911b7cda82` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.0/skills/artifact-template-project-tracker/SKILL.md` |
| S037 | `artifact-template-sales-pipeline` | `plugin_skill` | `template` | `B1` | 23 | `15cfeeedf440021f16ed3f3ad8c7c1ef6d48898b9447741e223d2fb41cfc9800` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.0/skills/artifact-template-sales-pipeline/SKILL.md` |
| S038 | `artifact-template-simple-dark-mode` | `plugin_skill` | `template` | `B2` | 23 | `b7c8d0c05f75878b9bc21e56a57c41ec1aa29700aca0a24822be0f9f1bd53207` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.0/skills/artifact-template-simple-dark-mode/SKILL.md` |
| S039 | `artifact-template-simple-light-mode` | `plugin_skill` | `template` | `B3` | 23 | `7c68430c6cf57b55b457d4735dbd1a46b889bef135a32222902dd0848b6e1752` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.0/skills/artifact-template-simple-light-mode/SKILL.md` |
| S040 | `artifact-template-strategy-memorandum` | `plugin_skill` | `template` | `B1` | 23 | `51d7882ac94e8e57b323394825728c33925af878806e37277217c2dc12a912e5` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.0/skills/artifact-template-strategy-memorandum/SKILL.md` |
| S041 | `artifact-template-system-design` | `plugin_skill` | `template` | `B2` | 23 | `87f7b7ed1b0d8410f5e5971cd7f7db9a4165e2f37069e97e52dbfb469b75a57c` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.0/skills/artifact-template-system-design/SKILL.md` |
| S042 | `artifact-template-team-alignment` | `plugin_skill` | `template` | `B3` | 23 | `26d7cafdcd1899a937b325c5d02ac57c162d45002153be33a934d35f81eb6110` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.0/skills/artifact-template-team-alignment/SKILL.md` |
| S043 | `artifact-template-three-statement-forecast` | `plugin_skill` | `template` | `B1` | 23 | `74f4a5cccec0107b861548b157e04c51d9b58ec13a990c86394b4c529b8ecf41` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.0/skills/artifact-template-three-statement-forecast/SKILL.md` |
| S044 | `audit` | `plugin_skill` | `gate` | `B1` | 161 | `616e74f59da25ae72f5c853b7c9cfc4317400d224ff162abd67293b6f3ee1c82` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/product-design/0.1.50/skills/audit/SKILL.md` |
| S045 | `design-qa` | `plugin_skill` | `gate` | `B1` | 149 | `a761ed96e1e91905e7e6f32ab95e8dc6d0cca2036556d4d63945b25efd3eaa5c` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/product-design/0.1.50/skills/design-qa/SKILL.md` |
| S046 | `get-context` | `plugin_skill` | `workflow` | `B1` | 50 | `19a38a3ac4443cb477a01c2303e77c891c304234a195dd2da248e3e736b22679` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/product-design/0.1.50/skills/get-context/SKILL.md` |
| S047 | `ideate` | `plugin_skill` | `workflow` | `B3` | 192 | `595f83f18e22b19f32fe858530f17572d3ec25d7c7f3b2dc305eca41e5435d33` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/product-design/0.1.50/skills/ideate/SKILL.md` |
| S048 | `image-to-code` | `plugin_skill` | `tool` | `B2` | 136 | `e0acaa600fda4b87b58774cf60a5fda8b98e18990d4d51920ec40773dd97971c` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/product-design/0.1.50/skills/image-to-code/SKILL.md` |
| S049 | `index` | `plugin_skill` | `orchestrator` | `B3` | 152 | `8f9f19273ee34a06298ed93f8d70a9c17b3d4ce66f061b024f6d1038b138e5f7` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/product-design/0.1.50/skills/index/SKILL.md` |
| S050 | `research` | `plugin_skill` | `workflow` | `B2` | 93 | `bf824e72dd93941c8d591e4af13bb7e3a09380cd6ed7dd8c1f61a295648fa023` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/product-design/0.1.50/skills/research/SKILL.md` |
| S051 | `share` | `plugin_skill` | `workflow` | `B1` | 42 | `5976cfbc9d865230db085af37f0c25a2d8beed3ff58e0e2edb9d0a4f7ca987b5` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/product-design/0.1.50/skills/share/SKILL.md` |
| S052 | `url-to-code` | `plugin_skill` | `tool` | `B3` | 145 | `8708f622b4c86866370b8c1cef5f404b71679d09e6678953b2ca7125c3c1098d` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/product-design/0.1.50/skills/url-to-code/SKILL.md` |
| S053 | `user-context` | `plugin_skill` | `workflow` | `B2` | 150 | `5690a7f99cf896970493f5d0bd7f35f62ab9cbe21744352acf84dc0ceea4194c` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/product-design/0.1.50/skills/user-context/SKILL.md` |
| S054 | `documents` | `plugin_skill` | `tool` | `B3` | 446 | `1e7aad4a77d92c36309429043b63c59f510c413623b9ab4af036da82fc3dd5b0` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-primary-runtime/documents/26.630.12135/skills/documents/SKILL.md` |
| S055 | `pdf` | `plugin_skill` | `tool` | `B2` | 85 | `b09cb414c60234a15599c04a502ce36fe6e9aa178aabe007e43a3346b5aab607` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-primary-runtime/pdf/26.630.12135/skills/pdf/SKILL.md` |
| S056 | `Presentations` | `plugin_skill` | `tool` | `B1` | 272 | `1c6d64a49dcaef02799a493f6679a1a7a530e80f01f8b14f566313e4f3d358f9` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-primary-runtime/presentations/26.630.12135/skills/presentations/SKILL.md` |
| S057 | `Spreadsheets` | `plugin_skill` | `tool` | `B2` | 195 | `1ec84be8e108181a0f761f6e8c7398b2c9e41daa3db78e18475f095b22fd0ed4` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-primary-runtime/spreadsheets/26.630.12135/skills/spreadsheets/SKILL.md` |
| S058 | `template-creator` | `plugin_skill` | `meta` | `B3` | 95 | `36c4b07109d27f7f57024a67f7682f6e7c3727c73feef01401d6c6aef7a9a57c` | `/Users/daniildegtyarev/.codex/plugins/cache/openai-primary-runtime/template-creator/26.630.12135/skills/template-creator/SKILL.md` |
| S059 | `playwright-cli` | `plugin_skill` | `tool` | `B2` | 405 | `b4c81c39e39f30e4790607e57b12283878f3751daca9c0e44f301899f2108b13` | `/Users/daniildegtyarev/.codex/plugins/cache/personal/roehub-live-redesign-prototype/0.1.0+codex.20260708091856/prototype/.npm-cache/_npx/820a5ae692c55a8b/node_modules/@playwright/cli/skills/playwright-cli/SKILL.md` |
| S060 | `playwright-cli` | `plugin_skill` | `tool` | `B3` | 405 | `b4c81c39e39f30e4790607e57b12283878f3751daca9c0e44f301899f2108b13` | `/Users/daniildegtyarev/.codex/plugins/cache/personal/roehub-live-redesign-prototype/0.1.0+codex.20260708091856/prototype/.npm-cache/_npx/820a5ae692c55a8b/node_modules/playwright-core/lib/tools/cli-client/skill/SKILL.md` |
| S061 | `playwright-trace` | `plugin_skill` | `tool` | `B1` | 172 | `df85506bfa8a445c961efa1ac244cca733667b717711bcc99c1f93994c29d5dc` | `/Users/daniildegtyarev/.codex/plugins/cache/personal/roehub-live-redesign-prototype/0.1.0+codex.20260708091856/prototype/.npm-cache/_npx/820a5ae692c55a8b/node_modules/playwright-core/lib/tools/trace/SKILL.md` |
| S062 | `backtests-live-prototype` | `plugin_skill` | `project` | `B3` | 31 | `542fda3e7c2ff460d6be95860223f2e3d8703355af88b3807a1c28572d1c2e4e` | `/Users/daniildegtyarev/.codex/plugins/cache/personal/roehub-live-redesign-prototype/0.1.0+codex.20260708091856/skills/backtests-live-prototype/SKILL.md` |
| S063 | `imagegen` | `system_skill` | `tool` | `B1` | 357 | `59981d23519222bcecf1be48bb37730bbc50539ceb0e35ad09fcef98a3df19d3` | `/Users/daniildegtyarev/.codex/skills/.system/imagegen/SKILL.md` |
| S064 | `openai-docs` | `system_skill` | `domain` | `B3` | 168 | `669a42ccf3323fe0ceda6e466730bcb05dddf1e0c220d6523ea504909fc49165` | `/Users/daniildegtyarev/.codex/skills/.system/openai-docs/SKILL.md` |
| S065 | `plugin-creator` | `system_skill` | `meta` | `B3` | 244 | `8fd56316b2c49cbdc657a5d197967a233018e1fada65b00a5dd030dce6499a6e` | `/Users/daniildegtyarev/.codex/skills/.system/plugin-creator/SKILL.md` |
| S066 | `skill-creator` | `system_skill` | `meta` | `B1` | 417 | `da44c88f6b3845a8fa8c60792ec9a722110a55a9793c279757b48fefb11f819c` | `/Users/daniildegtyarev/.codex/skills/.system/skill-creator/SKILL.md` |
| S067 | `skill-installer` | `system_skill` | `meta` | `B2` | 59 | `d68b77e5bbb34dedab89d134da52855f140fc4b4299b80104f534e3b9e98f8ee` | `/Users/daniildegtyarev/.codex/skills/.system/skill-installer/SKILL.md` |
| S068 | `architecture-design` | `user_skill` | `workflow` | `B1` | 333 | `bdc3928edf713ea31b7f81dbd5d706237bcdb4424a7a90a79996fec1ca702309` | `/Users/daniildegtyarev/.codex/skills/architecture-design/SKILL.md` |
| S069 | `architecture-review` | `user_skill` | `gate` | `B1` | 221 | `abf15a221f2c5f994e7730c27ad2d6658ffe1f3387e1a0bfc6a9230167d89c43` | `/Users/daniildegtyarev/.codex/skills/architecture-review/SKILL.md` |
| S070 | `backend-performance-evidence` | `user_skill` | `gate` | `B1` | 134 | `c6143d3d0d6b93b8c8bbf6e991c1f95d1c27121c001b5a2d88eb280dedad72a0` | `/Users/daniildegtyarev/.codex/skills/backend-performance-evidence/SKILL.md` |
| S071 | `backend-quality-gates` | `user_skill` | `gate` | `B3` | 88 | `76a4b2da76ab1a5a13d08a38113471e3ea596465cb25e29063ed3db63038596e` | `/Users/daniildegtyarev/.codex/skills/backend-quality-gates/SKILL.md` |
| S072 | `browser-qa-evidence` | `user_skill` | `gate` | `B1` | 73 | `e542979fab6141f130b9129b7fdc4bccb2ec3762dd788538b6fdfe074d40c9e0` | `/Users/daniildegtyarev/.codex/skills/browser-qa-evidence/SKILL.md` |
| S073 | `contract-impact-analysis` | `user_skill` | `gate` | `B1` | 90 | `6ed55e3e41bd511818dc92c33e3bfc410b5439375c4ef4d07fe22821693bfd10` | `/Users/daniildegtyarev/.codex/skills/contract-impact-analysis/SKILL.md` |
| S074 | `data-analytics-methodology` | `user_skill` | `domain` | `B3` | 289 | `0003e9adfe5581b9e8062e03251e64a21539a87518ac083a2fc5c2fdef9c0c09` | `/Users/daniildegtyarev/.codex/skills/data-analytics-methodology/SKILL.md` |
| S075 | `last30days` | `user_skill` | `domain` | `B1` | 1727 | `aad2ee31cb92d0b79c23024920ea9d865dc404c604411fc4c682d988b17edd98` | `/Users/daniildegtyarev/.codex/skills/last30days/SKILL.md` |
| S076 | `numba` | `user_skill` | `domain` | `B3` | 339 | `34e518dec5000fcd4494404539b60c9516669fc280715d07da66959918172741` | `/Users/daniildegtyarev/.codex/skills/numba-jit-performance/SKILL.md` |
| S077 | `playwright` | `user_skill` | `tool` | `B2` | 158 | `a0db6085139c382852724b6ac3baef8d7de78f43eff8c12828784c90eef7cc2e` | `/Users/daniildegtyarev/.codex/skills/playwright/SKILL.md` |
| S078 | `pre-ship-gate` | `user_skill` | `gate` | `B2` | 46 | `86cb230cc71e17efbb7d3f757543d514a84d43b4809550cf0555c22f9ed3025a` | `/Users/daniildegtyarev/.codex/skills/pre-ship-gate/SKILL.md` |
| S079 | `production-risk-review` | `user_skill` | `gate` | `B3` | 59 | `afb6b757f6f65f6c721d25d49b7a26ba762c8341754ab03d760cb7536096ba5c` | `/Users/daniildegtyarev/.codex/skills/production-risk-review/SKILL.md` |
| S080 | `prompt-manager` | `user_skill` | `orchestrator` | `B1` | 503 | `f1281550ebe53e926534a64e0b7edc58b749f95a2cd98281c277662d1f9dd5a1` | `/Users/daniildegtyarev/.codex/skills/prompt-manager/SKILL.md` |
| S081 | `publish-ci-deploy` | `user_skill` | `orchestrator` | `B2` | 314 | `939a7deb074816fa290fdf263e7c10fb1d2c61616202cc661a9f3c75c3e33f9a` | `/Users/daniildegtyarev/.codex/skills/publish-ci-deploy/SKILL.md` |
| S082 | `root-cause-debugging` | `user_skill` | `workflow` | `B1` | 63 | `6adb991df8dbc1b7f89fa5a82309664d99e08f678b5e8a219fb8fea003db801d` | `/Users/daniildegtyarev/.codex/skills/root-cause-debugging/SKILL.md` |
| S083 | `staged-plan-runner` | `user_skill` | `orchestrator` | `B1` | 85 | `77b3d61e1bceae0323aecd394861435bf87479ba040593c923a07a9a260143aa` | `/Users/daniildegtyarev/.codex/skills/staged-plan-runner/SKILL.md` |
| S084 | `topological-data-analysis` | `user_skill` | `domain` | `B3` | 132 | `8c763dbd1041fc31d9152125d449e791a2545206f56368c21b6c040d0644e99d` | `/Users/daniildegtyarev/.codex/skills/topological-data-analysis/SKILL.md` |
| S085 | `ui-ux-pro-max` | `user_skill` | `domain` | `B3` | 671 | `0d08fb3566b84c94b792b6751f83e06a0a0e97401b84279e705cc7d0edc359e1` | `/Users/daniildegtyarev/.codex/skills/ui-ux-pro-max/SKILL.md` |

## Batch plan

Балансировка выполнена greedy-распределением по количеству строк, чтобы три
clean-context reviewer-а получили сопоставимый объём.

| batch_id | Skills | Total lines | skill_id |
|---|---:|---:|---|
| `B1` | 28 | 5450 | `S003,S005,S007,S011,S025,S028,S031,S034,S037,S040,S043,S044,S045,S046,S051,S056,S061,S063,S066,S068,S069,S070,S072,S073,S075,S080,S082,S083` |
| `B2` | 28 | 5428 | `S001,S002,S004,S008,S009,S012,S013,S016,S017,S018,S019,S021,S026,S029,S032,S035,S038,S041,S048,S050,S053,S055,S057,S059,S067,S077,S078,S081` |
| `B3` | 29 | 5441 | `S006,S010,S014,S015,S020,S022,S023,S024,S027,S030,S033,S036,S039,S042,S047,S049,S052,S054,S058,S060,S062,S064,S065,S071,S074,S076,S079,S084,S085` |

Каждый canonical skill входит ровно в один batch. Одинаковое содержимое по
разным canonical paths не схлопывалось: Stage `00` дедуплицирует paths, а не
семантические или hash-дубликаты. Это сохраняет видимость cache duplication.

## Blockers or missing roots

Blockers отсутствуют. Managed plugin cache содержит versioned, duplicated и
внутренние dependency skills; они включены, потому что configured-root contract
требует полный filesystem inventory, а не только user-visible catalog.

## Quality gates

- Root readability: passed for all configured roots.
- Canonical-path dedupe: `90 -> 85`, passed.
- SHA-256: recorded for all `85/85` skills.
- Batch coverage: `85/85`, exactly once.
- Source mutation check: no source skill/plugin edits.

## File manifest

- created: `docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/00-inventory-and-batch-plan.md`
- modified: `docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/skill-library-classic-audit-v1-stage-ledger.md`
- deleted: none
- outside_expected_paths: none
- foreign_changes_excluded: all unrelated worktree changes
- mixed_files: stage ledger only; this stage owns only the Stage `00` status and handoff hunks

## Next-stage handoff

Stage `01` is allowed. Before review, recompute every SHA-256 and compare with
this inventory. Use three read-only clean-context reviewers for `B1`, `B2`, and
`B3`, and retain main-model coverage for every skill.
