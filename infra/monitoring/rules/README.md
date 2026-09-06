# Alert contract sources

These rules supply alert IDs and runbook links to `tools.docs.generate_runbooks`.
They retain historical monitoring definitions, including unmigrated host alerts;
they are not an automatically installed scrape configuration. The generator
reports unmigrated alerts explicitly. Installation monitoring is generated from
`configs/installation/runtime-service-manifest.json` and release tooling.

Former launchd, Monit, VPS edge and Mac Studio bootstrap assets were retired on
2026-09-06. Their contents remain available in Git history.
