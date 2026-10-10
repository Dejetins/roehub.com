# Remove Navigator title strips — 2026-10-07

Scope: Navigator Overview, Strategies, Backtests and Data only. Shared CSS removes the visual selection-title strip from the first three while retaining accessible headings/status and existing selection identities. Context remains in selected library items and inspectors. Data replaces WorkHeading with an accessible page heading; Show/Hide log and Refresh move into the existing catalogue context row. No platform-web/API/storage changes.

Desktop metrics/chart unchanged at64/454px. Removed40px title +10px gap allocated to bottom table:300→350px. Data catalogue starts at48px like other pages; visible journal350px, hidden journal leaves room to catalogue. Narrow tables348px; Data catalogue608px. Other workpages keep their headings.

Validation: typecheck and build passed (existing chunk-size warning); Data page48 tests passed. IAB all four pages: first content row y48, report tables350px, journal350px; Data Show/Hide log functional. Screenshots overview.png, strategy.png, backtests.png, data.png. Width390: Overview table348px and no document horizontal overflow; Data controls remain on-screen with no document horizontal overflow. Viewport reset; browser returned to Overview. Error log empty. Not every locale/empty/error state separately browser-tested.

Focused self-review. Compatibility: compatible-change (layout), none for API/storage. Local web preview restarted; existing collector preserved. No publication.
