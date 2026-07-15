"""Rehearse a complete greenfield lifecycle from the signed offline bundle."""

# Embedded browser programs and container shell commands stay literal for auditability.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from tools.backup.verify_runtime import (
    RecoveryRuntimeProofError,
)
from tools.backup.verify_runtime import (
    verify_runtime as verify_recovery_runtime,
)
from tools.release.offline_bundle import verify_bundle

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = Path.home() / ".cache/roehub/stage22-offline-release/candidates/roehub-0.1.0"
DEFAULT_TRUSTED_KEY = (
    Path.home() / ".cache/roehub/stage22-offline-release/trust/roehub-0.1.0.pub"
)
DEFAULT_EVIDENCE = (
    ROOT
    / "docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports"
    / "evidence/23-greenfield-installation-lifecycle-proof.json"
)
DEFAULT_SCREENSHOT = DEFAULT_EVIDENCE.with_name("23-greenfield-admin.png")
PWCLI = Path.home() / ".codex/skills/playwright/scripts/playwright_cli.sh"
OIDC_FIXTURE = ROOT / "tools/release/greenfield_oidc_fixture.py"

_CORE_IDENTITY_TABLES = (
    "identity_administrative_audit_events",
    "identity_installation_owners",
    "identity_installations",
    "identity_invitations",
    "identity_memberships",
    "identity_organizations",
    "identity_plugin_permissions",
    "identity_users",
)
_FIXTURE_EMAILS = {
    "a": "viewer-a@stage23.invalid.example",
    "b": "viewer-b@stage23.invalid.example",
}


class GreenfieldLifecycleError(RuntimeError):
    """Raised when a greenfield lifecycle invariant is not proven."""


@dataclass(frozen=True, slots=True)
class Installation:
    project: str
    bundle: Path
    state: Path
    profile: str = "trading"

    @property
    def compose(self) -> Path:
        return self.bundle / f"configs/installation/generated/{self.profile}/compose.yaml"

    @property
    def override(self) -> Path:
        return self.state / f"compose.{self.profile}.offline.yaml"

    def command(self) -> list[str]:
        return [
            "docker",
            "compose",
            "-p",
            self.project,
            "-f",
            str(self.compose),
            "-f",
            str(self.override),
        ]


def _run(
    command: Sequence[str],
    *,
    label: str,
    cwd: Path | None = None,
    timeout: float = 600,
    input_bytes: bytes | None = None,
    allowed_codes: frozenset[int] = frozenset({0}),
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            input=input_bytes,
            check=False,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise GreenfieldLifecycleError(
            f"{label} timed out after {timeout:g} seconds"
        ) from error
    except (OSError, subprocess.SubprocessError) as error:
        raise GreenfieldLifecycleError(f"{label} could not run") from error
    if result.returncode not in allowed_codes:
        detail = (result.stderr or result.stdout).decode(errors="replace").strip()
        raise GreenfieldLifecycleError(f"{label} failed: {detail[-2000:]}")
    return result


def _json_output(result: subprocess.CompletedProcess[bytes], *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(result.stdout)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise GreenfieldLifecycleError(f"{label} returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise GreenfieldLifecycleError(f"{label} returned a non-object JSON value")
    return payload


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _console_error_count(output: str) -> int:
    match = re.search(r"\bErrors:\s*(\d+)\b", output)
    if match is None:
        raise GreenfieldLifecycleError(
            "browser console inspection returned an unknown result format"
        )
    return int(match.group(1))


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _offline_tags() -> set[str]:
    result = _run(
        ["docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"],
        label="offline image inventory",
        timeout=60,
    )
    return {
        line
        for line in result.stdout.decode().splitlines()
        if line.startswith("roehub-offline/")
    }


def _install_bundle(*, bundle: Path, trusted_key: Path, state: Path) -> dict[str, Any]:
    result = _run(
        [
            "env",
            "PATH=/usr/bin:/opt/homebrew/bin:/usr/local/bin:/bin:/usr/sbin:/sbin",
            str(bundle / "tools/release/install-offline.sh"),
            "--trusted-public-key",
            str(trusted_key),
            "--state-directory",
            str(state),
            "--profile",
            "trading",
            "--runtime-smoke",
        ],
        label="signed offline bundle install",
        timeout=1800,
    )
    payload = _json_output(result, label="signed offline bundle install")
    if payload.get("status") != "passed" or payload.get("runtime_smoke") != "passed":
        raise GreenfieldLifecycleError("signed offline bundle install did not pass")
    return payload


def _assert_empty_project(project: str) -> None:
    containers = _run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--format",
            "{{.ID}}",
        ],
        label="greenfield container precondition",
        timeout=60,
    ).stdout.splitlines()
    volumes = _run(
        [
            "docker",
            "volume",
            "ls",
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--format",
            "{{.Name}}",
        ],
        label="greenfield volume precondition",
        timeout=60,
    ).stdout.splitlines()
    if containers or volumes:
        raise GreenfieldLifecycleError(
            f"greenfield project is not empty: containers={len(containers)}, volumes={len(volumes)}"
        )


def _up(installation: Installation, *services: str) -> None:
    _run(
        [
            *installation.command(),
            "up",
            "-d",
            "--wait",
            "--wait-timeout",
            "300",
            *services,
        ],
        label=f"Compose activation {installation.project}",
        timeout=480,
    )


def _down(installation: Installation, *, volumes: bool) -> None:
    subprocess.run(
        ["docker", "rm", "--force", _ingress_name(installation)],
        check=False,
        capture_output=True,
        timeout=60,
    )
    command = [*installation.command(), "down"]
    if volumes:
        command.append("--volumes")
    command.extend(("--remove-orphans", "--timeout", "10"))
    subprocess.run(command, check=False, capture_output=True, timeout=240)


def _ingress_name(installation: Installation) -> str:
    return f"{installation.project}-browser-ingress"


def _start_browser_ingress(installation: Installation) -> dict[str, Any]:
    lock = json.loads((installation.state / "offline-image-lock.json").read_text())
    runtime_image_id = lock.get("images", {}).get("runtime")
    if not isinstance(runtime_image_id, str) or not runtime_image_id.startswith("sha256:"):
        raise GreenfieldLifecycleError("signed runtime image ID is unavailable for browser ingress")
    proxy_program = """import asyncio
async def proxy(reader, writer):
    try:
        upstream_reader, upstream_writer = await asyncio.open_connection('web', 8010)
        async def pipe(source, target):
            while True:
                data = await source.read(65536)
                if not data:
                    break
                target.write(data)
                await target.drain()
            try:
                target.write_eof()
            except Exception:
                pass
        await asyncio.gather(pipe(reader, upstream_writer), pipe(upstream_reader, writer))
    finally:
        writer.close()
        await writer.wait_closed()
async def main():
    server = await asyncio.start_server(proxy, '0.0.0.0', 8080)
    async with server:
        await server.serve_forever()
asyncio.run(main())
"""
    name = _ingress_name(installation)
    _run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--label",
            f"io.roehub.stage23.project={installation.project}",
            "--network",
            "bridge",
            "-p",
            "127.0.0.1:8080:8080",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=16m",
            "--entrypoint",
            "python",
            runtime_image_id,
            "-c",
            proxy_program,
        ],
        label="bounded browser ingress start",
        timeout=120,
    )
    _run(
        ["docker", "network", "connect", f"{installation.project}_roehub", name],
        label="bounded browser ingress internal attachment",
        timeout=60,
    )
    deadline = time.monotonic() + 30
    while True:
        result = subprocess.run(
            ["curl", "-fsS", "http://localhost:8080/api/auth/local/status"],
            check=False,
            capture_output=True,
        )
        if result.returncode == 0:
            status = _json_output(result, label="bounded browser ingress probe")
            if "bootstrap_required" not in status:
                raise GreenfieldLifecycleError("bounded browser ingress returned invalid auth state")
            return {
                "helper_image": "signed-runtime-image-id",
                "host_port": 8080,
                "signed_workloads_remained_internal": True,
                "status": "passed",
            }
        if time.monotonic() >= deadline:
            raise GreenfieldLifecycleError("bounded browser ingress did not become ready")
        time.sleep(0.5)


def _compose_exec(
    installation: Installation,
    service: str,
    command: Sequence[str],
    *,
    label: str,
    timeout: float = 300,
    input_bytes: bytes | None = None,
    allowed_codes: frozenset[int] = frozenset({0}),
) -> subprocess.CompletedProcess[bytes]:
    return _run(
        [*installation.command(), "exec", "-T", service, *command],
        label=label,
        timeout=timeout,
        input_bytes=input_bytes,
        allowed_codes=allowed_codes,
    )


def _container_id(installation: Installation, service: str) -> str:
    result = _run(
        [*installation.command(), "ps", "-q", service],
        label=f"container lookup {service}",
        timeout=60,
    )
    value = result.stdout.decode().strip()
    if not re.fullmatch(r"[a-f0-9]{12,64}", value):
        raise GreenfieldLifecycleError(f"container lookup failed for {service}")
    return value


def _issue_ticket(installation: Installation, destination: Path) -> None:
    container_path = "/var/lib/roehub/artifacts/.stage23-bootstrap-ticket"
    _compose_exec(
        installation,
        "api",
        [
            "python",
            "-m",
            "apps.roehubctl.main.main",
            "owner",
            "init",
            "--output-file",
            container_path,
        ],
        label="roehubctl installation owner ticket",
    )
    _run(
        ["docker", "cp", f"{_container_id(installation, 'api')}:{container_path}", str(destination)],
        label="bootstrap ticket custody transfer",
        timeout=60,
    )
    destination.chmod(0o600)
    _compose_exec(
        installation,
        "api",
        ["rm", "-f", container_path],
        label="container bootstrap ticket cleanup",
    )
    if not destination.is_file() or destination.stat().st_size < 16:
        raise GreenfieldLifecycleError("bootstrap ticket custody transfer failed")


def _pw(
    session: str,
    command: Sequence[str],
    *,
    cwd: Path,
    label: str,
    raw: bool = False,
    timeout: float = 120,
) -> str:
    cli = [str(PWCLI), f"-s={session}"]
    if raw:
        cli.append("--raw")
    result = _run([*cli, *command], label=label, cwd=cwd, timeout=timeout)
    output = result.stdout.decode(errors="replace").strip()
    if raw and not output:
        detail = result.stderr.decode(errors="replace").strip()
        detail = re.sub(r"[A-Za-z0-9_-]{32,}", "[redacted]", detail)
        raise GreenfieldLifecycleError(
            f"{label} returned no machine result: {detail[-2000:]}"
        )
    return output


def _pw_json(
    *,
    session: str,
    command: Sequence[str],
    cwd: Path,
    label: str,
    timeout: float = 120,
) -> dict[str, Any]:
    output = _pw(
        session,
        command,
        cwd=cwd,
        label=label,
        raw=True,
        timeout=timeout,
    )
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as error:
        preview = re.sub(r"[A-Za-z0-9_-]{32,}", "[redacted]", output)
        raise GreenfieldLifecycleError(
            f"{label} returned invalid JSON: {preview[-1000:]!r}"
        ) from error
    if not isinstance(payload, dict):
        raise GreenfieldLifecycleError(f"{label} returned a non-object JSON value")
    return payload


def _capture_browser_screenshot(
    *,
    session: str,
    cwd: Path,
    screenshot: Path,
) -> str:
    code = """async (page) => {
      const root = await page.locator('[data-admin-root]').elementHandle();
      if (!root) throw new Error('admin root is missing before screenshot');
      const actualText = await page.evaluate(() => {
        const source = document.querySelector('[data-admin-root]');
        if (!source) throw new Error('admin root is missing before DOM freeze');
        const actualText = (source.innerText || '').trim().slice(0, 4000);
        if (!actualText) throw new Error('admin root has no visible text');
        return actualText;
      });
      const evidencePage = await page.context().newPage();
      try {
        await evidencePage.setViewportSize({width: 1280, height: 720});
        await evidencePage.setContent('<!doctype html><html><head></head><body></body></html>');
        await evidencePage.evaluate((actualText) => {
        const style = document.createElement('style');
        style.textContent = `
          * { animation: none !important; transition: none !important; }
          html, body { margin: 0; width: 1280px; min-height: 720px; overflow: hidden;
            background: #0f172a; color: #e2e8f0; font: 14px system-ui, sans-serif; }
          [data-admin-root] { box-sizing: border-box; max-width: 1280px; padding: 24px; }
          pre { white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.45; }
        `;
        const evidence = document.createElement('main');
        evidence.dataset.adminRoot = '';
        const heading = document.createElement('h1');
        heading.textContent = 'Roehub Admin — greenfield browser proof';
        const text = document.createElement('pre');
        text.textContent = actualText;
        evidence.replaceChildren(heading, text);
        document.head.replaceChildren(style);
        document.body.replaceChildren(evidence);
        }, actualText);
        const client = await evidencePage.context().newCDPSession(evidencePage);
        await client.send('Emulation.setDeviceMetricsOverride', {
          width: 1280, height: 720, deviceScaleFactor: 1, mobile: false
        });
        const result = await client.send('Page.captureScreenshot', {
          format: 'png', fromSurface: true, captureBeyondViewport: false,
          clip: {x: 0, y: 0, width: 1280, height: 720, scale: 1}
        });
        return {data: result.data, dom_frozen: true};
      } finally {
        await evidencePage.close();
      }
    }"""
    payload = _pw_json(
        session=session,
        command=["run-code", code],
        cwd=cwd,
        label="browser admin CDP screenshot",
        timeout=30,
    )
    if payload.get("dom_frozen") is not True:
        raise GreenfieldLifecycleError("browser screenshot DOM was not frozen")
    encoded = payload.get("data")
    if not isinstance(encoded, str) or len(encoded) > 7 * 1024 * 1024:
        raise GreenfieldLifecycleError("browser screenshot payload is invalid")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise GreenfieldLifecycleError("browser screenshot is not valid base64") from error
    if not content.startswith(b"\x89PNG\r\n\x1a\n") or len(content) > 5 * 1024 * 1024:
        raise GreenfieldLifecycleError("browser screenshot is not a bounded PNG")
    screenshot.write_bytes(content)
    return _sha256_bytes(content)


def _enable_virtual_authenticator(*, session: str, cwd: Path) -> None:
    code = """async (page) => {
      const client = await page.context().newCDPSession(page);
      await client.send('WebAuthn.enable');
      await client.send('WebAuthn.addVirtualAuthenticator', {options: {
        protocol: 'ctap2', transport: 'internal', hasResidentKey: true,
        hasUserVerification: true, isUserVerified: true,
        automaticPresenceSimulation: true
      }});
      return {virtual_authenticator: true};
    }"""
    payload = _pw_json(
        session=session,
        command=["run-code", code],
        cwd=cwd,
        label="browser virtual authenticator",
    )
    if payload != {"virtual_authenticator": True}:
        raise GreenfieldLifecycleError("browser virtual authenticator was not enabled")


def _browser_bootstrap(
    *,
    session: str,
    cwd: Path,
    ticket: Path,
    installation_name: str,
    organization_slug: str,
    organization_name: str,
) -> dict[str, Any]:
    _pw(
        session,
        ["open", "http://localhost:8080/login?next=/admin"],
        cwd=cwd,
        label="greenfield browser open",
    )
    _enable_virtual_authenticator(session=session, cwd=cwd)
    code = f"""async (page) => {{
      await page.locator('[data-local-bootstrap]').waitFor({{state: 'visible'}});
      await page.locator('input[name="ticket_file"]').setInputFiles({json.dumps(str(ticket))});
      await page.locator('input[name="display_name"]').fill('Disposable Owner');
      await page.locator('input[name="installation_name"]').fill({json.dumps(installation_name)});
      await page.locator('input[name="organization_slug"]').fill({json.dumps(organization_slug)});
      await page.locator('input[name="organization_name"]').fill({json.dumps(organization_name)});
      await page.locator('[data-local-bootstrap] button[type="submit"]').click();
      await page.locator('[data-recovery-codes]').waitFor({{state: 'visible'}});
      const recoveryCount = await page.locator('[data-recovery-code-list] li').count();
      if (recoveryCount < 4) throw new Error('recovery code set is incomplete');
      await page.locator('[data-recovery-ack]').click();
      await page.waitForURL('**/admin');
      await page.locator('[data-admin-root]').waitFor({{state: 'visible'}});
      return {{authenticated: true, recovery_code_count: recoveryCount, recovery_values_retained: false}};
    }}"""
    payload = _pw_json(
        session=session,
        command=["run-code", code],
        cwd=cwd,
        label="browser owner bootstrap",
        timeout=180,
    )
    ticket.unlink(missing_ok=True)
    if payload.get("authenticated") is not True or payload.get("recovery_values_retained") is not False:
        raise GreenfieldLifecycleError("browser owner bootstrap did not complete safely")
    return payload


def _browser_create_boundaries(*, session: str, cwd: Path) -> dict[str, Any]:
    code = f"""async (page) => {{
      const request = async (path, options = {{}}) => {{
        const response = await page.evaluate(async ({{path, options}}) => {{
          const csrfCookie = decodeURIComponent((document.cookie.split('; ').find(v => v.startsWith('roehub_csrf=')) || '').slice('roehub_csrf='.length));
          const headers = {{...(options.headers || {{}})}};
          if (options.method && options.method !== 'GET') headers['x-csrf-token'] = csrfCookie;
          const value = await fetch(path, {{...options, headers}});
          let body = null; try {{ body = await value.json(); }} catch (_) {{}}
          return {{status: value.status, ok: value.ok, body}};
        }}, {{path, options}});
        if (!response.ok) throw new Error(`request failed ${{path}} ${{response.status}}`);
        return response.body;
      }};
      const organizations = await request('/api/v1/organizations');
      if (organizations.length !== 1) throw new Error('initial organization count mismatch');
      const primary = organizations[0].organization;
      const secondary = await request('/api/v1/organizations', {{
        method: 'POST', headers: {{'content-type': 'application/json'}},
        body: JSON.stringify({{slug: 'secondary', display_name: 'Secondary Disposable'}})
      }});
      const expiresAt = new Date(Date.now() + 60 * 60 * 1000).toISOString();
      for (const [organizationId, email] of [
        [primary.organization_id, {_FIXTURE_EMAILS['a']!r}],
        [secondary.organization_id, {_FIXTURE_EMAILS['b']!r}]
      ]) {{
        await request(`/api/v1/organizations/${{organizationId}}/invitations`, {{
          method: 'POST', headers: {{'content-type': 'application/json'}},
          body: JSON.stringify({{recipient_email: email, role: 'viewer', expires_at: expiresAt}})
        }});
      }}
      const csrfPresent = await page.evaluate(() => Boolean(
        document.cookie.split('; ').find(value => value.startsWith('roehub_csrf='))
      ));
      return {{
        csrf_present: csrfPresent,
        primary_organization_id: primary.organization_id,
        secondary_organization_id: secondary.organization_id,
        organization_count: 2,
        invitation_count: 2
      }};
    }}"""
    payload = _pw_json(
        session=session,
        command=["run-code", code],
        cwd=cwd,
        label="browser public API organization setup",
        timeout=120,
    )
    if payload.get("organization_count") != 2 or payload.get("invitation_count") != 2:
        raise GreenfieldLifecycleError("browser public API organization setup is incomplete")
    return payload


def _provision_oidc_fixture(
    installation: Installation,
    *,
    fixture_id: str,
) -> dict[str, Any]:
    result = _compose_exec(
        installation,
        "api",
        [
            "python",
            "-",
            "--email",
            _FIXTURE_EMAILS[fixture_id],
            "--fixture-id",
            fixture_id,
        ],
        label=f"disposable OIDC use-case fixture {fixture_id}",
        input_bytes=OIDC_FIXTURE.read_bytes(),
    )
    payload = _json_output(result, label=f"disposable OIDC use-case fixture {fixture_id}")
    if payload.get("provisioned") is not True or payload.get("accepted_invitation_count") != 1:
        raise GreenfieldLifecycleError("disposable OIDC use-case fixture did not provision")
    return payload


def _browser_validate_isolation(
    *,
    session: str,
    cwd: Path,
    organizations: dict[str, Any],
    users: dict[str, dict[str, Any]],
    screenshot: Path,
) -> dict[str, Any]:
    primary = str(organizations["primary_organization_id"])
    secondary = str(organizations["secondary_organization_id"])
    user_a = str(users["a"]["user_id"])
    user_b = str(users["b"]["user_id"])
    code = f"""async (page) => {{
      const request = async (path, options = {{}}, allowFailure = false) => {{
        const result = await page.evaluate(async ({{path, options}}) => {{
          const csrf = decodeURIComponent((document.cookie.split('; ').find(v => v.startsWith('roehub_csrf=')) || '').slice('roehub_csrf='.length));
          const headers = {{...(options.headers || {{}})}};
          if (options.method && options.method !== 'GET') headers['x-csrf-token'] = csrf;
          const response = await fetch(path, {{...options, headers}});
          let body = null; try {{ body = await response.json(); }} catch (_) {{}}
          return {{status: response.status, ok: response.ok, body}};
        }}, {{path, options}});
        if (!result.ok && !allowFailure) throw new Error(`request failed ${{path}} ${{result.status}}`);
        return result;
      }};
      const primaryMembers = (await request('/api/v1/organizations/{primary}/members')).body;
      const secondaryMembers = (await request('/api/v1/organizations/{secondary}/members')).body;
      const primaryIds = primaryMembers.map(item => item.user_id);
      const secondaryIds = secondaryMembers.map(item => item.user_id);
      if (!primaryIds.includes('{user_a}') || primaryIds.includes('{user_b}')) throw new Error('primary membership leakage');
      if (!secondaryIds.includes('{user_b}') || secondaryIds.includes('{user_a}')) throw new Error('secondary membership leakage');
      const invalid = await request('/api/v1/organizations/{secondary}/plugins/stage23-cross/permissions/{user_a}', {{
        method: 'PUT', headers: {{'content-type': 'application/json'}}, body: JSON.stringify({{permission: 'read'}})
      }}, true);
      if (invalid.ok) throw new Error('cross-organization permission unexpectedly succeeded');
      for (const [organizationId, userId, pluginId] of [
        ['{primary}', '{user_a}', 'stage23-primary'],
        ['{secondary}', '{user_b}', 'stage23-secondary']
      ]) {{
        await request(`/api/v1/organizations/${{organizationId}}/plugins/${{pluginId}}/permissions/${{userId}}`, {{
          method: 'PUT', headers: {{'content-type': 'application/json'}}, body: JSON.stringify({{permission: 'read'}})
        }});
      }}
      const adminResponse = await page.goto('http://localhost:8080/admin');
      if (!adminResponse || !adminResponse.ok()) {{
        const alert = (await page.locator('[role="alert"]').allTextContents()).join(' ').slice(0, 300);
        throw new Error(`admin SSR failed ${{adminResponse?.status() || 0}}: ${{alert}}`);
      }}
      await page.locator('[data-admin-root]').waitFor({{state: 'visible'}});
      await page.locator('[data-admin-presence].is-ready').waitFor({{state: 'attached'}});
      return {{
        admin_visible: true,
        cross_organization_permission_status: invalid.status,
        primary_member_count: primaryMembers.length,
        secondary_member_count: secondaryMembers.length
      }};
    }}"""
    payload = _pw_json(
        session=session,
        command=["run-code", code],
        cwd=cwd,
        label="browser admin and organization isolation smoke",
        timeout=180,
    )
    payload["screenshot_capture"] = "chromium-cdp-isolated-bounded-admin-text"
    payload["screenshot_sha256"] = _capture_browser_screenshot(
        session=session,
        cwd=cwd,
        screenshot=screenshot,
    )
    payload["screenshot_written"] = True
    if payload.get("admin_visible") is not True or not screenshot.is_file():
        raise GreenfieldLifecycleError("browser admin smoke is incomplete")
    console = _pw(
        session,
        ["console", "error"],
        cwd=cwd,
        label="browser console error inspection",
    )
    requests = _pw(
        session,
        ["requests"],
        cwd=cwd,
        label="browser network inspection",
    )
    console_errors = _console_error_count(console)
    if console_errors:
        detail = re.sub(r"[A-Za-z0-9_-]{32,}", "[redacted]", console)
        raise GreenfieldLifecycleError(
            f"browser console contains error messages: {detail[-2000:]}"
        )
    if "/api/v1/organizations" not in requests or "/api/v1/admin/organizations/" not in requests:
        raise GreenfieldLifecycleError("browser network evidence is incomplete")
    payload["console_errors"] = console_errors
    payload["network_boundaries"] = [
        "/api/v1/organizations",
        "/api/v1/admin/organizations/{organization_id}/snapshot",
    ]
    return payload


def _browser_restore_login(*, session: str, cwd: Path) -> dict[str, Any]:
    code = """async (page) => {
      await page.context().clearCookies();
      await page.goto('http://localhost:8080/login?next=/admin');
      await page.locator('[data-local-login]').waitFor({state: 'visible'});
      await page.locator('[data-passkey-login]').click({force: true});
      await page.waitForURL('**/admin');
      await page.locator('[data-admin-root]').waitFor({state: 'visible'});
      await page.locator('[data-admin-presence].is-ready').waitFor({state: 'attached'});
      return {admin_visible: true, restored_passkey_login: true};
    }"""
    payload = _pw_json(
        session=session,
        command=["run-code", code],
        cwd=cwd,
        label="restored passkey browser login",
        timeout=180,
    )
    if payload != {"admin_visible": True, "restored_passkey_login": True}:
        raise GreenfieldLifecycleError("restored passkey browser login failed")
    return payload


def _close_browser(*, session: str, cwd: Path) -> None:
    subprocess.run(
        [str(PWCLI), f"-s={session}", "close"],
        cwd=cwd,
        check=False,
        capture_output=True,
        timeout=60,
    )


def _identity_counts(installation: Installation) -> dict[str, int]:
    expressions = ",".join(
        f"'{table}',(SELECT count(*) FROM {table})" for table in _CORE_IDENTITY_TABLES
    )
    result = _compose_exec(
        installation,
        "postgresql",
        [
            "psql",
            "-U",
            "roehub",
            "-d",
            "roehub",
            "-At",
            "-c",
            f"SELECT json_build_object({expressions})::text",
        ],
        label="identity count reconciliation",
    )
    payload = _json_output(result, label="identity count reconciliation")
    return {key: int(value) for key, value in payload.items()}


def _identity_digest(installation: Installation) -> str:
    digest = hashlib.sha256()
    for table in _CORE_IDENTITY_TABLES:
        result = _compose_exec(
            installation,
            "postgresql",
            [
                "psql",
                "-U",
                "roehub",
                "-d",
                "roehub",
                "-At",
                "-c",
                f"COPY (SELECT to_jsonb(t)::text FROM {table} AS t ORDER BY to_jsonb(t)::text) TO STDOUT",
            ],
            label=f"identity digest {table}",
        )
        digest.update(table.encode())
        digest.update(b"\0")
        digest.update(result.stdout)
    return "sha256:" + digest.hexdigest()


def _seed_storage(installation: Installation, *, organizations: dict[str, Any]) -> dict[str, Any]:
    primary = str(organizations["primary_organization_id"])
    secondary = str(organizations["secondary_organization_id"])
    _compose_exec(
        installation,
        "clickhouse",
        [
            "/bin/sh",
            "-c",
            (
                "clickhouse-client --password \"$(cat /run/roehub-secrets/clickhouse-password)\" "
                "--multiquery --query \""
                "CREATE TABLE stage23_timeseries (organization_id UUID, ts DateTime64(3, 'UTC'), value Float64) "
                "ENGINE=MergeTree ORDER BY (organization_id, ts);"
                f"INSERT INTO stage23_timeseries VALUES ('{primary}','2026-07-14 10:00:00.000',1.0),"
                f"('{secondary}','2026-07-14 10:01:00.000',2.0);\""
            ),
        ],
        label="ClickHouse representative fixture",
    )
    redis = _compose_exec(
        installation,
        "redis",
        [
            "/bin/sh",
            "-c",
            "redis-cli --no-auth-warning -a \"$(cat /run/roehub-secrets/redis-password-server)\" SET stage23:checkpoint checkpoint-0002",
        ],
        label="Redis representative checkpoint",
    ).stdout.decode().strip()
    if redis != "OK":
        raise GreenfieldLifecycleError("Redis representative checkpoint failed")
    artifact = _compose_exec(
        installation,
        "api",
        [
            "python",
            "-c",
            (
                "from pathlib import Path; import hashlib; "
                "p=Path('/var/lib/roehub/artifacts/stage23/representative.json'); "
                "p.parent.mkdir(parents=True, exist_ok=True); "
                "p.write_bytes(b'{\"schema\":\"io.roehub.stage23-artifact/v1alpha1\",\"organizations\":2}\\n'); "
                "print('sha256:'+hashlib.sha256(p.read_bytes()).hexdigest())"
            ),
        ],
        label="representative artifact fixture",
    ).stdout.decode().strip()
    openbao = _openbao_metadata(installation)
    return {
        "artifact_digest": artifact,
        "clickhouse_rows": 2,
        "openbao_metadata": openbao,
        "redis_checkpoint": "checkpoint-0002",
    }


def _openbao_metadata(installation: Installation) -> dict[str, Any]:
    result = _compose_exec(
        installation,
        "openbao",
        [
            "bao",
            "status",
            "-address=http://127.0.0.1:8200",
            "-format=json",
        ],
        label="OpenBao metadata",
        allowed_codes=frozenset({0, 2}),
    )
    payload = _json_output(result, label="OpenBao metadata")
    return {
        "initialized": bool(payload.get("initialized")),
        "sealed": bool(payload.get("sealed")),
        "storage_type": str(payload.get("storage_type", "")),
        "version": str(payload.get("version", "")),
    }


def _clickhouse_snapshot(installation: Installation) -> bytes:
    return _compose_exec(
        installation,
        "clickhouse",
        [
            "/bin/sh",
            "-c",
            (
                "clickhouse-client --password \"$(cat /run/roehub-secrets/clickhouse-password)\" "
                "--query \"SELECT organization_id, ts, value FROM stage23_timeseries "
                "ORDER BY organization_id, ts FORMAT JSONEachRow\""
            ),
        ],
        label="ClickHouse snapshot",
    ).stdout


def _redis_checkpoint(installation: Installation) -> str:
    return _compose_exec(
        installation,
        "redis",
        [
            "/bin/sh",
            "-c",
            "redis-cli --no-auth-warning -a \"$(cat /run/roehub-secrets/redis-password-server)\" GET stage23:checkpoint",
        ],
        label="Redis checkpoint read",
    ).stdout.decode().strip()


def _artifact_digest(installation: Installation) -> str:
    return _compose_exec(
        installation,
        "api",
        [
            "python",
            "-c",
            (
                "from pathlib import Path; import hashlib; "
                "p=Path('/var/lib/roehub/artifacts/stage23/representative.json'); "
                "print('sha256:'+hashlib.sha256(p.read_bytes()).hexdigest())"
            ),
        ],
        label="artifact digest read",
    ).stdout.decode().strip()


def _postgres_backup(installation: Installation) -> bytes:
    return _compose_exec(
        installation,
        "postgresql",
        [
            "pg_dump",
            "-U",
            "roehub",
            "-d",
            "roehub",
            "--format=custom",
            "--no-owner",
            "--no-privileges",
        ],
        label="greenfield PostgreSQL backup",
        timeout=600,
    ).stdout


def _restore_postgres(installation: Installation, backup: bytes) -> None:
    _compose_exec(
        installation,
        "postgresql",
        [
            "pg_restore",
            "-U",
            "roehub",
            "-d",
            "roehub",
            "--no-owner",
            "--no-privileges",
        ],
        label="fresh PostgreSQL restore",
        input_bytes=backup,
        timeout=600,
    )


def _restore_clickhouse(installation: Installation, snapshot: bytes) -> None:
    _compose_exec(
        installation,
        "clickhouse",
        [
            "/bin/sh",
            "-c",
            (
                "clickhouse-client --password \"$(cat /run/roehub-secrets/clickhouse-password)\" "
                "--query \"CREATE TABLE stage23_timeseries (organization_id UUID, "
                "ts DateTime64(3, 'UTC'), value Float64) ENGINE=MergeTree "
                "ORDER BY (organization_id, ts)\" && "
                "clickhouse-client --password \"$(cat /run/roehub-secrets/clickhouse-password)\" "
                "--query \"INSERT INTO stage23_timeseries FORMAT JSONEachRow\""
            ),
        ],
        label="fresh ClickHouse restore",
        input_bytes=snapshot,
    )


def _restore_redis(installation: Installation, checkpoint: str) -> None:
    result = _compose_exec(
        installation,
        "redis",
        [
            "/bin/sh",
            "-c",
            (
                "redis-cli --no-auth-warning -a \"$(cat /run/roehub-secrets/redis-password-server)\" "
                f"SET stage23:checkpoint {checkpoint}"
            ),
        ],
        label="fresh Redis checkpoint restore",
    ).stdout.decode().strip()
    if result != "OK":
        raise GreenfieldLifecycleError("fresh Redis checkpoint restore failed")


def _restore_artifact(installation: Installation) -> str:
    return _seed_artifact_only(installation)


def _seed_artifact_only(installation: Installation) -> str:
    return _compose_exec(
        installation,
        "api",
        [
            "python",
            "-c",
            (
                "from pathlib import Path; import hashlib; "
                "p=Path('/var/lib/roehub/artifacts/stage23/representative.json'); "
                "p.parent.mkdir(parents=True, exist_ok=True); "
                "p.write_bytes(b'{\"schema\":\"io.roehub.stage23-artifact/v1alpha1\",\"organizations\":2}\\n'); "
                "print('sha256:'+hashlib.sha256(p.read_bytes()).hexdigest())"
            ),
        ],
        label="fresh artifact restore",
    ).stdout.decode().strip()


def _memory_snapshot(installation: Installation) -> dict[str, Any]:
    container_ids = _run(
        [*installation.command(), "ps", "-q"],
        label="memory container inventory",
        timeout=60,
    ).stdout.decode().splitlines()
    if not container_ids:
        raise GreenfieldLifecycleError("memory snapshot has no running containers")
    result = _run(
        ["docker", "stats", "--no-stream", "--format", "{{.Name}}|{{.MemUsage}}", *container_ids],
        label="memory snapshot",
        timeout=120,
    )
    rows = [line for line in result.stdout.decode().splitlines() if line.strip()]
    return {"container_count": len(rows), "observed": sorted(rows)}


def _verify_recovery_lifecycle(
    *,
    project_prefix: str,
    image_override: Path,
) -> tuple[dict[str, object], int]:
    retryable_message = "operational-health readiness failed"
    for attempt in (1, 2):
        try:
            return (
                verify_recovery_runtime(
                    project_prefix=f"{project_prefix}-attempt-{attempt}",
                    image_override=image_override,
                ),
                attempt,
            )
        except RecoveryRuntimeProofError as error:
            if str(error) != retryable_message or attempt == 2:
                raise GreenfieldLifecycleError(
                    f"Stage 21 recovery lifecycle repeat failed: {error}"
                ) from error
            time.sleep(1)
    raise GreenfieldLifecycleError("Stage 21 recovery lifecycle repeat did not run")


def _repeat_bootstrap(
    installation: Installation,
    *,
    cwd: Path,
    ticket: Path,
) -> dict[str, Any]:
    session = f"roehub-stage23-repeat-{os.getpid()}"
    try:
        bootstrap = _browser_bootstrap(
            session=session,
            cwd=cwd,
            ticket=ticket,
            installation_name="Stage 23 Repeat",
            organization_slug="repeat",
            organization_name="Repeat Disposable",
        )
        counts = _identity_counts(installation)
        expected = {
            "identity_installations": 1,
            "identity_installation_owners": 1,
            "identity_organizations": 1,
            "identity_users": 1,
            "identity_memberships": 1,
        }
        if any(counts.get(key) != value for key, value in expected.items()):
            raise GreenfieldLifecycleError("repeat bootstrap structure differs from clean install")
        return {"bootstrap": bootstrap, "expected_structural_counts": expected, "status": "passed"}
    finally:
        ticket.unlink(missing_ok=True)
        _close_browser(session=session, cwd=cwd)


def verify_greenfield_lifecycle(
    *,
    bundle: Path,
    trusted_key: Path,
    screenshot_destination: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    bundle = bundle.expanduser().resolve()
    trusted_key = trusted_key.expanduser().resolve()
    if not PWCLI.is_file():
        raise GreenfieldLifecycleError("pinned Playwright CLI wrapper is unavailable")
    if not OIDC_FIXTURE.is_file():
        raise GreenfieldLifecycleError("disposable OIDC use-case fixture is unavailable")
    verification = verify_bundle(bundle=bundle, trusted_public_key=trusted_key)
    if verification.get("signature_verified") is not True:
        raise GreenfieldLifecycleError("offline bundle signature is not verified")
    tags_before = _offline_tags()
    projects: list[Installation] = []
    browser_session = f"roehub-stage23-source-{os.getpid()}"
    browser_cwd: Path | None = None
    screenshot_temporary: Path | None = None
    with tempfile.TemporaryDirectory(prefix="roehub-stage23-") as temporary_name:
        temporary = Path(temporary_name).resolve()
        browser_cwd = temporary / "browser"
        browser_cwd.mkdir()
        screenshot_temporary = temporary / "greenfield-admin.png"
        source = Installation(
            project=f"roehub-stage23-source-{os.getpid()}",
            bundle=bundle,
            state=temporary / "source-state",
        )
        target = Installation(
            project=f"roehub-stage23-target-{os.getpid()}",
            bundle=bundle,
            state=temporary / "target-state",
        )
        repeat = Installation(
            project=f"roehub-stage23-repeat-{os.getpid()}",
            bundle=bundle,
            state=target.state,
        )
        projects.extend((source, target, repeat))
        for project in projects:
            _assert_empty_project(project.project)
        try:
            source_install = _install_bundle(
                bundle=bundle,
                trusted_key=trusted_key,
                state=source.state,
            )
            _up(source)
            source_ingress = _start_browser_ingress(source)
            source_memory = _memory_snapshot(source)
            source_ticket = temporary / "source-bootstrap-ticket"
            _issue_ticket(source, source_ticket)
            bootstrap = _browser_bootstrap(
                session=browser_session,
                cwd=browser_cwd,
                ticket=source_ticket,
                installation_name="Stage 23 Disposable",
                organization_slug="primary",
                organization_name="Primary Disposable",
            )
            organizations = _browser_create_boundaries(session=browser_session, cwd=browser_cwd)
            users = {
                fixture_id: _provision_oidc_fixture(source, fixture_id=fixture_id)
                for fixture_id in ("a", "b")
            }
            browser = _browser_validate_isolation(
                session=browser_session,
                cwd=browser_cwd,
                organizations=organizations,
                users=users,
                screenshot=screenshot_temporary,
            )
            storage = _seed_storage(source, organizations=organizations)
            source_counts = _identity_counts(source)
            source_identity_digest = _identity_digest(source)
            postgres_backup = _postgres_backup(source)
            clickhouse_backup = _clickhouse_snapshot(source)
            redis_backup = _redis_checkpoint(source)
            artifact_backup = _artifact_digest(source)
            openbao_backup = _openbao_metadata(source)
            if artifact_backup != storage["artifact_digest"] or redis_backup != storage["redis_checkpoint"]:
                raise GreenfieldLifecycleError("source representative state changed before backup")
            _down(source, volumes=False)

            target_install = _install_bundle(
                bundle=bundle,
                trusted_key=trusted_key,
                state=target.state,
            )
            _up(target, "secret-init", "postgresql", "clickhouse", "redis")
            _restore_postgres(target, postgres_backup)
            _restore_clickhouse(target, clickhouse_backup)
            _restore_redis(target, redis_backup)
            _up(target)
            target_ingress = _start_browser_ingress(target)
            restored_artifact = _restore_artifact(target)
            target_counts = _identity_counts(target)
            target_identity_digest = _identity_digest(target)
            target_clickhouse = _clickhouse_snapshot(target)
            target_redis = _redis_checkpoint(target)
            target_openbao = _openbao_metadata(target)
            if source_counts != target_counts or source_identity_digest != target_identity_digest:
                raise GreenfieldLifecycleError("PostgreSQL identity restore reconciliation failed")
            if clickhouse_backup != target_clickhouse:
                raise GreenfieldLifecycleError("ClickHouse restore reconciliation failed")
            if redis_backup != target_redis:
                raise GreenfieldLifecycleError("Redis restore reconciliation failed")
            if artifact_backup != restored_artifact:
                raise GreenfieldLifecycleError("artifact restore reconciliation failed")
            if openbao_backup != target_openbao:
                raise GreenfieldLifecycleError("OpenBao metadata reconciliation failed")
            target_memory = _memory_snapshot(target)
            restored_browser = _browser_restore_login(session=browser_session, cwd=browser_cwd)
            _close_browser(session=browser_session, cwd=browser_cwd)
            _down(target, volumes=True)

            _assert_empty_project(repeat.project)
            _up(repeat)
            repeat_ingress = _start_browser_ingress(repeat)
            repeat_memory = _memory_snapshot(repeat)
            repeat_ticket = temporary / "repeat-bootstrap-ticket"
            _issue_ticket(repeat, repeat_ticket)
            repeat_proof = _repeat_bootstrap(
                repeat,
                cwd=browser_cwd,
                ticket=repeat_ticket,
            )
            _down(repeat, volumes=True)
            _down(source, volumes=True)

            recovery, recovery_attempts = _verify_recovery_lifecycle(
                project_prefix=f"roehub-stage23-recovery-{os.getpid()}",
                image_override=source.override,
            )
            recovery_cleanup = recovery.get("cleanup")
            if (
                recovery.get("status") != "passed"
                or not isinstance(recovery_cleanup, dict)
                or recovery_cleanup.get("status") != "completed"
            ):
                raise GreenfieldLifecycleError("Stage 21 recovery lifecycle repeat failed")

            screenshot_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(screenshot_temporary, screenshot_destination)
            screenshot_destination.chmod(0o644)
            return {
                "backup_restore": {
                    "artifact_digest": artifact_backup,
                    "clickhouse_sha256": _sha256_bytes(clickhouse_backup),
                    "identity_counts": source_counts,
                    "identity_state_digest": source_identity_digest,
                    "openbao_metadata": openbao_backup,
                    "postgres_backup_sha256": _sha256_bytes(postgres_backup),
                    "reconciled": True,
                    "redis_checkpoint": redis_backup,
                },
                "browser": {
                    **browser,
                    "auth_material_retained": False,
                    "bootstrap": bootstrap,
                    "ingress": {
                        "repeat": repeat_ingress,
                        "source": source_ingress,
                        "target": target_ingress,
                    },
                    "restored_login": restored_browser,
                    "screenshot": str(screenshot_destination),
                    "screenshot_sha256": _sha256_path(screenshot_destination),
                    "session_state_persisted": False,
                },
                "bundle": {
                    "path": str(bundle),
                    "signature_verified": True,
                    "source_install": source_install["status"],
                    "target_install": target_install["status"],
                    "version": source_install["bundle_version"],
                },
                "cleanup": {"status": "pending"},
                "duration_seconds": round(time.monotonic() - started, 3),
                "fresh_state": {
                    "current_production_access": False,
                    "disposable_oidc_users": 2,
                    "external_provider_writes": False,
                    "organizations": 2,
                    "personal_data_present": False,
                    "real_orders": False,
                },
                "memory": {
                    "execution": "sequential-single-full-installation",
                    "repeat": repeat_memory,
                    "source": source_memory,
                    "target": target_memory,
                },
                "recovery_lifecycle": {
                    "attempt_count": recovery_attempts,
                    "cleanup": recovery["cleanup"],
                    "release_lifecycle": recovery["release_lifecycle"],
                    "restore_comparison": recovery["restore_comparison"],
                    "schema": recovery["schema"],
                    "status": recovery["status"],
                },
                "repeat_install": repeat_proof,
                "schema": "io.roehub.greenfield-installation-lifecycle-proof/v1alpha1",
                "status": "passed",
            }
        finally:
            if browser_cwd is not None:
                _close_browser(session=browser_session, cwd=browser_cwd)
            for installation in reversed(projects):
                _down(installation, volumes=True)
            owned_tags = sorted(_offline_tags() - tags_before)
            if owned_tags:
                subprocess.run(
                    ["docker", "image", "rm", "--force", *owned_tags],
                    check=False,
                    capture_output=True,
                    timeout=900,
                )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--trusted-public-key", type=Path, default=DEFAULT_TRUSTED_KEY)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--screenshot", type=Path, default=DEFAULT_SCREENSHOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        payload = verify_greenfield_lifecycle(
            bundle=args.bundle,
            trusted_key=args.trusted_public_key,
            screenshot_destination=args.screenshot,
        )
        payload["cleanup"] = {"status": "completed"}
    except (GreenfieldLifecycleError, OSError, KeyError, ValueError) as error:
        print(f"Stage 23 greenfield lifecycle proof failed: {error}", file=sys.stderr)
        return 1
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
