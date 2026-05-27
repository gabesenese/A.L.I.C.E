"""
Desktop notification system for A.L.I.C.E — Windows toast alerts.

Sends native Windows toast notifications so ALICE can surface proactive
alerts (imminent meetings, stale goals, system health) even when the
terminal is minimized. No external dependencies: uses PowerShell's
built-in Windows Runtime bridge.

Falls back silently on non-Windows platforms or if PowerShell is unavailable.
"""

from __future__ import annotations

import logging
import platform
import subprocess

logger = logging.getLogger(__name__)

_IS_WINDOWS = platform.system() == "Windows"

# PowerShell script template — WinRT toast via .NET/WinRT bridge
_TOAST_PS = """
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null

$appId = 'A.L.I.C.E'
$title = {title!r}
$body  = {body!r}

$xml = @"
<toast>
  <visual>
    <binding template='ToastGeneric'>
      <text>{{}}</text>
      <text>{{}}</text>
    </binding>
  </visual>
</toast>
"@

$xml = $xml -f $title, $body
$doc = [Windows.Data.Xml.Dom.XmlDocument]::new()
$doc.LoadXml($xml)
$toast = [Windows.UI.Notifications.ToastNotification]::new($doc)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($appId).Show($toast)
"""


def send_notification(title: str, body: str, timeout_ms: int = 5000) -> bool:
    """Send a Windows toast notification. Returns True if sent successfully."""
    if not _IS_WINDOWS:
        logger.debug("[Notifier] non-Windows platform — skipping toast")
        return False

    title = str(title or "A.L.I.C.E").strip()[:64]
    body = str(body or "").strip()[:256]

    script = _TOAST_PS.format(title=title, body=body)
    try:
        result = subprocess.run(
            ["powershell", "-NonInteractive", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if result.returncode != 0:
            logger.debug("[Notifier] toast failed: %s", result.stderr.strip())
            return False
        return True
    except Exception as exc:
        logger.debug("[Notifier] toast error: %s", exc)
        return False


def notify_meeting(title: str, minutes: int) -> bool:
    return send_notification(
        title="Upcoming meeting",
        body=f"{title!r} starts in {minutes} minute{'s' if minutes != 1 else ''}.",
    )


def notify_proactive(message: str) -> bool:
    return send_notification(title="A.L.I.C.E", body=message)


def notify_system_alert(message: str) -> bool:
    return send_notification(title="System Alert", body=message)
