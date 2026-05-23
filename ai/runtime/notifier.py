"""Windows toast notification surface. Zero new dependencies."""

from __future__ import annotations

import base64
import subprocess
import sys


def _xml_escape(s: str) -> str:
    return (
        str(s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'@", "' @")  # prevent accidental here-string terminator
    )


def toast(title: str, body: str) -> None:
    """Fire a Windows WinRT toast. No-op on non-Windows. Fire-and-forget."""
    if sys.platform != "win32":
        return
    t = _xml_escape(title)[:64]
    b = _xml_escape(body)[:200]
    xml = (
        f'<toast><visual><binding template="ToastGeneric">'
        f"<text>{t}</text><text>{b}</text>"
        f"</binding></visual></toast>"
    )
    # Single-quoted PS here-string is fully literal — no escaping needed for content.
    script = (
        "[Windows.Data.Xml.Dom.XmlDocument,Windows.Data.Xml.Dom.Document,ContentType=WindowsRuntime]|Out-Null\n"
        "$xml=@'\n"
        f"{xml}\n"
        "'@\n"
        "$x=[Windows.Data.Xml.Dom.XmlDocument]::new()\n"
        "$x.LoadXml($xml)\n"
        "$n=[Windows.UI.Notifications.ToastNotification,Windows.UI.Notifications,ContentType=WindowsRuntime]::new($x)\n"
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('ALICE').Show($n)\n"
    )
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    subprocess.Popen(
        [
            "powershell",
            "-WindowStyle",
            "Hidden",
            "-NonInteractive",
            "-EncodedCommand",
            encoded,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=0x08000000,  # CREATE_NO_WINDOW
    )


def heartbeat_output(text: str) -> None:
    """Drop-in for Heartbeat.output — prints to stdout and fires a toast."""
    print(text, file=sys.stdout, flush=True)
    msg = str(text or "").removeprefix("\nA.L.I.C.E: ").strip()
    if msg:
        toast("A.L.I.C.E", msg)


def daemon_notify(message: str, *, priority: str = "normal") -> None:
    """Drop-in for CompanionDaemon.notify_callback."""
    toast("A.L.I.C.E", str(message or ""))
