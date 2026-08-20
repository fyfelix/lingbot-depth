#!/usr/bin/env python3

from __future__ import annotations

import os
import platform
import plistlib
import subprocess
from typing import Any

INTEL_VENDOR_ID = 0x8086
REALSENSE_PRODUCT_IDS = {0x0B07, 0x0B3A, 0x0B5C}


def _realsense_usb_interfaces() -> list[dict[str, Any]]:
    if platform.system() != "Darwin":
        return []

    result = subprocess.run(
        ["ioreg", "-a", "-r", "-c", "IOUSBHostInterface"],
        check=True,
        capture_output=True,
    )
    interfaces = plistlib.loads(result.stdout)
    return [
        interface
        for interface in interfaces
        if interface.get("idVendor") == INTEL_VENDOR_ID
        and interface.get("idProduct") in REALSENSE_PRODUCT_IDS
    ]


def _print_usb_owners() -> bool:
    try:
        interfaces = _realsense_usb_interfaces()
    except (OSError, subprocess.CalledProcessError, plistlib.InvalidFileException) as exc:
        print(f"USB ownership check failed: {exc}")
        return False

    if not interfaces:
        print("USB interfaces: no supported RealSense D4xx interface found")
        return False

    print("USB interfaces:")
    uvc_assistant_owns_interface = False
    for interface in sorted(interfaces, key=lambda item: item.get("bInterfaceNumber", -1)):
        number = interface.get("bInterfaceNumber", "?")
        name = interface.get("IORegistryEntryName", "unknown")
        owner = interface.get("UsbExclusiveOwner", "unclaimed")
        print(f"  interface {number}: {name}; owner={owner}")
        uvc_assistant_owns_interface |= "UVCAssistant" in str(owner)
    return uvc_assistant_owns_interface


def main() -> int:
    print(f"macOS: {platform.mac_ver()[0] or 'unknown'}")
    print(f"Python: {platform.python_version()} ({platform.machine()})")
    print(f"euid: {os.geteuid()} ({'elevated' if os.geteuid() == 0 else 'standard user'})")

    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        print(f"pyrealsense2 import: FAILED ({exc})")
        return 2

    print(f"pyrealsense2: {rs.__version__} ({rs.__file__})")
    uvc_assistant_owns_interface = _print_usb_owners()

    try:
        context = rs.context()
        devices = context.query_devices()
        print(f"SDK device count: {len(devices)}")
        if not devices:
            return 3

        for device in devices:
            name = device.get_info(rs.camera_info.name)
            serial = device.get_info(rs.camera_info.serial_number)
            print(f"Device ready: {name} (serial {serial})")
    except RuntimeError as exc:
        print(f"SDK device access: FAILED ({exc})")
        if platform.system() == "Darwin" and os.geteuid() != 0:
            print(
                "macOS 12+ requires elevated privileges for librealsense USB access. "
                "Re-run this checker with sudo -H using the same virtual-environment Python."
            )
        elif uvc_assistant_owns_interface:
            print("macOS UVCAssistant still owns one or more RealSense UVC interfaces.")
        return 4

    print("RealSense SDK access is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
