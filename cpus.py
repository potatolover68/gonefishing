"""How many CPUs this process is allowed to use."""

import os
from pathlib import Path


def cpus_from_quota(text: str) -> int | None:
    parts = text.split()
    if len(parts) != 2 or parts[0] == "max":
        return None
    quota, period = int(parts[0]), int(parts[1])
    if period <= 0:
        return None
    return max(1, round(quota / period))


def provisioned_cpus() -> int:
    """CPU limit from the container cgroup, otherwise the machine's CPU count."""
    for path in (
        Path("/sys/fs/cgroup/cpu.max"),
        Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
    ):
        if not path.is_file():
            continue
        if path.name == "cpu.max":
            parsed = cpus_from_quota(path.read_text(encoding="utf-8"))
            if parsed is not None:
                return parsed
            continue
        period_path = path.with_name("cpu.cfs_period_us")
        if not period_path.is_file():
            continue
        quota = int(path.read_text(encoding="utf-8").strip())
        if quota < 0:
            break
        period = int(period_path.read_text(encoding="utf-8").strip())
        if period > 0:
            return max(1, round(quota / period))
    return os.cpu_count() or 1


def compute_cpus() -> int:
    """Cores for a comparison. One core stays free for the web process when possible."""
    total = provisioned_cpus()
    if total <= 1:
        return 1
    return total - 1
