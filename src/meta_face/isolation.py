"""Turn native process crashes into recorded failures instead of core dumps."""

from __future__ import annotations

import faulthandler
import os
import resource
import signal


def disable_core_dumps() -> None:
    """Prevent SIGSEGV/SIGABRT from writing core files."""
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):
        pass


def enable_crash_tracebacks() -> None:
    """Dump Python frames on fatal signals, then let the process die."""
    faulthandler.enable(all_threads=True)


def prepare_isolated_process() -> None:
    """Apply crash policy in a worker parent or forked job process."""
    disable_core_dumps()
    enable_crash_tracebacks()


def crash_message(wait_status: int | None) -> str:
    """Human-readable failure for a child that exited or died on a signal."""
    if wait_status is None:
        return "Job process terminated unexpectedly."
    if os.WIFSIGNALED(wait_status):
        signum = os.WTERMSIG(wait_status)
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = f"signal {signum}"
        return (
            f"Job process aborted with {name} ({signum}). "
            "Native crash; the worker stayed up and marked this job failed."
        )
    if os.WIFEXITED(wait_status):
        code = os.WEXITSTATUS(wait_status)
        return f"Job process exited with status {code}."
    return f"Job process terminated unexpectedly (wait status {wait_status})."


def crash_message_from_exitcode(exitcode: int | None) -> str:
    """Same message from multiprocessing.Process.exitcode (negative if signaled)."""
    if exitcode is None:
        return "Job process failed to start."
    if exitcode < 0:
        # os.waitpid status for signal N is N; multiprocessing uses -N.
        return crash_message(-exitcode)
    if exitcode == 0:
        return "Job process exited with status 0."
    return f"Job process exited with status {exitcode}."
