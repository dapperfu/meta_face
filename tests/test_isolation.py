"""Tests for native-crash isolation helpers."""

from __future__ import annotations

import os
import resource
import signal

from meta_face.isolation import (
    crash_message,
    crash_message_from_exitcode,
    disable_core_dumps,
)


def test_disable_core_dumps_sets_soft_limit_to_zero() -> None:
    disable_core_dumps()
    soft, _hard = resource.getrlimit(resource.RLIMIT_CORE)
    assert soft == 0


def test_crash_message_for_sigsegv_wait_status() -> None:
    status = signal.SIGSEGV  # waitpid status for an uncaught SIGSEGV
    assert os.WIFSIGNALED(status)
    assert os.WTERMSIG(status) == signal.SIGSEGV
    message = crash_message(status)
    assert "SIGSEGV" in message
    assert "worker stayed up" in message


def test_crash_message_from_multiprocessing_exitcode() -> None:
    message = crash_message_from_exitcode(-signal.SIGSEGV)
    assert "SIGSEGV" in message
    assert crash_message_from_exitcode(2) == "Job process exited with status 2."
    assert "failed to start" in crash_message_from_exitcode(None)
