"""Phase 15 — forward paper/shadow infrastructure for the frozen bos_5m_trigger.

This package runs the EXACT frozen Phase-14 candidate forward in paper/shadow mode to
collect new, unseen trades. It deliberately has **no live order execution** — there is
no broker/order API anywhere in this package; it only ingests market data and simulates
fills with the same (locked-buffer Topstep) backtest semantics. The strategy is loaded
from a hash-verified frozen registry and refuses to run if its config was modified.
"""
