"""Typed, validated access to the YAML configs.

Everything is loaded into frozen dataclasses so nothing can silently mutate
mid-run. Loud failures over silent defaults: missing keys raise ConfigError.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


class ConfigError(Exception):
    pass


def _req(d: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise ConfigError(f"missing key '{key}' in {ctx}")
    return d[key]


def _parse_ct(hhmm: str) -> time:
    h, m = hhmm.split(":")
    return time(int(h), int(m))


def _parse_date(x: Any) -> date:
    if isinstance(x, date):
        return x
    return date.fromisoformat(str(x))


# --------------------------------------------------------------------------- #
# dataclasses
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    name: str
    exchange: str
    currency: str
    tick_size: float
    tick_value: float
    point_value: float
    micro: Optional[str] = None
    micro_of: Optional[str] = None
    micros_per_mini: Optional[int] = None
    spread_symbol_marker: Optional[str] = None

    def ticks_to_dollars(self, ticks: float, qty: int = 1) -> float:
        return ticks * self.tick_value * qty


@dataclass(frozen=True)
class CostSpec:
    instrument: str
    commission_per_side: float
    exchange_fee_per_side: float
    regulatory_fee_per_side: float
    spread_ticks_rth: float
    spread_ticks_overnight: float
    slippage_ticks_market: float
    slippage_ticks_stop: float
    verified: bool

    @property
    def fees_per_side(self) -> float:
        return (self.commission_per_side + self.exchange_fee_per_side
                + self.regulatory_fee_per_side)


@dataclass(frozen=True)
class AccountSpec:
    size: str                 # "50k" | "100k" | "150k"
    start_balance: float
    profit_target: float
    mll: float
    dll: float
    max_contracts_minis: int
    max_contracts_micros: int


@dataclass(frozen=True)
class CombineRules:
    mll_mode: str                     # "eod_trailing"
    mll_stops_at_start_balance: bool
    mll_check: str                    # "realtime" | "eod"
    dll_optional: bool
    dll_includes_open_pnl: bool
    dll_breach_policy: str            # "lockout" | "fail"
    min_trading_days: int
    consistency_best_day_max_pct_of_target: float
    micros_per_mini_ratio: int
    verified: bool


# scaling tier: (exclusive upper bound on XFA balance or None for top, max minis)
ScalingTier = Tuple[Optional[float], int]


@dataclass(frozen=True)
class FundedRules:
    monthly_fee: float
    mll_mode: str
    mll_stops_at_start_balance: bool
    mll_check: str
    dll_breach_policy: str
    can_reset: bool
    back2funded_fee: Dict[str, float]
    scaling_plan: Dict[str, Tuple[ScalingTier, ...]]
    verified: bool

    def scaling_max_minis(self, size: str, xfa_balance: float) -> int:
        """Max mini-equivalent lots for the current XFA balance (= profit
        over start in combine coordinates).

        Boundary semantics follow the official chart ("Below $1,500" then
        "$1,500-$2,000" then "Above $2,000"): the first bound is exclusive
        (balance < 1500 -> tier 1); a balance exactly on a shared boundary
        belongs to the LOWER tier (exactly $2,000 -> 3 lots on 50K), which is
        also the conservative reading."""
        tiers = self.scaling_plan.get(size)
        if not tiers:
            raise ConfigError(f"no scaling plan for size {size}")
        for i, (bound, max_minis) in enumerate(tiers):
            if bound is None:
                return max_minis
            hit = xfa_balance < bound if i == 0 else xfa_balance <= bound
            if hit:
                return max_minis
        return tiers[-1][1]


@dataclass(frozen=True)
class PayoutPolicy:
    path: str                          # "standard" (modeled) | "consistency"
    min_winning_days: int
    winning_day_min_profit: float
    require_positive_since_last_payout: bool
    winning_day_counter_resets_after_payout: bool
    max_pct_of_balance: float
    per_payout_cap: Dict[str, float]
    dll_at_checkout_doubles_caps: bool
    min_payout: float
    profit_split_trader: float
    mll_locks_at_zero_after_first_payout: bool
    verified: bool


@dataclass(frozen=True)
class PricingPlan:
    plan: str                 # "standard" | "no_activation_fee"
    size: str
    monthly: float
    activation: float


@dataclass(frozen=True)
class RiskOverrides:
    personal_dll_default: float
    personal_dll_allowed: List[float]
    use_topstep_dll: bool


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path                # instrument data root
    path: Path                # absolute dir containing the files
    pattern: str
    kind: str
    ts_col: str
    notes: str = ""


@dataclass(frozen=True)
class PathsConfig:
    datasets: Dict[str, DatasetSpec]   # key: "<root>.<dataset>", e.g. "nq.mbp1"
    outputs_dir: Path
    full_scan_max_bytes: int

    def data_roots(self) -> List[Path]:
        return sorted({d.root for d in self.datasets.values()})


@dataclass(frozen=True)
class SessionConfig:
    timezone: str
    globex_open_ct: time
    globex_close_ct: time
    eod_flatten_ct: time
    reopen_ct: time
    allowed_entry_start_ct: time
    allowed_entry_end_ct: time
    rth: Dict[str, Dict[str, time]]
    holidays: Tuple[date, ...] = ()                      # NOT valid trade dates
    early_closes: Dict[date, time] = field(default_factory=dict)  # halt time CT


@dataclass(frozen=True)
class Config:
    instruments: Dict[str, InstrumentSpec]
    costs: Dict[str, CostSpec]
    fill_models: Dict[str, str]
    accounts: Dict[str, AccountSpec]
    combine_rules: CombineRules
    funded_rules: FundedRules
    payout: PayoutPolicy
    pricing: Dict[str, Dict[str, PricingPlan]]   # plan -> size -> PricingPlan
    reset_fee_mode: str
    payout_fee: float
    risk: RiskOverrides
    volatility_restrictions: Dict[str, Dict[str, Dict[str, int]]]
    sessions: SessionConfig
    paths: PathsConfig
    project_root: Path

    def pricing_plan(self, plan: str, size: str) -> PricingPlan:
        try:
            return self.pricing[plan][size]
        except KeyError:
            raise ConfigError(f"no pricing for plan={plan!r} size={size!r}")

    def reset_fee(self, plan: str, size: str) -> float:
        if self.reset_fee_mode == "monthly_rebill":
            return self.pricing_plan(plan, size).monthly
        raise ConfigError(f"unknown reset_fee_mode {self.reset_fee_mode!r}")

    def effective_dll(self, size: str, personal_dll: Optional[float] = None) -> float:
        """min(Topstep DLL if enabled, user's personal DLL)."""
        dll = personal_dll if personal_dll is not None else self.risk.personal_dll_default
        if dll not in self.risk.personal_dll_allowed:
            raise ConfigError(
                f"personal DLL {dll} not in allowed set {self.risk.personal_dll_allowed}")
        if self.risk.use_topstep_dll:
            return min(self.accounts[size].dll, dll)
        return dll

    def instrument_max_minis(self, size: str, instrument: str) -> Optional[int]:
        """Temporary volatility cap in MINI contracts (None = no extra cap).
        Micro-contract caps are a separate unit — see instrument_max_micros;
        mixing them here would treat '30 micros' as a 30-mini allowance."""
        caps = self.volatility_restrictions.get(instrument, {}).get("max_minis")
        return caps.get(size) if caps else None

    def instrument_max_micros(self, size: str, instrument: str) -> Optional[int]:
        """Temporary volatility cap in MICRO contracts (None = no extra cap)."""
        caps = self.volatility_restrictions.get(instrument, {}).get("max_micros")
        return caps.get(size) if caps else None


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with open(path, "r") as f:
        out = yaml.safe_load(f)
    if not isinstance(out, dict):
        raise ConfigError(f"config file {path} did not parse to a mapping")
    return out


def load_config(configs_dir: os.PathLike = None) -> Config:
    configs = Path(configs_dir) if configs_dir else _default_configs_dir()
    project_root = configs.parent

    inst_raw = _load_yaml(configs / "instruments.yaml")
    costs_raw = _load_yaml(configs / "costs.yaml")
    top_raw = _load_yaml(configs / "topstep.yaml")
    sess_raw = _load_yaml(configs / "sessions.yaml")
    paths_raw = _load_yaml(configs / "paths.yaml")

    instruments: Dict[str, InstrumentSpec] = {}
    for sym, d in _req(inst_raw, "instruments", "instruments.yaml").items():
        instruments[sym] = InstrumentSpec(
            symbol=sym,
            name=_req(d, "name", f"instrument {sym}"),
            exchange=_req(d, "exchange", f"instrument {sym}"),
            currency=_req(d, "currency", f"instrument {sym}"),
            tick_size=float(_req(d, "tick_size", f"instrument {sym}")),
            tick_value=float(_req(d, "tick_value", f"instrument {sym}")),
            point_value=float(_req(d, "point_value", f"instrument {sym}")),
            micro=d.get("micro"),
            micro_of=d.get("micro_of"),
            micros_per_mini=d.get("micros_per_mini"),
            spread_symbol_marker=d.get("spread_symbol_marker"),
        )
        if instruments[sym].tick_size <= 0 or instruments[sym].tick_value <= 0:
            raise ConfigError(f"instrument {sym}: tick_size/tick_value must be > 0")
        implied = instruments[sym].tick_value / instruments[sym].tick_size
        if abs(implied - instruments[sym].point_value) > 1e-9:
            raise ConfigError(
                f"instrument {sym}: point_value {instruments[sym].point_value} "
                f"inconsistent with tick_value/tick_size = {implied}")

    costs: Dict[str, CostSpec] = {}
    for sym, d in _req(costs_raw, "costs", "costs.yaml").items():
        if sym not in instruments:
            raise ConfigError(f"costs.yaml references unknown instrument {sym}")
        costs[sym] = CostSpec(
            instrument=sym,
            commission_per_side=float(_req(d, "commission_per_side", f"costs {sym}")),
            exchange_fee_per_side=float(_req(d, "exchange_fee_per_side", f"costs {sym}")),
            regulatory_fee_per_side=float(_req(d, "regulatory_fee_per_side", f"costs {sym}")),
            spread_ticks_rth=float(_req(d, "spread_ticks_rth", f"costs {sym}")),
            spread_ticks_overnight=float(_req(d, "spread_ticks_overnight", f"costs {sym}")),
            slippage_ticks_market=float(_req(d, "slippage_ticks_market", f"costs {sym}")),
            slippage_ticks_stop=float(_req(d, "slippage_ticks_stop", f"costs {sym}")),
            verified=bool(d.get("verified", False)),
        )

    accounts: Dict[str, AccountSpec] = {}
    for size, d in _req(_req(top_raw, "combine", "topstep.yaml"), "accounts",
                        "topstep.yaml combine").items():
        accounts[size] = AccountSpec(
            size=size,
            start_balance=float(_req(d, "start_balance", f"account {size}")),
            profit_target=float(_req(d, "profit_target", f"account {size}")),
            mll=float(_req(d, "mll", f"account {size}")),
            dll=float(_req(d, "dll", f"account {size}")),
            max_contracts_minis=int(_req(d, "max_contracts_minis", f"account {size}")),
            max_contracts_micros=int(_req(d, "max_contracts_micros", f"account {size}")),
        )

    cr = _req(top_raw["combine"], "rules", "topstep.yaml combine")
    combine_rules = CombineRules(
        mll_mode=_req(cr, "mll_mode", "combine.rules"),
        mll_stops_at_start_balance=bool(_req(cr, "mll_stops_at_start_balance", "combine.rules")),
        mll_check=_req(cr, "mll_check", "combine.rules"),
        dll_optional=bool(_req(cr, "dll_optional", "combine.rules")),
        dll_includes_open_pnl=bool(_req(cr, "dll_includes_open_pnl", "combine.rules")),
        dll_breach_policy=_req(cr, "dll_breach_policy", "combine.rules"),
        min_trading_days=int(_req(cr, "min_trading_days", "combine.rules")),
        consistency_best_day_max_pct_of_target=float(
            _req(cr, "consistency_best_day_max_pct_of_target", "combine.rules")),
        micros_per_mini_ratio=int(_req(cr, "micros_per_mini_ratio", "combine.rules")),
        verified=bool(cr.get("verified", False)),
    )

    fr = _req(_req(top_raw, "funded", "topstep.yaml"), "rules", "topstep.yaml funded")
    scaling: Dict[str, Tuple[ScalingTier, ...]] = {}
    for size, tiers in _req(fr, "scaling_plan", "funded.rules").items():
        parsed: List[ScalingTier] = []
        for t in tiers:
            bound = t["balance_below"]
            parsed.append((float(bound) if bound is not None else None,
                           int(t["max_minis"])))
        if parsed[-1][0] is not None:
            raise ConfigError(f"scaling plan for {size}: last tier must be unbounded")
        bounds = [b for b, _ in parsed[:-1]]
        if bounds != sorted(bounds):
            raise ConfigError(f"scaling plan for {size}: tiers must be ascending")
        scaling[size] = tuple(parsed)
    funded_rules = FundedRules(
        monthly_fee=float(_req(fr, "monthly_fee", "funded.rules")),
        mll_mode=_req(fr, "mll_mode", "funded.rules"),
        mll_stops_at_start_balance=bool(_req(fr, "mll_stops_at_start_balance", "funded.rules")),
        mll_check=_req(fr, "mll_check", "funded.rules"),
        dll_breach_policy=_req(fr, "dll_breach_policy", "funded.rules"),
        can_reset=bool(_req(fr, "can_reset", "funded.rules")),
        back2funded_fee={k: float(v) for k, v in
                         _req(fr, "back2funded_fee", "funded.rules").items()},
        scaling_plan=scaling,
        verified=bool(fr.get("verified", False)),
    )

    # the simulator implements exactly these modes; anything else in the YAML
    # must fail loudly rather than being silently ignored
    for ctx, mll_mode, mll_check, dll_policy in (
            ("combine.rules", combine_rules.mll_mode, combine_rules.mll_check,
             combine_rules.dll_breach_policy),
            ("funded.rules", funded_rules.mll_mode, funded_rules.mll_check,
             funded_rules.dll_breach_policy)):
        if mll_mode != "eod_trailing":
            raise ConfigError(f"{ctx}: unsupported mll_mode {mll_mode!r} "
                              "(simulator implements 'eod_trailing')")
        if mll_check != "realtime":
            raise ConfigError(f"{ctx}: unsupported mll_check {mll_check!r} "
                              "(simulator implements 'realtime')")
        if dll_policy not in ("lockout", "fail"):
            raise ConfigError(f"{ctx}: unsupported dll_breach_policy {dll_policy!r}")

    po = _req(top_raw["funded"], "payout", "topstep.yaml funded")
    payout = PayoutPolicy(
        path=_req(po, "path", "funded.payout"),
        min_winning_days=int(_req(po, "min_winning_days", "funded.payout")),
        winning_day_min_profit=float(_req(po, "winning_day_min_profit", "funded.payout")),
        require_positive_since_last_payout=bool(
            _req(po, "require_positive_since_last_payout", "funded.payout")),
        winning_day_counter_resets_after_payout=bool(
            _req(po, "winning_day_counter_resets_after_payout", "funded.payout")),
        max_pct_of_balance=float(_req(po, "max_pct_of_balance", "funded.payout")),
        per_payout_cap={k: float(v) for k, v in
                        _req(po, "per_payout_cap", "funded.payout").items()},
        dll_at_checkout_doubles_caps=bool(
            _req(po, "dll_at_checkout_doubles_caps", "funded.payout")),
        min_payout=float(_req(po, "min_payout", "funded.payout")),
        profit_split_trader=float(_req(po, "profit_split_trader", "funded.payout")),
        mll_locks_at_zero_after_first_payout=bool(
            _req(po, "mll_locks_at_zero_after_first_payout", "funded.payout")),
        verified=bool(po.get("verified", False)),
    )

    pricing_raw = _req(top_raw, "pricing", "topstep.yaml")
    pricing: Dict[str, Dict[str, PricingPlan]] = {}
    for plan in ("standard", "no_activation_fee"):
        pricing[plan] = {}
        for size, d in _req(pricing_raw, plan, "topstep.yaml pricing").items():
            pricing[plan][size] = PricingPlan(
                plan=plan, size=size,
                monthly=float(_req(d, "monthly", f"pricing {plan} {size}")),
                activation=float(d.get("activation", 0)),
            )
            if size not in accounts:
                raise ConfigError(f"pricing for unknown account size {size}")

    ro = _req(top_raw, "risk_overrides", "topstep.yaml")
    risk = RiskOverrides(
        personal_dll_default=float(_req(ro, "personal_dll_default", "risk_overrides")),
        personal_dll_allowed=[float(x) for x in
                              _req(ro, "personal_dll_allowed", "risk_overrides")],
        use_topstep_dll=bool(_req(ro, "use_topstep_dll", "risk_overrides")),
    )
    if risk.personal_dll_default not in risk.personal_dll_allowed:
        raise ConfigError("personal_dll_default must be in personal_dll_allowed")

    restrictions = {
        sym: {kind: {sz: int(n) for sz, n in caps.items()}
              for kind, caps in d.items()}
        for sym, d in top_raw.get("restrictions", {}).get("volatility_current", {}).items()
    }

    s = _req(sess_raw, "sessions", "sessions.yaml")
    rth = {}
    for sym, w in s.get("rth", {}).items():
        rth[sym] = {"start_ct": _parse_ct(w["start_ct"]), "end_ct": _parse_ct(w["end_ct"])}
    sessions = SessionConfig(
        timezone=_req(s, "timezone", "sessions"),
        globex_open_ct=_parse_ct(_req(s, "globex_open_ct", "sessions")),
        globex_close_ct=_parse_ct(_req(s, "globex_close_ct", "sessions")),
        eod_flatten_ct=_parse_ct(_req(s, "eod_flatten_ct", "sessions")),
        reopen_ct=_parse_ct(_req(s, "reopen_ct", "sessions")),
        allowed_entry_start_ct=_parse_ct(s["allowed_entry"]["start_ct"]),
        allowed_entry_end_ct=_parse_ct(s["allowed_entry"]["end_ct"]),
        rth=rth,
        holidays=tuple(sorted(_parse_date(x) for x in (s.get("holidays") or []))),
        early_closes={_parse_date(k): _parse_ct(str(v))
                      for k, v in (s.get("early_closes") or {}).items()},
    )

    datasets: Dict[str, DatasetSpec] = {}
    for root_key, root_d in _req(paths_raw, "data_roots", "paths.yaml").items():
        root = Path(_req(root_d, "root", f"data_roots.{root_key}"))
        for ds_key, ds in _req(root_d, "datasets", f"data_roots.{root_key}").items():
            rel = _req(ds, "path", f"dataset {root_key}.{ds_key}")
            datasets[f"{root_key}.{ds_key}"] = DatasetSpec(
                name=f"{root_key}.{ds_key}",
                root=root,
                path=(root / rel).resolve() if rel != "." else root,
                pattern=_req(ds, "pattern", f"dataset {root_key}.{ds_key}"),
                kind=_req(ds, "kind", f"dataset {root_key}.{ds_key}"),
                ts_col=_req(ds, "ts_col", f"dataset {root_key}.{ds_key}"),
                notes=str(ds.get("notes", "")).strip(),
            )

    outputs_dir = (project_root / _req(paths_raw, "outputs_dir", "paths.yaml")).resolve()
    paths = PathsConfig(
        datasets=datasets,
        outputs_dir=outputs_dir,
        full_scan_max_bytes=int(paths_raw.get("full_scan_max_bytes", 300_000_000)),
    )
    # outputs must never sit under a data root (read-only guarantee);
    # resolve() both sides so symlinks can't defeat the check
    for r in paths.data_roots():
        try:
            outputs_dir.relative_to(r.resolve())
            raise ConfigError(f"outputs_dir {outputs_dir} is inside data root {r}")
        except ValueError:
            pass

    return Config(
        instruments=instruments, costs=costs,
        fill_models=dict(costs_raw.get("fill_models", {})),
        accounts=accounts, combine_rules=combine_rules, funded_rules=funded_rules,
        payout=payout, pricing=pricing,
        reset_fee_mode=str(_req(pricing_raw, "reset_fee_mode", "pricing")),
        payout_fee=float(pricing_raw.get("payout_fee", 0)),
        risk=risk, volatility_restrictions=restrictions,
        sessions=sessions, paths=paths, project_root=project_root,
    )


def _default_configs_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs"
