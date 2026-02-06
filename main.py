import os
import uuid
import time
import logging
import re as _re
from typing import Optional, Dict, Any, Tuple

import httpx
import psycopg2

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# App / Config
# -----------------------------------------------------------------------------
app = FastAPI(title="Cerbero Executor (CEFI)")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

EXECUTOR_API_KEY = (os.getenv("EXECUTOR_API_KEY", "") or "").strip()

METAAPI_TOKEN = (os.getenv("METAAPI_TOKEN", "") or "").strip()
METAAPI_ACCOUNT_ID = (os.getenv("METAAPI_ACCOUNT_ID", "") or "").strip()
METAAPI_API_BASE = (
    (os.getenv("METAAPI_API_BASE", "") or "https://mt-client-api-v1.new-york.agiliumtrade.ai")
    .strip()
    .rstrip("/")
)

DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()

DEFAULT_VOLUME = float(os.getenv("DEFAULT_VOLUME", "0.01"))
FALLBACK_MIN_TICKS = int(os.getenv("FALLBACK_MIN_TICKS", "10"))

EXECUTOR_SAFE_MODE_REQUIRE_SLTP = True
SYMBOLS_CACHE_TTL_SEC = 3600  # 1h

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _check_key(got: Optional[str]) -> None:
    if EXECUTOR_API_KEY and got != EXECUTOR_API_KEY:
        raise HTTPException(status_code=401, detail="invalid executor key")


def _db_conn():
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail={"code": "DB_NOT_CONFIGURED", "msg": "DATABASE_URL missing"})
    return psycopg2.connect(DATABASE_URL, connect_timeout=5)


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _pick(meta: Optional[Dict[str, Any]], *keys: str) -> Any:
    if not meta:
        return None
    for k in keys:
        if k in meta and meta[k] is not None:
            return meta[k]
    return None


def _spec_float(spec: dict, *keys):
    if not isinstance(spec, dict):
        return None
    for k in keys:
        try:
            v = spec.get(k)
            if v is None:
                continue
            return float(v)
        except Exception:
            continue
    return None


def _round_to_step(x: float, step: float) -> float:
    if step is None or step <= 0:
        return float(x)
    return float(round(float(x) / float(step)) * float(step))


def _round_price(px: float, digits: Optional[int]) -> float:
    if digits is None:
        return float(px)
    try:
        return float(round(float(px), int(digits)))
    except Exception:
        return float(px)


async def _metaapi_get_json(client: httpx.AsyncClient, url: str, headers: Dict[str, str]) -> Any:
    r = await client.get(url, headers=headers)
    try:
        data = r.json()
    except Exception:
        data = {"raw": r.text}
    if r.status_code >= 300:
        raise HTTPException(status_code=502, detail={"metaapi_http": r.status_code, "url": url, "body": data})
    return data


# -----------------------------------------------------------------------------
# Symbol mapping (executor-side)
# -----------------------------------------------------------------------------
_SYMBOLS_CACHE: Dict[str, Tuple[float, list]] = {}  # account_id -> (ts, symbols)


def _norm_sym(s: str) -> str:
    s = (s or "").upper().strip()
    s = _re.sub(r"[\s\-\._/]", "", s)
    return s


def _pick_best(candidates: list) -> Optional[str]:
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: (len(x), x))[0]


def _resolve_symbol_from_inventory(canonical: str, inventory: list) -> Optional[str]:
    c = (canonical or "").upper().strip()
    inv = inventory or []
    inv_norm = [(s, _norm_sym(s)) for s in inv]
    c_norm = _norm_sym(c)

    # 1) normalized contains (handles suffix/prefix: EURUSDm, m.EURUSD, EURUSD-ECN)
    hits = [orig for (orig, n) in inv_norm if c_norm and c_norm in n]
    if hits:
        return _pick_best(hits)

    # 2) minimal aliases for your universe
    aliases = {
        "XAUUSD": ["XAUUSD", "GOLD"],
        "XAGUSD": ["XAGUSD", "SILVER"],
        "LIGHTCMDUSD": ["WTI", "USOIL", "XTIUSD", "CL", "OIL"],
        "BTCUSD": ["BTCUSD", "BTC"],
        "ETHUSD": ["ETHUSD", "ETH"],
    }
    for a in aliases.get(c, []):
        a_norm = _norm_sym(a)
        hits = [orig for (orig, n) in inv_norm if a_norm and a_norm in n]
        if hits:
            return _pick_best(hits)

    return None


async def _metaapi_list_symbols(client: httpx.AsyncClient, account_id: str, headers: dict) -> list:
    now = time.time()
    hit = _SYMBOLS_CACHE.get(account_id)
    if hit and (now - hit[0] < SYMBOLS_CACHE_TTL_SEC):
        return hit[1]

    url = f"{METAAPI_API_BASE}/users/current/accounts/{account_id}/symbols"
    data = await _metaapi_get_json(client, url, headers)

    if isinstance(data, list):
        syms = [str(x) for x in data]
    elif isinstance(data, dict) and "symbols" in data and isinstance(data["symbols"], list):
        syms = [str(x) for x in data["symbols"]]
    else:
        syms = []

    _SYMBOLS_CACHE[account_id] = (now, syms)
    return syms


async def resolve_broker_symbol(client: httpx.AsyncClient, account_id: str, canonical_symbol: str, headers: dict) -> str:
    inv = await _metaapi_list_symbols(client, account_id, headers)
    resolved = _resolve_symbol_from_inventory(canonical_symbol, inv)
    if not resolved:
        raise HTTPException(
            status_code=400,
            detail={"code": "SYMBOL_UNMAPPED", "canonical_symbol": canonical_symbol, "msg": "Symbol not found on broker inventory"},
        )
    return resolved


# -----------------------------------------------------------------------------
# DB mapping helpers
# -----------------------------------------------------------------------------
def _get_cefi_account_id(tenant_email: str, provider_account_id: str) -> str:
    q_tenant = "SELECT id FROM tenants WHERE lower(email) = lower(%s) LIMIT 1"
    q_acc = """
    SELECT id, provider, metaapi_account_id
    FROM cefi_accounts
    WHERE tenant_id = %s
    ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST
    LIMIT 1
    """
    q_set = "UPDATE cefi_accounts SET metaapi_account_id = %s, updated_at = now() WHERE id = %s"

    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q_tenant, (tenant_email,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(
                    status_code=400,
                    detail={"code": "TENANT_NOT_FOUND", "msg": "No tenant for tenant_email", "tenant_email": tenant_email},
                )
            tenant_id = str(row[0])

            cur.execute(q_acc, (tenant_id,))
            acc = cur.fetchone()
            if not acc:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "code": "CEFI_ACCOUNT_NOT_FOUND",
                        "msg": "No cefi_accounts row for tenant",
                        "tenant_email": tenant_email,
                        "tenant_id": tenant_id,
                    },
                )

            cefi_account_id, _provider, metaapi_id = str(acc[0]), acc[1], acc[2]

            if provider_account_id:
                if metaapi_id is None or str(metaapi_id).strip() == "":
                    cur.execute(q_set, (provider_account_id, cefi_account_id))
                    conn.commit()
                else:
                    if str(metaapi_id).strip() != str(provider_account_id).strip():
                        raise HTTPException(
                            status_code=400,
                            detail={
                                "code": "CEFI_ACCOUNT_MISMATCH",
                                "msg": "provider_account_id does not match cefi_accounts.metaapi_account_id",
                                "tenant_email": tenant_email,
                                "cefi_account_id": cefi_account_id,
                                "db_metaapi_account_id": str(metaapi_id),
                                "provider_account_id": str(provider_account_id),
                            },
                        )

            return cefi_account_id


def _db_get_symbol_map(cefi_account_id: str, canonical_symbol: str) -> Optional[str]:
    q = """
    SELECT broker_symbol
    FROM cefi_symbol_map
    WHERE cefi_account_id = %s AND canonical_symbol = %s
    LIMIT 1
    """
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (cefi_account_id, canonical_symbol))
            row = cur.fetchone()
            return str(row[0]) if row else None


def _db_upsert_symbol_map(
    cefi_account_id: str,
    canonical_symbol: str,
    broker_symbol: str,
    confidence: str = "AUTO",
    verified: bool = True,
) -> None:
    q = """
    INSERT INTO cefi_symbol_map (cefi_account_id, canonical_symbol, broker_symbol, confidence, verified, updated_at)
    VALUES (%s, %s, %s, %s, %s, now())
    ON CONFLICT (cefi_account_id, canonical_symbol)
    DO UPDATE SET broker_symbol = EXCLUDED.broker_symbol,
                  confidence    = EXCLUDED.confidence,
                  verified      = EXCLUDED.verified,
                  updated_at    = now()
    """
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (cefi_account_id, canonical_symbol, broker_symbol, confidence, verified))
        conn.commit()


# -----------------------------------------------------------------------------
# SL/TP helpers
# -----------------------------------------------------------------------------
def _compute_abs_sl_tp_from_pct(
    *,
    direction: str,
    entry: float,
    sl_pct: Optional[float],
    tp_pct: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    # sl_pct/tp_pct are decimals: e.g. 0.0025 == 0.25%
    if entry <= 0:
        return None, None
    d = (direction or "").upper()
    sl = tp = None

    if sl_pct is not None and sl_pct > 0:
        if d == "LONG":
            sl = entry * (1.0 - sl_pct)
        else:
            sl = entry * (1.0 + sl_pct)

    if tp_pct is not None and tp_pct > 0:
        if d == "LONG":
            tp = entry * (1.0 + tp_pct)
        else:
            tp = entry * (1.0 - tp_pct)

    return sl, tp


def _apply_min_distance(
    *,
    direction: str,
    entry: float,
    sl: Optional[float],
    tp: Optional[float],
    min_dist: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    if not min_dist or min_dist <= 0 or entry <= 0:
        return sl, tp

    d = (direction or "").upper()

    def ensure_sl(x: Optional[float]) -> Optional[float]:
        if x is None:
            return None
        if abs(entry - x) >= min_dist:
            return x
        return (entry - min_dist) if d == "LONG" else (entry + min_dist)

    def ensure_tp(x: Optional[float]) -> Optional[float]:
        if x is None:
            return None
        if abs(x - entry) >= min_dist:
            return x
        return (entry + min_dist) if d == "LONG" else (entry - min_dist)

    return ensure_sl(sl), ensure_tp(tp)


async def _fetch_equity_from_metaapi(
    client: httpx.AsyncClient,
    account_id: str,
    headers: Dict[str, str],
) -> Optional[float]:
    acc_url = f"{METAAPI_API_BASE}/users/current/accounts/{account_id}/account-information?refreshTerminalState=true"
    try:
        logging.info("SIZING_EQUITY_FETCH start account_id=%s url=%s", account_id, acc_url)
        acc = await _metaapi_get_json(client, acc_url, headers)

        # Some providers might wrap, but your example is flat.
        if isinstance(acc, dict) and "accountInformation" in acc and isinstance(acc["accountInformation"], dict):
            acc = acc["accountInformation"]

        if not isinstance(acc, dict):
            logging.warning("SIZING_EQUITY_FETCH unexpected acc type=%s", type(acc))
            return None

        for k in ("equity", "balance", "marginEquity"):
            try:
                if acc.get(k) is not None:
                    eq = float(acc.get(k))
                    logging.info("SIZING_EQUITY_FETCH done account_id=%s %s=%s", account_id, k, eq)
                    return eq
            except Exception:
                pass

        logging.warning("SIZING_EQUITY_FETCH no equity fields account_id=%s keys=%s", account_id, list(acc.keys())[:40])
        return None
    except Exception as e:
        logging.exception("SIZING_EQUITY_FETCH fail account_id=%s err=%s", account_id, e)
        return None


# -----------------------------------------------------------------------------
# API Models
# -----------------------------------------------------------------------------
class ExecuteIn(BaseModel):
    provider_account_id: Optional[str] = None
    tenant_email: str
    symbol: str
    direction: str = Field(..., pattern="^(LONG|SHORT)$")
    timeframe: Optional[str] = None
    risk_pct: Optional[float] = None
    strength_class: Optional[str] = None
    ts_signal: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/ping", response_class=PlainTextResponse)
def ping():
    return "ok"


@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"


@app.post("/v1/execute")
async def v1_execute(
    body: ExecuteIn,
    x_executor_key: Optional[str] = Header(None, alias="X-Executor-Key"),
):
    _check_key(x_executor_key)

    if not METAAPI_TOKEN:
        raise HTTPException(status_code=500, detail="metaapi not configured (missing METAAPI_TOKEN)")

    symbol = (body.symbol or "").strip().upper()
    direction = (body.direction or "").strip().upper()
    if direction not in ("LONG", "SHORT"):
        raise HTTPException(status_code=400, detail="direction must be LONG/SHORT")

    account_id = (body.provider_account_id or METAAPI_ACCOUNT_ID or "").strip()
    if not account_id:
        raise HTTPException(
            status_code=500,
            detail="metaapi not configured (missing provider_account_id and METAAPI_ACCOUNT_ID fallback)",
        )

    headers = {
        "Content-Type": "application/json",
        "auth-token": METAAPI_TOKEN,
        "X-Request-Id": str(uuid.uuid4()),
    }

    meta = body.meta or {}

    # SAFE MODE: require SL/TP as either absolute (sl/tp) or percent (sl_pct/tp_pct)
    if EXECUTOR_SAFE_MODE_REQUIRE_SLTP:
        sl_abs = _to_float(_pick(meta, "sl", "stopLoss"))
        tp_abs = _to_float(_pick(meta, "tp", "takeProfit"))
        sl_dec = _to_float(_pick(meta, "sl_pct_dec", "sl_pct"))
        tp_dec = _to_float(_pick(meta, "tp_pct_dec", "tp_pct"))

        if (sl_abs is None and sl_dec is None) or (tp_abs is None and tp_dec is None):
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "MISSING_SLTP",
                    "msg": "Safe-mode: SL/TP richiesti (assoluti o percentuali decimali)",
                    "sl_abs": sl_abs,
                    "tp_abs": tp_abs,
                    "sl_dec": sl_dec,
                    "tp_dec": tp_dec,
                },
            )

    action_type = "ORDER_TYPE_BUY" if direction == "LONG" else "ORDER_TYPE_SELL"

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Map tenant -> cefi_account_id and broker symbol
        cefi_account_id = _get_cefi_account_id(body.tenant_email, account_id)

        broker_symbol = _db_get_symbol_map(cefi_account_id, symbol)
        if not broker_symbol:
            broker_symbol = await resolve_broker_symbol(client, account_id, symbol, headers)
            _db_upsert_symbol_map(cefi_account_id, symbol, broker_symbol, confidence="AUTO", verified=True)

        quote_url = f"{METAAPI_API_BASE}/users/current/accounts/{account_id}/symbols/{broker_symbol}/current-price"
        spec_url = f"{METAAPI_API_BASE}/users/current/accounts/{account_id}/symbols/{broker_symbol}/specification"

        quote = await _metaapi_get_json(client, quote_url, headers)
        try:
            spec = await _metaapi_get_json(client, spec_url, headers)
        except Exception:
            spec = {}

        bid = _to_float(quote.get("bid")) if isinstance(quote, dict) else None
        ask = _to_float(quote.get("ask")) if isinstance(quote, dict) else None

        entry_ref = ask if direction == "LONG" else bid
        if entry_ref is None:
            entry_ref = bid if bid is not None else ask
        if entry_ref is None:
            raise HTTPException(status_code=502, detail={"metaapi_http": "no_quote", "quote": quote})

        digits = spec.get("digits") if isinstance(spec, dict) else None
        tick_size = _to_float(spec.get("tickSize")) if isinstance(spec, dict) else None
        stops_level = _to_float(spec.get("stopsLevel")) if isinstance(spec, dict) else None

        min_dist = None
        if stops_level is not None and tick_size is not None and stops_level >= 0 and tick_size > 0:
            min_dist = float(stops_level) * float(tick_size)
        elif tick_size is not None and tick_size > 0:
            min_dist = float(FALLBACK_MIN_TICKS) * float(tick_size)

        # SL/TP percent (decimals) preferred if present
        sl_pct = _to_float(_pick(meta, "sl_pct_dec", "sl_pct", "stop_loss_pct", "slPercent"))
        tp_pct = _to_float(_pick(meta, "tp_pct_dec", "tp_pct", "take_profit_pct", "tpPercent"))

        abs_sl = abs_tp = None
        used_mode = "ABSOLUTE_META"

        if sl_pct is not None or tp_pct is not None:
            abs_sl, abs_tp = _compute_abs_sl_tp_from_pct(
                direction=direction, entry=float(entry_ref), sl_pct=sl_pct, tp_pct=tp_pct
            )
            abs_sl, abs_tp = _apply_min_distance(
                direction=direction, entry=float(entry_ref), sl=abs_sl, tp=abs_tp, min_dist=min_dist
            )
            if abs_sl is not None:
                abs_sl = _round_price(abs_sl, digits)
            if abs_tp is not None:
                abs_tp = _round_price(abs_tp, digits)

            # If pct path fails to produce, fallback to absolute meta
            if abs_sl is not None and abs_tp is not None:
                used_mode = "PCT_FROM_BROKER_ENTRY"

        if used_mode == "ABSOLUTE_META":
            abs_tp = _to_float(_pick(meta, "tp", "takeProfit"))
            abs_sl = _to_float(_pick(meta, "sl", "stopLoss"))

        # ---------------------------------------------------------------------
        # RISK_BASED_POSITION_SIZING_V1_STRICT
        # ---------------------------------------------------------------------
        explicit_vol = None
        try:
            if meta.get("volume_lots") is not None:
                explicit_vol = float(meta.get("volume_lots"))
        except Exception:
            explicit_vol = None

        if explicit_vol is not None:
            vol = float(explicit_vol)
        else:
            rp = body.risk_pct
            if rp is None:
                try:
                    rp = float(meta.get("risk_pct")) if meta.get("risk_pct") is not None else None
                except Exception:
                    rp = None

            equity = None
            for k in ("equity_usd", "equity", "balance_usd", "balance"):
                try:
                    if meta.get(k) is not None:
                        equity = float(meta.get(k))
                        break
                except Exception:
                    pass

            if equity is None:
                equity = await _fetch_equity_from_metaapi(client, account_id, headers)

            if abs_sl is None or entry_ref is None:
                raise HTTPException(
                    status_code=400,
                    detail={"code": "SIZING_NO_SL", "msg": "Sizing richiede SL assoluto (abs_sl) e entry_ref"},
                )

            stop_dist = abs(float(entry_ref) - float(abs_sl))
            if stop_dist <= 0:
                raise HTTPException(
                    status_code=400,
                    detail={"code": "SIZING_BAD_STOP", "msg": "stop_dist non valido", "stop_dist": stop_dist},
                )

            contract_size = _spec_float(spec, "contractSize", "contract_size", "contractsize")

            if rp is None or equity is None or contract_size is None or contract_size <= 0:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "code": "SIZING_UNAVAILABLE",
                        "msg": "Impossibile calcolare volume (mancano risk_pct/equity/spec contractSize)",
                        "risk_pct": rp,
                        "equity": equity,
                        "contractSize": contract_size,
                    },
                )

            risk_amount = float(equity) * float(rp)
            risk_per_lot = float(stop_dist) * float(contract_size)

            if risk_per_lot <= 0:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "code": "SIZING_BAD_RISK_PER_LOT",
                        "msg": "risk_per_lot non valido",
                        "risk_per_lot": risk_per_lot,
                        "stop_dist": stop_dist,
                        "contractSize": contract_size,
                    },
                )

            lots = float(risk_amount) / float(risk_per_lot)

            min_vol = _spec_float(spec, "minVolume", "min_volume", "minimumVolume")
            max_vol = _spec_float(spec, "maxVolume", "max_volume", "maximumVolume")
            step_vol = _spec_float(spec, "volumeStep", "volume_step", "stepVolume")

            if min_vol is not None:
                lots = max(lots, float(min_vol))
            if max_vol is not None:
                lots = min(lots, float(max_vol))
            if step_vol is not None and step_vol > 0:
                lots = _round_to_step(lots, float(step_vol))

            if lots <= 0:
                raise HTTPException(status_code=400, detail={"code": "SIZING_NONPOSITIVE", "msg": "lots <= 0", "lots": lots})

            vol = float(lots)

        trade_payload: Dict[str, Any] = {
            "actionType": action_type,
            "symbol": broker_symbol,
            "volume": vol,
        }
        if abs_tp is not None:
            trade_payload["takeProfit"] = float(abs_tp)
        if abs_sl is not None:
            trade_payload["stopLoss"] = float(abs_sl)

        trade_url = f"{METAAPI_API_BASE}/users/current/accounts/{account_id}/trade"
        r = await client.post(trade_url, json=trade_payload, headers=headers)
        try:
            data = r.json()
        except Exception:
            data = {"raw": r.text}

    if r.status_code >= 300:
        raise HTTPException(status_code=502, detail={"metaapi_http": r.status_code, "body": data, "sent": trade_payload})

    return {
        "ok": True,
        "provider": "metaapi",
        "status": "SUBMITTED",
        "provider_order_id": str(data.get("orderId") or ""),
        "provider_position_id": str(data.get("positionId") or data.get("dealId") or ""),
        "echo": body.model_dump(),
        "metaapi": data,
        "risk_translation": {
            "mode": used_mode,
            "entry_ref": float(entry_ref),
            "quote": {"bid": bid, "ask": ask},
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
            "computed": {"stopLoss": trade_payload.get("stopLoss"), "takeProfit": trade_payload.get("takeProfit")},
            "spec_hint": {"digits": digits, "tickSize": tick_size, "stopsLevel": stops_level, "min_dist": min_dist},
            "spec_keys": sorted(list(spec.keys())) if isinstance(spec, dict) else [],
            "spec_sample": {
                "tickValue": spec.get("tickValue") if isinstance(spec, dict) else None,
                "tickSize": spec.get("tickSize") if isinstance(spec, dict) else None,
                "contractSize": spec.get("contractSize") if isinstance(spec, dict) else None,
                "minVolume": spec.get("minVolume") if isinstance(spec, dict) else None,
                "maxVolume": spec.get("maxVolume") if isinstance(spec, dict) else None,
                "volumeStep": spec.get("volumeStep") if isinstance(spec, dict) else None,
            },
        },
    }
