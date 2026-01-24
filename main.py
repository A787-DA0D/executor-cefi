import os
import uuid
from typing import Optional, Dict, Any, Tuple

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Cerbero Executor (CEFI)")

EXECUTOR_API_KEY = (os.getenv("EXECUTOR_API_KEY", "") or "").strip()

def _check_key(got: Optional[str]):
    if EXECUTOR_API_KEY and got != EXECUTOR_API_KEY:
        raise HTTPException(status_code=401, detail="invalid executor key")

METAAPI_TOKEN = (os.getenv("METAAPI_TOKEN", "") or "").strip()
METAAPI_ACCOUNT_ID = (os.getenv("METAAPI_ACCOUNT_ID", "") or "").strip()
METAAPI_API_BASE = (os.getenv("METAAPI_API_BASE", "") or "https://mt-client-api-v1.new-york.agiliumtrade.ai").strip().rstrip("/")
DEFAULT_VOLUME = float(os.getenv("DEFAULT_VOLUME", "0.01"))

# Safety: minimum SL/TP distance fallback (in "ticks" when spec not available)
FALLBACK_MIN_TICKS = int(os.getenv("FALLBACK_MIN_TICKS", "10"))

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

async def _metaapi_get_json(client: httpx.AsyncClient, url: str, headers: Dict[str, str]) -> Dict[str, Any]:
    r = await client.get(url, headers=headers)
    try:
        data = r.json()
    except Exception:
        data = {"raw": r.text}
    if r.status_code >= 300:
        raise HTTPException(status_code=502, detail={"metaapi_http": r.status_code, "url": url, "body": data})
    return data

def _round_price(px: float, digits: Optional[int]) -> float:
    if digits is None:
        return float(px)
    try:
        return float(round(float(px), int(digits)))
    except Exception:
        return float(px)

def _compute_abs_sl_tp_from_pct(*, direction: str, entry: float, sl_pct: Optional[float], tp_pct: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
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

def _apply_min_distance(*, direction: str, entry: float, sl: Optional[float], tp: Optional[float], min_dist: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    if not min_dist or min_dist <= 0 or entry <= 0:
        return sl, tp

    d = (direction or "").upper()

    def ensure_sl(x: Optional[float]) -> Optional[float]:
        if x is None:
            return None
        if abs(entry - x) >= min_dist:
            return x
        # push SL outward
        if d == "LONG":
            return entry - min_dist
        else:
            return entry + min_dist

    def ensure_tp(x: Optional[float]) -> Optional[float]:
        if x is None:
            return None
        if abs(x - entry) >= min_dist:
            return x
        # push TP outward
        if d == "LONG":
            return entry + min_dist
        else:
            return entry - min_dist

    return ensure_sl(sl), ensure_tp(tp)

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

    vol = DEFAULT_VOLUME
    try:
        if body.meta and body.meta.get("volume_lots") is not None:
            vol = float(body.meta.get("volume_lots"))
    except Exception:
        vol = DEFAULT_VOLUME

    action_type = "ORDER_TYPE_BUY" if direction == "LONG" else "ORDER_TYPE_SELL"

    account_id = (body.provider_account_id or METAAPI_ACCOUNT_ID or "").strip()
    if not account_id:
        raise HTTPException(status_code=500, detail="metaapi not configured (missing provider_account_id and METAAPI_ACCOUNT_ID fallback)")

    headers = {
        "Content-Type": "application/json",
        "auth-token": METAAPI_TOKEN,
        "X-Request-Id": str(uuid.uuid4()),
    }

    # --- Broker-aware entry price (bid/ask) + spec (digits/tickSize) ---
    # REST docs:
    # - current price: GET /users/current/accounts/:accountId/symbols/:symbol/current-price
    # - specification: GET /users/current/accounts/:accountId/symbols/:symbol/specification
    quote_url = f"{METAAPI_API_BASE}/users/current/accounts/{account_id}/symbols/{symbol}/current-price"
    spec_url  = f"{METAAPI_API_BASE}/users/current/accounts/{account_id}/symbols/{symbol}/specification"

    async with httpx.AsyncClient(timeout=30.0) as client:
        quote = await _metaapi_get_json(client, quote_url, headers)
        spec = {}
        try:
            spec = await _metaapi_get_json(client, spec_url, headers)
        except Exception:
            spec = {}

        bid = _to_float(quote.get("bid"))
        ask = _to_float(quote.get("ask"))
        entry_ref = ask if direction == "LONG" else bid
        if entry_ref is None:
            # fallback: if one side missing, try the other
            entry_ref = bid if bid is not None else ask
        if entry_ref is None:
            raise HTTPException(status_code=502, detail={"metaapi_http": "no_quote", "quote": quote})

        digits = spec.get("digits")
        tick_size = _to_float(spec.get("tickSize"))
        stops_level = _to_float(spec.get("stopsLevel"))  # may be absent depending on broker/symbol

        # min_dist: prefer stopsLevel*tickSize; else fallback ticks*tickSize; else None
        min_dist = None
        if stops_level is not None and tick_size is not None and stops_level >= 0 and tick_size > 0:
            min_dist = float(stops_level) * float(tick_size)
        elif tick_size is not None and tick_size > 0:
            min_dist = float(FALLBACK_MIN_TICKS) * float(tick_size)

        # --- Prefer percent-based SL/TP if provided ---
        meta = body.meta or {}
        sl_pct = _to_float(_pick(meta, "sl_pct", "stop_loss_pct", "slPercent", "sl_pct_dec"))
        tp_pct = _to_float(_pick(meta, "tp_pct", "take_profit_pct", "tpPercent", "tp_pct_dec"))

        abs_sl = abs_tp = None
        used_mode = "ABSOLUTE_META"
        if sl_pct is not None or tp_pct is not None:
            used_mode = "PCT_FROM_BROKER_ENTRY"
            abs_sl, abs_tp = _compute_abs_sl_tp_from_pct(direction=direction, entry=float(entry_ref), sl_pct=sl_pct, tp_pct=tp_pct)
            abs_sl, abs_tp = _apply_min_distance(direction=direction, entry=float(entry_ref), sl=abs_sl, tp=abs_tp, min_dist=min_dist)
            if abs_sl is not None:
                abs_sl = _round_price(abs_sl, digits)
            if abs_tp is not None:
                abs_tp = _round_price(abs_tp, digits)

        # fallback to absolute sl/tp in meta if pct missing (or pct produced None)
        if used_mode != "PCT_FROM_BROKER_ENTRY":
            tp = _pick(meta, "takeProfit", "tp")
            sl = _pick(meta, "stopLoss", "sl")
            abs_tp = _to_float(tp)
            abs_sl = _to_float(sl)

        trade_payload: Dict[str, Any] = {
            "actionType": action_type,
            "symbol": symbol,
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
        "provider_order_id": str(data.get("orderId") or data.get("positionId") or data.get("dealId") or ""),
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
        }
    }
