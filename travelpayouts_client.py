"""
Thin wrapper around the Travelpayouts Data API (Aviasales cached flight
prices) - a free, self-serve alternative to Amadeus, which shut down its
self-service developer portal on July 17, 2026.

Sign up (free, no credit card, no traffic minimum for this endpoint) at:
https://www.travelpayouts.com/programs/100/tools/api

Note: this API serves CACHED prices (recent searches by real users, not
a live shopping call), which is normal for a free tier and is fine for
a demo/student project - just don't present it as guaranteed bookable
pricing.
"""

import requests

DATA_API_BASE = "https://api.travelpayouts.com"
AUTOCOMPLETE_BASE = "https://autocomplete.travelpayouts.com"

# ... [keep your existing get_iata_code function] ...

def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    token: str,
    currency: str = "inr",
) -> list[dict]:
    """
    Look up cached cheapest-flight data for a route/month using the
    Travelpayouts v1/prices/cheap endpoint. 
    If the exact date is not cached, searches for any available dates.
    """
    origin_code = get_iata_code(origin)
    dest_code = get_iata_code(destination)

    if not origin_code or not dest_code:
        raise ValueError(
            f"Could not resolve airport codes for '{origin}' -> '{destination}'."
        )

    # First attempt: Search with the specific date
    params = {
        "origin": origin_code,
        "destination": dest_code,
        "depart_date": departure_date,
        "currency": currency,
        "token": token,
    }
    
    resp = requests.get(f"{DATA_API_BASE}/v1/prices/cheap", params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()

    if not payload.get("success", False):
        raise ValueError(payload.get("error") or "Travelpayouts request failed.")

    route_data = payload.get("data", {}).get(dest_code, {})

    # Second attempt: If exact date has no cache, search WITHOUT the date
    if not route_data:
        del params["depart_date"]
        fallback_resp = requests.get(f"{DATA_API_BASE}/v1/prices/cheap", params=params, timeout=20)
        fallback_payload = fallback_resp.json()
        
        if fallback_payload.get("success", False):
            route_data = fallback_payload.get("data", {}).get(dest_code, {})

    # Format the real data
    simplified = []
    for offer in route_data.values():
        simplified.append({
            "origin": origin_code,
            "destination": dest_code,
            "airline": offer.get("airline"),
            "flight_number": offer.get("flight_number"),
            "price": offer.get("price"),
            "currency": currency.upper(),
            "departure_at": offer.get("departure_at"),
        })

    return simplified
