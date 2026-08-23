"""
Thin wrapper around the Travelpayouts Data API (Aviasales cached flight prices).
"""

import requests

DATA_API_BASE = "https://api.travelpayouts.com"
AUTOCOMPLETE_BASE = "https://autocomplete.travelpayouts.com"

def get_iata_code(keyword: str) -> str | None:
    """
    Resolve a free-text city/airport name (e.g. 'Mumbai', 'Goa') to an
    IATA code (e.g. 'BOM', 'GOI') using Travelpayouts' free autocomplete
    endpoint.
    """
    keyword = keyword.strip()
    if len(keyword) == 3 and keyword.isalpha():
        return keyword.upper()

    resp = requests.get(
        f"{AUTOCOMPLETE_BASE}/places2",
        params={"term": keyword, "locale": "en", "types[]": "city"},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        return None
    return results[0]["code"]

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
