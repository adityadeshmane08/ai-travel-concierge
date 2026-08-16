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


def get_iata_code(keyword: str) -> str | None:
    """
    Resolve a free-text city/airport name (e.g. 'Mumbai', 'Goa') to an
    IATA code (e.g. 'BOM', 'GOI') using Travelpayouts' free autocomplete
    endpoint (no token required). Returns None if nothing is found.
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
    Travelpayouts v1/prices/cheap endpoint. departure_date can be a full
    date (YYYY-MM-DD) or a month (YYYY-MM); the API buckets by month
    internally either way.

    Returns a list of simplified offer dicts: airline, flight_number,
    price, currency, departure_at.
    """
    origin_code = get_iata_code(origin)
    dest_code = get_iata_code(destination)

    if not origin_code or not dest_code:
        raise ValueError(
            f"Could not resolve airport codes for '{origin}' -> '{destination}'."
        )

    resp = requests.get(
        f"{DATA_API_BASE}/v1/prices/cheap",
        params={
            "origin": origin_code,
            "destination": dest_code,
            "depart_date": departure_date,
            "currency": currency,
            "token": token,
        },
        timeout=20,
    )
    resp.raise_for_status()
    payload = resp.json()

    if not payload.get("success", False):
        raise ValueError(payload.get("error") or "Travelpayouts request failed.")

    route_data = payload.get("data", {}).get(dest_code, {})
    if not route_data:
        return []

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
