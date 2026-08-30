"""
Thin wrapper around Open-Meteo's free geocoding + forecast APIs.
No API key required, no signup, no rate-limit headaches for a student
project - this is a real, working weather integration rather than
just relying on the general web-search tool to guess at conditions.
"""

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes -> (description, emoji)
WEATHER_CODES = {
    0: ("Clear sky", "☀️"),
    1: ("Mainly clear", "🌤️"),
    2: ("Partly cloudy", "⛅"),
    3: ("Overcast", "☁️"),
    45: ("Fog", "🌫️"),
    48: ("Rime fog", "🌫️"),
    51: ("Light drizzle", "🌦️"),
    53: ("Drizzle", "🌦️"),
    55: ("Dense drizzle", "🌧️"),
    56: ("Freezing drizzle", "🌧️"),
    57: ("Freezing drizzle", "🌧️"),
    61: ("Slight rain", "🌦️"),
    63: ("Rain", "🌧️"),
    65: ("Heavy rain", "🌧️"),
    66: ("Freezing rain", "🌧️"),
    67: ("Freezing rain", "🌧️"),
    71: ("Slight snow", "🌨️"),
    73: ("Snow", "🌨️"),
    75: ("Heavy snow", "❄️"),
    77: ("Snow grains", "❄️"),
    80: ("Rain showers", "🌦️"),
    81: ("Rain showers", "🌧️"),
    82: ("Violent showers", "⛈️"),
    85: ("Snow showers", "🌨️"),
    86: ("Snow showers", "🌨️"),
    95: ("Thunderstorm", "⛈️"),
    96: ("Thunderstorm, hail", "⛈️"),
    99: ("Thunderstorm, hail", "⛈️"),
}


def _describe(code) -> tuple[str, str]:
    return WEATHER_CODES.get(code, ("Conditions unavailable", "🌡️"))


def geocode_city(name: str) -> dict | None:
    """Resolve a free-text place name to lat/lon via Open-Meteo geocoding."""
    resp = requests.get(
        GEOCODE_URL,
        params={"name": name.strip(), "count": 1, "language": "en", "format": "json"},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("results")
    if not results:
        return None
    hit = results[0]
    return {
        "name": hit.get("name"),
        "country": hit.get("country"),
        "latitude": hit["latitude"],
        "longitude": hit["longitude"],
    }


def get_forecast(place: str, days: int = 5) -> dict | None:
    """
    Return current conditions + a short daily forecast for a place name.
    Returns None if the place name can't be resolved to a location.
    """
    location = geocode_city(place)
    if not location:
        return None

    resp = requests.get(
        FORECAST_URL,
        params={
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": "temperature_2m,weather_code,relative_humidity_2m,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": max(1, min(days, 7)),
        },
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()

    current = payload.get("current", {})
    current_desc, current_icon = _describe(current.get("weather_code", -1))

    daily = payload.get("daily", {})
    days_out = []
    for i, d in enumerate(daily.get("time", [])):
        desc, icon = _describe(daily["weather_code"][i])
        days_out.append(
            {
                "date": d,
                "icon": icon,
                "desc": desc,
                "high": daily["temperature_2m_max"][i],
                "low": daily["temperature_2m_min"][i],
                "rain_chance": (daily.get("precipitation_probability_max") or [None])[i],
            }
        )

    return {
        "place": location["name"],
        "country": location["country"],
        "current": {
            "temp": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "wind": current.get("wind_speed_10m"),
            "desc": current_desc,
            "icon": current_icon,
        },
        "daily": days_out,
    }


def format_forecast_text(forecast: dict | None) -> str:
    """Plain-text summary of a forecast dict, safe to drop into an LLM prompt."""
    if not forecast:
        return "No weather data available for this destination."

    lines = [
        f"Current weather in {forecast['place']}, {forecast['country']}: "
        f"{forecast['current']['temp']}°C, {forecast['current']['desc']} "
        f"(humidity {forecast['current']['humidity']}%, "
        f"wind {forecast['current']['wind']} km/h)."
    ]
    for d in forecast["daily"]:
        lines.append(
            f"{d['date']}: {d['desc']}, {d['low']}-{d['high']}°C, "
            f"rain chance {d['rain_chance']}%"
        )
    return "\n".join(lines)
