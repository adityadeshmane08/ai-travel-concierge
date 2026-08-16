"""
Simple SQLite persistence for saved trip searches.

Uses stdlib sqlite3 - no extra dependency needed. The DB file
(travel_concierge.db) is created automatically on first use.
"""

import sqlite3
from datetime import datetime, date
from pathlib import Path

DB_PATH = Path(__file__).parent / "travel_concierge.db"


def init_db() -> None:
    """Create the searches table if it doesn't already exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS searches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                destination TEXT NOT NULL,
                origin TEXT,
                start_date TEXT,
                end_date TEXT,
                travelers INTEGER,
                budget INTEGER,
                travel_style TEXT,
                interests TEXT,
                itinerary TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_search(
    destination: str,
    origin: str,
    start_date: date,
    end_date: date,
    travelers: int,
    budget: int,
    travel_style: str,
    interests: list[str],
    itinerary: str,
) -> int:
    """Insert a completed trip search + generated itinerary. Returns the new row id."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            INSERT INTO searches
                (destination, origin, start_date, end_date, travelers,
                 budget, travel_style, interests, itinerary, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                destination,
                origin,
                str(start_date),
                str(end_date),
                travelers,
                budget,
                travel_style,
                ", ".join(interests) if interests else "",
                itinerary,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        return cur.lastrowid


def get_saved_searches(limit: int = 20) -> list[sqlite3.Row]:
    """Return the most recent saved searches, newest first."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM searches ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return rows


def delete_search(search_id: int) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM searches WHERE id = ?", (search_id,))
        conn.commit()
