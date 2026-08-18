"""Data layer: feeds, storage, quality, point-in-time universes.

Depends only on `core` (§3.2). It must never import `trading` or `engine`:
data must never be shaped by what the trader wants it to say.
"""
