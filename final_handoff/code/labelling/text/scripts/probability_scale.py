"""Allowed probability values per status (single source for schema.json, examples and checks)."""
SCALE = {
    "meets": [0.97, 0.9, 0.8],
    "unsettled": [0.75, 0.6, 0.45],
    "conflict": [0.65, 0.55, 0.45],
    "hedged": [0.5, 0.4, 0.3],
    "below": [0.35, 0.2, 0.1],
    "general_normal": [0.15, 0.08],
    "excluded": [0.12, 0.05],
    "structure_normal": [0.05, 0.02],
    "denied": [0.03, 0.01],
    "not_mentioned": [None],
}
