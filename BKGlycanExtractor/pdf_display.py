PDF_COLOR_NAMES = {
    "blue": (0, 0, 1),
    "green": (0, 1, 0),
    "red": (1, 0, 0),
    "yellow": (1, 1, 0),
    "pink": (1, 0.4, 0.8),
    "black": (0, 0, 0),
}

PDF_BORDER_DASHES = {
    "solid": [],
    "dashed": [3, 3],
    "dotted": [1, 2],
}

def pdf_color(value, default=(0, 0, 1)):
    if value is None:
        return default
    if isinstance(value, str):
        return PDF_COLOR_NAMES.get(value.lower(), default)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            color = tuple(float(v) for v in value)
        except (TypeError, ValueError):
            return default
        if max(color) > 1:
            color = tuple(v / 255 for v in color)
        return tuple(max(0, min(1, v)) for v in color)
    return default

def get_glycan_display(glycan):
    votes = glycan.get('upvotes', 0) - glycan.get('downvotes', 0)
    default_color = (
        (0, 1, 0) if votes > 0
        else (1, 0, 0) if votes < 0
        else (0, 0, 1)
    )
    style = str(glycan.get("drawing_style", "solid")).lower()
    return (
        pdf_color(glycan.get("color"), default_color),
        PDF_BORDER_DASHES.get(style, []),
    )

if __name__ == "__main__":
    assert get_glycan_display(
        {"color": "red", "drawing_style": "dashed"}
    ) == ((1, 0, 0), [3, 3])
    assert get_glycan_display({"downvotes": 1}) == ((1, 0, 0), [])
    assert get_glycan_display(
        {"color": [255, 128, 0]}
    ) == ((1, 128 / 255, 0), [])
