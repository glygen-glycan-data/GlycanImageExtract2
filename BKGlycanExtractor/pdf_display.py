PDF_COLOR_NAMES = {
    "blue": (0, 0, 1),
    "green": (0, 1, 0),
    "red": (1, 0, 0),
}

PDF_BORDER_DASHES = {
    "solid": [],
    "dashed": [3, 3],
}

def pdf_color(value, default=(0, 0, 1)):
    if isinstance(value, str):
        return PDF_COLOR_NAMES.get(value.lower(), default)
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