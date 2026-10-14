from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .geometry import Point3, WorldMap


COLORS = {
    "background": "white",
    "grid": "#E5E7EB",
    "obstacle": "#B7B7B7",
    "obstacle_outline": "#666666",
    "baseline": "#1769E0",
    "candidate": "#FF6B00",
    "start": "#169B16",
    "goal": "#E31A1C",
    "text": "#111111",
}


def render_projections(
    world: WorldMap,
    baseline: list[Point3],
    candidate: list[Point3],
    output: str | Path,
):
    width, height = 1500, 720
    margin = 80
    gutter = 80
    panel_w = (width - 2 * margin - gutter) // 2
    panel_h = height - 2 * margin
    image = Image.new("RGB", (width, height), COLORS["background"])
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=24)

    def project(point: Point3, panel: int, mode: str):
        left = margin + panel * (panel_w + gutter)
        if mode == "xy":
            u, v = point[0] / world.bounds[0], point[1] / world.bounds[1]
        else:
            u, v = point[0] / world.bounds[0], point[2] / world.bounds[2]
        return left + u * panel_w, margin + (1.0 - v) * panel_h

    def draw_panel(panel: int, mode: str, title: str):
        left = margin + panel * (panel_w + gutter)
        draw.rectangle((left, margin, left + panel_w, margin + panel_h), outline="#333333", width=2)
        for frac in (0.2, 0.4, 0.6, 0.8):
            x = left + frac * panel_w
            y = margin + frac * panel_h
            draw.line((x, margin, x, margin + panel_h), fill=COLORS["grid"], width=1)
            draw.line((left, y, left + panel_w, y), fill=COLORS["grid"], width=1)
        for box in world.inflated_obstacles:
            if mode == "xy":
                a = project((box.xmin, box.ymin, 0), panel, mode)
                b = project((box.xmax, box.ymax, 0), panel, mode)
            else:
                a = project((box.xmin, 0, box.zmin), panel, mode)
                b = project((box.xmax, 0, box.zmax), panel, mode)
            rect = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
            draw.rectangle(rect, fill=COLORS["obstacle"], outline=COLORS["obstacle_outline"], width=1)
        for path, color, line_width in (
            (baseline, COLORS["baseline"], 4),
            (candidate, COLORS["candidate"], 5),
        ):
            if len(path) >= 2:
                draw.line([project(p, panel, mode) for p in path], fill=color, width=line_width, joint="curve")
        if baseline:
            for point, color in ((baseline[0], COLORS["start"]), (baseline[-1], COLORS["goal"])):
                x, y = project(point, panel, mode)
                draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color, outline="black")
        draw.text((left, 28), title, font=font, fill=COLORS["text"])

    draw_panel(0, "xy", f"{world.name}: X-Y projection")
    draw_panel(1, "xz", f"{world.name}: X-Z projection")
    draw.line((margin, height - 28, margin + 60, height - 28), fill=COLORS["baseline"], width=4)
    draw.text((margin + 70, height - 42), "Astar-L", font=font, fill=COLORS["text"])
    draw.line((margin + 190, height - 28, margin + 250, height - 28), fill=COLORS["candidate"], width=5)
    draw.text((margin + 260, height - 42), "Astar-L-Psi-K", font=font, fill=COLORS["text"])
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    image.save(output)

