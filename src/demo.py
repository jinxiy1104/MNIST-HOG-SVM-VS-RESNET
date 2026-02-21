import sys

import numpy as np
import pygame

from .config import (
    BRUSH_RADIUS,
    CANVAS_ORIGIN,
    CANVAS_SIZE,
    FPS,
    GRID_SIZE,
    THROTTLE_MS,
    WINDOW_H,
    WINDOW_W,
)
from .infer import infer, load_models
from .preprocessing import canvas600_to_image28


def is_blank_image28(image28: np.ndarray) -> bool:
    ink = float(image28.mean())
    return ink < 0.01


def draw_circle_on_canvas(canvas: np.ndarray, cx: int, cy: int, radius: int) -> None:
    h, w = canvas.shape
    x0 = max(0, cx - radius)
    x1 = min(w - 1, cx + radius)
    y0 = max(0, cy - radius)
    y1 = min(h - 1, cy + radius)
    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    canvas[y0 : y1 + 1, x0 : x1 + 1][mask] = 1.0


def draw_interpolated_stroke(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    radius: int,
    step_px: int = 2,
) -> bool:
    dist = float(np.hypot(x1 - x0, y1 - y0))
    steps = max(1, int(np.ceil(dist / max(1, step_px))))
    drew_any = False
    h, w = canvas.shape

    for i in range(steps + 1):
        t = i / steps
        xi = int(round(x0 + (x1 - x0) * t))
        yi = int(round(y0 + (y1 - y0) * t))
        if 0 <= xi < w and 0 <= yi < h:
            draw_circle_on_canvas(canvas, xi, yi, radius)
            drew_any = True
    return drew_any


def render_image28_grid(screen: pygame.Surface, image28: np.ndarray) -> None:
    ox, oy = CANVAS_ORIGIN
    cell = CANVAS_SIZE // GRID_SIZE
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            g = int(np.clip(image28[r, c], 0.0, 1.0) * 255)
            rect = pygame.Rect(ox + c * cell, oy + r * cell, cell, cell)
            pygame.draw.rect(screen, (g, g, g), rect)
            pygame.draw.rect(screen, (35, 35, 35), rect, 1)
    pygame.draw.rect(screen, (200, 200, 200), (ox - 2, oy - 2, CANVAS_SIZE + 4, CANVAS_SIZE + 4), 2)
    pygame.draw.rect(screen, (90, 90, 90), (ox - 1, oy - 1, CANVAS_SIZE + 2, CANVAS_SIZE + 2), 1)


def render_probs(screen: pygame.Surface, probs: np.ndarray, font: pygame.font.Font) -> None:
    left = 820
    bottom = 620
    bar_w = 45
    gap = 20
    max_h = 380

    pygame.draw.line(screen, (190, 190, 190), (left - 20, bottom), (left + 10 * (bar_w + gap), bottom), 2)
    for i in range(10):
        h = int(float(np.clip(probs[i], 0.0, 1.0)) * max_h)
        x = left + i * (bar_w + gap)
        bar_rect = pygame.Rect(x, bottom - h, bar_w, h)
        pygame.draw.rect(screen, (80, 180, 255), bar_rect)
        label = font.render(str(i), True, (220, 220, 220))
        screen.blit(label, (x + 14, bottom + 8))


def main() -> None:
    pygame.init()
    pygame.display.set_caption("MNIST HOG+SVM vs ResNet Demo")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 30)

    load_models()

    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
    image28 = canvas600_to_image28(canvas)
    svm_pred = None
    probs = np.zeros(10, dtype=np.float32)

    drawing = False
    prev_pos: tuple[int, int] | None = None
    canvas_changed = False
    last_infer_time_ms = pygame.time.get_ticks()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    canvas.fill(0.0)
                    image28 = canvas600_to_image28(canvas)
                    svm_pred = None
                    probs = np.zeros(10, dtype=np.float32)
                    canvas_changed = False
                    prev_pos = None
                    last_infer_time_ms = pygame.time.get_ticks()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                drawing = True
                mx, my = event.pos
                cx = mx - CANVAS_ORIGIN[0]
                cy = my - CANVAS_ORIGIN[1]
                prev_pos = (cx, cy)
                if 0 <= cx < CANVAS_SIZE and 0 <= cy < CANVAS_SIZE:
                    draw_circle_on_canvas(canvas, cx, cy, BRUSH_RADIUS)
                    canvas_changed = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                drawing = False
                prev_pos = None

        if drawing and pygame.mouse.get_pressed(num_buttons=3)[0]:
            mx, my = pygame.mouse.get_pos()
            cx = mx - CANVAS_ORIGIN[0]
            cy = my - CANVAS_ORIGIN[1]
            if prev_pos is None:
                prev_pos = (cx, cy)
            drew = draw_interpolated_stroke(
                canvas,
                prev_pos[0],
                prev_pos[1],
                cx,
                cy,
                BRUSH_RADIUS,
                step_px=2,
            )
            if drew:
                canvas_changed = True
            prev_pos = (cx, cy)

        now_ms = pygame.time.get_ticks()
        if (drawing or canvas_changed) and (now_ms - last_infer_time_ms >= THROTTLE_MS):
            image28 = canvas600_to_image28(canvas)
            if is_blank_image28(image28):
                svm_pred = None
                probs = np.zeros(10, dtype=np.float32)
            else:
                svm_pred, probs = infer(image28)
            last_infer_time_ms = now_ms
            canvas_changed = False

        screen.fill((20, 22, 25))
        render_image28_grid(screen, image28)
        render_probs(screen, probs, font)
        resnet_text = font.render("ResNet Prediction:", True, (235, 235, 235))
        screen.blit(resnet_text, (820, 80))
        if svm_pred is None:
            display_text = "Draw a digit"
        else:
            display_text = f"SVM Prediction: {svm_pred}"
        svm_text = font.render(display_text, True, (235, 235, 235))
        screen.blit(svm_text, (820, 680))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
