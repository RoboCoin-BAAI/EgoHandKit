"""Draw canonical fallback hands without inventing missing MANO meshes."""

import cv2
import numpy as np


FALLBACK_COLOR = (0, 165, 255)  # BGR, distinct from left/right backend meshes.
HAND_EDGES = tuple(
    edge
    for start in (1, 5, 9, 13, 17)
    for edge in ((0, start), (start, start + 1), (start + 1, start + 2),
                 (start + 2, start + 3))
)


def draw_frontend_fallback(image, frame, fallback_sides):
    """Draw visible portions, including fingers whose wrist is off-screen."""
    height, width = image.shape[:2]
    drawn = False
    sources = set()
    for hand in frame.get('hands', []):
        if hand.get('handedness') not in fallback_sides:
            continue
        sources.add(str(hand.get('source', 'frontend')).upper())
        points = np.asarray(hand.get('keypoints_2d'), dtype=np.float64)
        if points.shape != (21, 3):
            continue
        valid = (np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
                 & (np.abs(points[:, :2]) < 1e8).all(axis=1))
        pixels = [tuple(np.rint(p[:2]).astype(int)) if ok else None
                  for p, ok in zip(points, valid)]
        for a, b in HAND_EDGES:
            if valid[a] and valid[b]:
                visible, p, q = cv2.clipLine((0, 0, width, height), pixels[a], pixels[b])
                if visible:
                    cv2.line(image, p, q, FALLBACK_COLOR, 2, cv2.LINE_AA)
                    drawn = True
        for point in pixels:
            if point is not None and 0 <= point[0] < width and 0 <= point[1] < height:
                cv2.circle(image, point, 3, FALLBACK_COLOR, -1, cv2.LINE_AA)
                drawn = True
    if drawn:
        label = next(iter(sources)) if len(sources) == 1 else 'Frontend'
        cv2.putText(image, f'{label} fallback', (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, FALLBACK_COLOR, 2, cv2.LINE_AA)
    return image
