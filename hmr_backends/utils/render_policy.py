"""Left-hand mesh policy for rendering.

The repository now uses a single, stable left-hand rendering strategy across
all backends: mirror the predicted hand-space mesh along the x-axis for left
hands, and render it with left-hand face winding. This preserves the current
renderer/camera semantics that are already validated to align with the image.
"""


class LeftHandMeshPolicy:
    def apply(self, verts, is_right, mano_params, cam_t):
        if is_right:
            return verts, 1
        v = verts.copy()
        v[:, 0] *= -1
        return v, 0


def build_left_hand_policy(device=None):
    return LeftHandMeshPolicy()
