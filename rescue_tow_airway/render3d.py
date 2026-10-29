"""A small software renderer for the oblique view of the sea and divers.

Pinhole projection, painter's ordering, Lambert shading with sky
reflection and sun glint for the water, and tube meshes for the divers
built from the model's cross-section primitives. Diver surfaces below
the local free surface are drawn under a translucent water layer.
"""

import numpy as np
from matplotlib.colors import to_rgb

SUN = np.array([0.45, -0.35, 0.82])
SUN = SUN / np.linalg.norm(SUN)


class Camera:
    """Pinhole camera.

    Parameters
    ----------
    eye, target : array_like (3,)
        Camera position and the point it looks at (m).
    fov : float
        Horizontal field of view (deg).
    aspect : float
        Width / height of the image.
    """

    def __init__(self, eye, target, fov=50.0, aspect=16 / 9):
        self.eye = np.asarray(eye, float)
        f = np.asarray(target, float) - self.eye
        f /= np.linalg.norm(f)
        r = np.cross(f, [0.0, 0.0, 1.0])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        self.f, self.r, self.u = f, r, u
        self.scale = 1.0 / np.tan(np.radians(fov) / 2)
        self.aspect = aspect

    def project(self, p):
        """Project points (..., 3) to image coordinates and depth."""
        d = p - self.eye
        z = d @ self.f
        x = (d @ self.r) / z * self.scale
        y = (d @ self.u) / z * self.scale
        return np.stack([x, y], -1), z


def shade_water(n, view, depth_col, sky_col, sun_col):
    """Water color from normals, view directions and sky reflection.

    Uses Schlick's approximation of the Fresnel reflectance of water
    (normal-incidence reflectance 0.02) and a Phong glint.
    """
    cosv = np.clip(-(view * n).sum(-1), 0.0, 1.0)
    fres = 0.02 + 0.98 * (1.0 - cosv) ** 5
    refl = view - 2.0 * (view * n).sum(-1)[..., None] * n
    spec = np.clip((refl * SUN).sum(-1), 0.0, 1.0) ** 180
    diff = 0.45 + 0.75 * np.clip((n * SUN).sum(-1), 0.0, 1.0) ** 2
    col = (1 - fres)[..., None] * depth_col * diff[..., None] \
        + fres[..., None] * sky_col
    col = col + 1.4 * spec[..., None] * sun_col
    return np.clip(col, 0, 1)


def tube_quads(rings):
    """Quads joining consecutive rings (R, M, 3) of a tube."""
    a = rings[:-1]
    b = rings[1:]
    a2 = np.roll(a, -1, 1)
    b2 = np.roll(b, -1, 1)
    return np.stack([a, a2, b2, b], 2).reshape(-1, 4, 3)


def quad_normals(q):
    """Unit normals of quads (N, 4, 3)."""
    n = np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1])
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)


def lambert(base, n, ambient=0.35):
    """Diffuse shading of a base color."""
    d = np.clip(np.abs(n @ SUN), 0, 1)
    rgb = np.asarray(to_rgb(base)) if isinstance(base, str) else base
    return np.clip(rgb * (ambient + (1 - ambient) * d)[:, None], 0, 1)
