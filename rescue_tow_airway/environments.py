"""The idealized environments of the study.

All values are round, illustrative numbers typical of the settings named
below. None of them is an estimate for a particular site.

E1  Open-ocean swell: narrow-band, long-crested, long period.
E2  Wind chop: short, steep, short-crested wind sea in a sheltered bay.
E3  Mixed sea: a long-crested swell crossed at 60 degrees by a wind sea.
E4  Tidal current: the E2-type sea entering a 0.75 m/s current that
    opposes (E4-) or follows (E4+) the waves.
E5  Beach exit: a moderate swell shoaling over decreasing depth.
E6  Boat hull: the wind chop reflected from a vertical hull
    (reflection coefficient 0.9).
E7  Confused sea: the E2 wind sea with broad directional spreading.
"""

import numpy as np

from .waves import SeaState, Wall, WaveSystem

DEG = np.pi / 180.0


def e1_swell():
    """E1: open-ocean swell."""
    return SeaState([WaveSystem(1.0, 12.0, 5.0, 0.0, None)],
                    name="E1", label="Open-ocean swell")


def e2_chop():
    """E2: wind chop."""
    return SeaState([WaveSystem(0.6, 3.5, 3.3, 0.0, 10.0)],
                    name="E2", label="Wind chop")


def e3_mixed():
    """E3: swell crossed by a wind sea."""
    return SeaState([WaveSystem(0.8, 11.0, 5.0, 0.0, None),
                     WaveSystem(0.5, 4.0, 3.3, 60 * DEG, 10.0)],
                    name="E3", label="Mixed sea")


def e4_current(sign=-1):
    """E4: wind sea on a uniform current (sign -1 opposing, +1 following)."""
    tag = "E4-" if sign < 0 else "E4+"
    lab = "Opposing current" if sign < 0 else "Following current"
    return SeaState([WaveSystem(0.5, 4.0, 3.3, 0.0, 10.0)],
                    current=(sign * 0.75, 0.0), name=tag, label=lab)


def e5_beach(depth):
    """E5: deep-water swell shoaled to a local depth (m)."""
    return SeaState([WaveSystem(0.8, 9.0, 3.3, 0.0, 20.0)], depth=depth,
                    name="E5", label="Beach exit, h = %.1f m" % depth)


def e6_hull(refl=0.9):
    """E6: wind chop reflected from a vertical hull at x = 0.

    The waves travel toward +x and the hull occupies x > 0, so a body at
    distance ``d`` from the hull sits at x = -d.
    """
    return SeaState([WaveSystem(0.6, 3.5, 3.3, 0.0, 20.0)],
                    wall=Wall((0.0, 0.0), (1.0, 0.0), refl),
                    name="E6", label="Boat hull")


def e7_confused():
    """E7: short-crested, broadly spread wind sea."""
    return SeaState([WaveSystem(0.6, 3.5, 3.3, 0.0, 2.0)],
                    name="E7", label="Confused sea")


def calm():
    """A calm reference sea."""
    return SeaState([WaveSystem(0.25, 3.0, 3.3, 0.0, 10.0)],
                    name="E0", label="Calm")


def suite():
    """The open-water environments used for risk maps."""
    return [e1_swell(), e2_chop(), e3_mixed(), e7_confused(),
            e4_current(-1), e4_current(+1)]
