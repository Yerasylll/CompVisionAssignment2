import numpy as np
from types import SimpleNamespace
from gesture_core import features

def test_translation_invariance():
    points=[SimpleNamespace(x=float(i)/100,y=float(i*i)/1000,z=0.0) for i in range(21)]
    moved=[SimpleNamespace(x=p.x+.2,y=p.y+.1,z=p.z) for p in points]
    np.testing.assert_allclose(features(points),features(moved),atol=1e-5)
