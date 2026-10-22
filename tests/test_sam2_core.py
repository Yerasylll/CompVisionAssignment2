from types import SimpleNamespace
from sam2_core import hand_box

def test_box_clipped():
    landmarks=[SimpleNamespace(x=.05,y=.05),SimpleNamespace(x=.9,y=.95)]
    x1,y1,x2,y2=hand_box(landmarks,(100,200))
    assert 0<=x1<x2<=99
    assert 0<=y1<y2<=199

def test_box_needs_hand():
    try: hand_box([], (100,100))
    except ValueError: return
    assert False, 'Expected ValueError'
