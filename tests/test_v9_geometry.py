from defense4uavswarm.v8_sim import make_agents, make_intrinsics
from defense4uavswarm.v9_geomdyn import image_bottom_center_to_ground


def test_image_bottom_center_projects_to_ground():
    intr = make_intrinsics(640, 480, 80)
    agent = make_agents(1, 0, 640, 480, intr)[0]
    point = image_bottom_center_to_ground((320, 480), agent)
    assert point is not None
    assert len(point) == 2
