import numpy as np
import timml

model = timml.ModelMaq(
    kaq=[0.1, 5.0, 15.0, 5.0],
    z=[1.0, -3.0, -7.0, -7.0, -14.0, -14.0, -30.0, -30.0, -40.0],
    c=[1000.0, 2.0, 2.0, 2.0],
    topboundary="semi",
    npor=[None, None, None, None, None, None, None, None],
    hstar=-2.4,
)
dewatering_east_0 = timml.Well(
    xw=59224.43941000212,
    yw=387382.7326884054,
    Qw=1950.0,
    rw=0.5,
    res=1.0,
    layers=1,
    label=None,
    model=model,
)
dewatering_east_1 = timml.Well(
    xw=59359.38892532585,
    yw=387375.9852126392,
    Qw=1950.0,
    rw=0.5,
    res=1.0,
    layers=1,
    label=None,
    model=model,
)
dewatering_east_2 = timml.Well(
    xw=59360.51350462022,
    yw=387311.88419286045,
    Qw=1950.0,
    rw=0.5,
    res=1.0,
    layers=1,
    label=None,
    model=model,
)
dewatering_east_3 = timml.Well(
    xw=59234.5606236514,
    yw=387298.38924132806,
    Qw=1950.0,
    rw=0.5,
    res=1.0,
    layers=1,
    label=None,
    model=model,
)
dewatering_west_0 = timml.Well(
    xw=58781.35516802254,
    yw=387375.9852126392,
    Qw=900.0,
    rw=0.5,
    res=1.0,
    layers=1,
    label=None,
    model=model,
)
dewatering_west_1 = timml.Well(
    xw=58785.8534852,
    yw=387307.385875683,
    Qw=900.0,
    rw=0.5,
    res=1.0,
    layers=1,
    label=None,
    model=model,
)
channel_0 = timml.PolygonInhomMaq(
    kaq=[0.1, 5.0, 15.0, 5.0],
    z=[0.0, -3.0, -7.0, -7.0, -14.0, -14.0, -30.0, -30.0, -40.0],
    c=[30.0, 2.0, 2.0, 2.0],
    topboundary="semi",
    npor=[None, None, None, None, None, None, None, None],
    hstar=0.0,
    xy=[
        [58921.92757981809, 388617.52075361746],
        [59065.873729496736, 388608.5241192626],
        [59110.85690127131, 387996.7529831283],
        [59146.84343869097, 387447.9582874785],
        [59263.799685304875, 386809.19724827947],
        [59317.77949143437, 386260.40255262965],
        [59110.85690127131, 386251.4059182747],
        [58966.910751592666, 386863.177054409],
        [58921.92757981809, 388617.52075361746],
    ],
    order=4,
    ndeg=6,
    model=model,
)
model.solve()
head = model.headgrid(
    xg=np.arange(53700.0, 63675.0, 150.0), yg=np.arange(390375.0, 382800.0, -150.0)
)

