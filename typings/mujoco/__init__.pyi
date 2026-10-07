# Stub for MuJoCo 3.10. MjModel, MjData, and the mj_* functions exist only in
# compiled modules, and the wheel ships no .pyi file or py.typed marker. Without
# this stub, pyright cannot see them.
from enum import IntEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

type _F = NDArray[np.float64]
type _I = NDArray[np.int32]

__version__: str

class mjtObj(IntEnum):
    mjOBJ_BODY = 1
    mjOBJ_JOINT = 3
    mjOBJ_SITE = 6
    mjOBJ_TENDON = 18
    mjOBJ_ACTUATOR = 19
    mjOBJ_KEY = 24

class mjtJoint(IntEnum):
    mjJNT_FREE = 0
    mjJNT_BALL = 1
    mjJNT_SLIDE = 2
    mjJNT_HINGE = 3

class mjtTrn(IntEnum):
    mjTRN_JOINT = 0
    mjTRN_JOINTINPARENT = 1
    mjTRN_SLIDERCRANK = 2
    mjTRN_TENDON = 3
    mjTRN_SITE = 4
    mjTRN_BODY = 5

class mjtWrap(IntEnum):
    mjWRAP_NONE = 0
    mjWRAP_JOINT = 1

class mjtDyn(IntEnum):
    mjDYN_NONE = 0

class mjtGain(IntEnum):
    mjGAIN_FIXED = 0

class mjtBias(IntEnum):
    mjBIAS_NONE = 0
    mjBIAS_AFFINE = 1

class mjtIntegrator(IntEnum):
    mjINT_EULER = 0
    mjINT_RK4 = 1
    mjINT_IMPLICIT = 2
    mjINT_IMPLICITFAST = 3

class mjtWarning(IntEnum):
    mjWARN_INERTIA = 0
    mjWARN_CONTACTFULL = 1
    mjWARN_CNSTRFULL = 2
    mjWARN_BADQPOS = 3
    mjWARN_BADQVEL = 4
    mjWARN_BADQACC = 5
    mjWARN_BADCTRL = 6

class MjOption:
    timestep: float
    integrator: int
    def __getattr__(self, name: str) -> Any: ...

class MjModel:
    nq: int
    nv: int
    nu: int
    njnt: int
    nbody: int
    nsite: int
    nkey: int
    opt: MjOption
    jnt_type: _I
    jnt_qposadr: _I
    jnt_dofadr: _I
    jnt_bodyid: _I
    body_parentid: _I
    site_bodyid: _I
    tendon_adr: _I
    tendon_num: _I
    wrap_type: _I
    wrap_objid: _I
    wrap_prm: _F
    actuator_trntype: _I
    actuator_trnid: _I
    actuator_dyntype: _I
    actuator_gaintype: _I
    actuator_biastype: _I
    actuator_gainprm: _F
    actuator_biasprm: _F
    actuator_gear: _F
    actuator_ctrllimited: NDArray[np.bool_]
    actuator_ctrlrange: _F
    actuator_forcelimited: NDArray[np.bool_]
    actuator_forcerange: _F
    @staticmethod
    def from_xml_path(
        filename: str, assets: dict[str, bytes] | None = None
    ) -> MjModel: ...
    @staticmethod
    def from_xml_string(
        xml: str, assets: dict[str, bytes] | None = None
    ) -> MjModel: ...
    def __getattr__(self, name: str) -> Any: ...

class MjWarningStat:
    lastinfo: int
    number: int

class _Warnings:
    number: NDArray[np.int32]
    lastinfo: NDArray[np.int32]
    def __getitem__(self, i: int) -> MjWarningStat: ...

class MjData:
    time: float
    qpos: _F
    qvel: _F
    ctrl: _F
    actuator_force: _F
    qfrc_actuator: _F
    xfrc_applied: _F
    xmat: _F
    site_xpos: _F
    ten_length: _F
    ten_velocity: _F
    actuator_length: _F
    actuator_velocity: _F
    warning: _Warnings
    def __init__(self, model: MjModel) -> None: ...
    def __getattr__(self, name: str) -> Any: ...

def mj_step(m: MjModel, d: MjData, nstep: int = 1) -> None: ...
def mj_step1(m: MjModel, d: MjData) -> None: ...
def mj_step2(m: MjModel, d: MjData) -> None: ...
def mj_forward(m: MjModel, d: MjData) -> None: ...
def mj_resetData(m: MjModel, d: MjData) -> None: ...
def mj_resetDataKeyframe(m: MjModel, d: MjData, key: int) -> None: ...
def mj_name2id(m: MjModel, type: int, name: str) -> int: ...
def mj_id2name(m: MjModel, type: int, id: int) -> str | None: ...
def mj_jacSite(
    m: MjModel, d: MjData, jacp: _F | None, jacr: _F | None, site: int
) -> None: ...
def __getattr__(name: str) -> Any: ...
