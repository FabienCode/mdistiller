from ._base import Vanilla
from .KD import KD
from .AT import AT
from .OFD import OFD
from .RKD import RKD
from .FitNet import FitNet
from .KDSVD import KDSVD
from .CRD import CRD
from .NST import NST
from .PKT import PKT
from .SP import SP
from .VID import VID
from .ReviewKD import ReviewKD
from .DKD import DKD
from .MDis import MDis
from .MLogits import MLogits
from .CLD import CLD
from .SRT import SRT
from .MV1 import MV1
from .RegKD import RegKD
from .RegKD_KR import RegKD_KR
from .UniLogits import UniLogitsKD
from .UniLogits_single import UniLogitsKD_single
from .MLKD import MLKD
from .UniKDMLReview import UniMLKD
from .hybrid import Hybrid
from .kr_mlkd import KR_MLKD
from .fitnet_kd import FitNet_KD
from .KR_DKD import KR_DKD
from .MVKD import MVKD

distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "AT": AT,
    "OFD": OFD,
    "RKD": RKD,
    "FITNET": FitNet,
    "KDSVD": KDSVD,
    "CRD": CRD,
    "NST": NST,
    "PKT": PKT,
    "SP": SP,
    "VID": VID,
    "REVIEWKD": ReviewKD,
    "DKD": DKD,
    "MDis": MDis,
    "MLogits": MLogits,
    "CLD": CLD,
    "SRT": SRT,
    "MV1": MV1,
    "RegKD": RegKD,
    "RegKD_KR": RegKD_KR,
    "UniLogitsKD": UniLogitsKD,
    "UniLogitsKD_single": UniLogitsKD_single,
    "MLKD": MLKD,
    "UniMLKD": UniMLKD,
    "Hybrid": Hybrid,
    "KR_MLKD": KR_MLKD,
    "FitNet_KD": FitNet_KD,
    "KR_DKD": KR_DKD,
    "MVKD": MVKD,
}
