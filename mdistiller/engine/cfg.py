from yacs.config import CfgNode as CN
from .utils import log_msg


def show_cfg(cfg):
    dump_cfg = CN()
    dump_cfg.EXPERIMENT = cfg.EXPERIMENT
    dump_cfg.DATASET = cfg.DATASET
    dump_cfg.DISTILLER = cfg.DISTILLER
    dump_cfg.SOLVER = cfg.SOLVER
    dump_cfg.LOG = cfg.LOG
    if cfg.DISTILLER.TYPE in cfg:
        dump_cfg.update({cfg.DISTILLER.TYPE: cfg.get(cfg.DISTILLER.TYPE)})
    print(log_msg("CONFIG:\n{}".format(dump_cfg.dump()), "INFO"))


CFG = CN()

# Experiment
CFG.EXPERIMENT = CN()
CFG.EXPERIMENT.PROJECT = "distill"
CFG.EXPERIMENT.NAME = ""
CFG.EXPERIMENT.TAG = "default"

# Dataset
CFG.DATASET = CN()
CFG.DATASET.TYPE = "cifar100"
CFG.DATASET.NUM_WORKERS = 2
CFG.DATASET.TEST = CN()
CFG.DATASET.TEST.BATCH_SIZE = 64

# Distiller
CFG.DISTILLER = CN()
CFG.DISTILLER.TYPE = "NONE"  # Vanilla as default
CFG.DISTILLER.TEACHER = "ResNet50"
CFG.DISTILLER.STUDENT = "resnet32"

# Solver
CFG.SOLVER = CN()
CFG.SOLVER.TRAINER = "base"
CFG.SOLVER.BATCH_SIZE = 64
CFG.SOLVER.EPOCHS = 240
CFG.SOLVER.LR = 0.05
CFG.SOLVER.LR_DECAY_STAGES = [150, 180, 210]
CFG.SOLVER.LR_DECAY_RATE = 0.1
CFG.SOLVER.WEIGHT_DECAY = 0.0001
CFG.SOLVER.MOMENTUM = 0.9
CFG.SOLVER.TYPE = "SGD"

# Log
CFG.LOG = CN()
CFG.LOG.TENSORBOARD_FREQ = 500
CFG.LOG.SAVE_CHECKPOINT_FREQ = 40
CFG.LOG.PREFIX = "./output"
# CFG.LOG.WANDB = True

# Distillation Methods

# KD CFG
CFG.KD = CN()
CFG.KD.TEMPERATURE = 4
CFG.KD.LOSS = CN()
CFG.KD.LOSS.CE_WEIGHT = 0.1
CFG.KD.LOSS.KD_WEIGHT = 0.9

# AT CFG
CFG.AT = CN()
CFG.AT.P = 2
CFG.AT.LOSS = CN()
CFG.AT.LOSS.CE_WEIGHT = 1.0
CFG.AT.LOSS.FEAT_WEIGHT = 1000.0

# RKD CFG
CFG.RKD = CN()
CFG.RKD.DISTANCE_WEIGHT = 25
CFG.RKD.ANGLE_WEIGHT = 50
CFG.RKD.LOSS = CN()
CFG.RKD.LOSS.CE_WEIGHT = 1.0
CFG.RKD.LOSS.FEAT_WEIGHT = 1.0
CFG.RKD.PDIST = CN()
CFG.RKD.PDIST.EPSILON = 1e-12
CFG.RKD.PDIST.SQUARED = False

# FITNET CFG
CFG.FITNET = CN()
CFG.FITNET.HINT_LAYER = 2  # (0, 1, 2, 3, 4)
CFG.FITNET.INPUT_SIZE = (32, 32)
CFG.FITNET.LOSS = CN()
CFG.FITNET.LOSS.CE_WEIGHT = 1.0
CFG.FITNET.LOSS.FEAT_WEIGHT = 100.0

# KDSVD CFG
CFG.KDSVD = CN()
CFG.KDSVD.K = 1
CFG.KDSVD.LOSS = CN()
CFG.KDSVD.LOSS.CE_WEIGHT = 1.0
CFG.KDSVD.LOSS.FEAT_WEIGHT = 1.0

# OFD CFG
CFG.OFD = CN()
CFG.OFD.LOSS = CN()
CFG.OFD.LOSS.CE_WEIGHT = 1.0
CFG.OFD.LOSS.FEAT_WEIGHT = 0.001
CFG.OFD.CONNECTOR = CN()
CFG.OFD.CONNECTOR.KERNEL_SIZE = 1

# NST CFG
CFG.NST = CN()
CFG.NST.LOSS = CN()
CFG.NST.LOSS.CE_WEIGHT = 1.0
CFG.NST.LOSS.FEAT_WEIGHT = 50.0

# PKT CFG
CFG.PKT = CN()
CFG.PKT.LOSS = CN()
CFG.PKT.LOSS.CE_WEIGHT = 1.0
CFG.PKT.LOSS.FEAT_WEIGHT = 30000.0

# SP CFG
CFG.SP = CN()
CFG.SP.LOSS = CN()
CFG.SP.LOSS.CE_WEIGHT = 1.0
CFG.SP.LOSS.FEAT_WEIGHT = 3000.0

# VID CFG
CFG.VID = CN()
CFG.VID.LOSS = CN()
CFG.VID.LOSS.CE_WEIGHT = 1.0
CFG.VID.LOSS.FEAT_WEIGHT = 1.0
CFG.VID.EPS = 1e-5
CFG.VID.INIT_PRED_VAR = 5.0
CFG.VID.INPUT_SIZE = (32, 32)

# CRD CFG
CFG.CRD = CN()
CFG.CRD.MODE = "exact"  # ("exact", "relax")
CFG.CRD.FEAT = CN()
CFG.CRD.FEAT.DIM = 128
CFG.CRD.FEAT.STUDENT_DIM = 256
CFG.CRD.FEAT.TEACHER_DIM = 256
CFG.CRD.LOSS = CN()
CFG.CRD.LOSS.CE_WEIGHT = 1.0
CFG.CRD.LOSS.FEAT_WEIGHT = 0.8
CFG.CRD.NCE = CN()
CFG.CRD.NCE.K = 16384
CFG.CRD.NCE.MOMENTUM = 0.5
CFG.CRD.NCE.TEMPERATURE = 0.07

# ReviewKD CFG
CFG.REVIEWKD = CN()
CFG.REVIEWKD.CE_WEIGHT = 1.0
CFG.REVIEWKD.REVIEWKD_WEIGHT = 1.0
CFG.REVIEWKD.WARMUP_EPOCHS = 20
CFG.REVIEWKD.SHAPES = [1, 8, 16, 32]
CFG.REVIEWKD.OUT_SHAPES = [1, 8, 16, 32]
CFG.REVIEWKD.IN_CHANNELS = [64, 128, 256, 256]
CFG.REVIEWKD.OUT_CHANNELS = [64, 128, 256, 256]
CFG.REVIEWKD.MAX_MID_CHANNEL = 512
CFG.REVIEWKD.STU_PREACT = False

# DKD(Decoupled Knowledge Distillation) CFG
CFG.DKD = CN()
CFG.DKD.CE_WEIGHT = 1.0
CFG.DKD.ALPHA = 1.0
CFG.DKD.BETA = 8.0
CFG.DKD.T = 4.0
CFG.DKD.WARMUP = 20

# RegKD
CFG.RegKD = CN()
CFG.RegKD.CE_WEIGHT = 1.0
CFG.RegKD.ALPHA = 1.0
CFG.RegKD.BETA = 8.0
CFG.RegKD.T = 4.0
CFG.RegKD.WARMUP = 20
CFG.RegKD.AREA_NUM = 8
CFG.RegKD.HINT_LAYER = -1
CFG.RegKD.CHANNEL_MASK = 0.5
CFG.RegKD.CHANNEL_KD_WEIGHT = 1.0
CFG.RegKD.AREA_KD_WEIGHT = 3.0
CFG.RegKD.INPUT_SIZE = (32, 32)
CFG.RegKD.HEAT_WEIGHT = 1.0
CFG.RegKD.SIZE_REG_WEIGHT = 1.0
CFG.RegKD.REG_WEIGHT = 0.01
CFG.RegKD.LOGITS_THRESH = 0.80
CFG.RegKD.SHAPES = [1, 8, 16, 32]
CFG.RegKD.OUT_SHAPES = [1, 8, 16, 32]
CFG.RegKD.IN_CHANNELS = [64, 128, 256, 256]
CFG.RegKD.OUT_CHANNELS = [64, 128, 256, 256]
CFG.RegKD.MAX_MID_CHANNEL = 512
CFG.RegKD.STU_PREACT = False

# UniLogits
CFG.Uni = CN()
CFG.Uni.TEMPERATURE = 4
CFG.Uni.GMM_NUM = 5
CFG.Uni.LOSS = CN()
CFG.Uni.LOSS.CE_WEIGHT = 1.0
CFG.Uni.LOSS.LOGITS_WEIGHT = 1.0
CFG.Uni.LOSS.FEAT_KD_WEIGHT = 1.0
CFG.Uni.LOSS.SUPP_WEIGHT = 1.0

