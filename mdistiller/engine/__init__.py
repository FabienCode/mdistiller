from .trainer import BaseTrainer, CRDTrainer, AugTrainer, DoubleLRTrainer, DFKDTrainer

trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "aug": AugTrainer,
    "double": DoubleLRTrainer,
    "dfkd": DFKDTrainer,
}
