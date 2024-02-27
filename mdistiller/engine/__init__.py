from .trainer import BaseTrainer, CRDTrainer, AugTrainer, DoubleLRTrainer, DFKDTrainer, BaseTrainer_addloss_info

trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "aug": AugTrainer,
    "double": DoubleLRTrainer,
    "dfkd": DFKDTrainer,
    "base_aug": BaseTrainer_addloss_info
}
