from .trainer import BaseTrainer, CRDTrainer, AugTrainer, DoubleLRTrainer

trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "aug": AugTrainer,
    "double": DoubleLRTrainer,
}
