"""Training entry point.

    python src/train.py experiment=ctgen
    python src/train.py experiment=mrgen
"""

from __future__ import annotations

import hydra
import lightning as L
import rootutils
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def _instantiate_all(cfg: DictConfig | None) -> list:
    if not cfg:
        return []
    return [hydra.utils.instantiate(c) for c in cfg.values() if c is not None and "_target_" in c]


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    callbacks = _instantiate_all(cfg.get("callbacks"))
    loggers = _instantiate_all(cfg.get("logger"))
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)
    for lg in loggers:
        lg.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))


if __name__ == "__main__":
    main()
