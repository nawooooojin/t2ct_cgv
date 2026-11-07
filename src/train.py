"""Hydra training entry point: ``python src/train.py experiment=ctgen``."""

from __future__ import annotations

import hydra
import lightning as L
import rootutils
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def _instantiate_group(group: DictConfig | None) -> list:
    """Instantiate every entry of a callbacks/logger group that has a ``_target_``."""
    if not group:
        return []
    return [hydra.utils.instantiate(c) for c in group.values() if c is not None and "_target_" in c]


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """Build datamodule, model, callbacks, loggers and trainer from ``cfg`` and fit."""
    rank_zero_only(print)(OmegaConf.to_yaml(cfg, resolve=True))
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)
    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    callbacks = _instantiate_group(cfg.get("callbacks"))
    loggers = _instantiate_group(cfg.get("logger"))
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))


if __name__ == "__main__":
    main()
