"""PyTorch Lightning DataModule for LangFlow training.

Wraps the existing duo dataloader infrastructure to provide a standard
Lightning DataModule interface. Supports both streaming and offline datasets.
"""

import typing
import lightning as L
import torch


class LangFlowDataModule(L.LightningDataModule):
    """Lightning DataModule wrapping the duo dataloader for LangFlow training.

    Handles:
      - Dataset loading and tokenization (via duo.dataloader.get_dataset)
      - Batching and shuffling
      - Distributed sampling
      - Streaming datasets

    Config keys (mirrors the existing duo config structure):
      data:
        train: str           - training dataset name (e.g. "lm1b", "openwebtext")
        valid: str           - validation dataset name
        cache_dir: str       - cache directory
        wrap: bool           - whether to wrap with BOS/EOS
        streaming: bool      - whether to use streaming mode
        insert_train_eos: bool
        insert_valid_eos: bool
      loader:
        batch_size: int      - per-device batch size
        eval_batch_size: int - per-device eval batch size
        num_workers: int     - data loading workers
        pin_memory: bool     - pin memory for faster transfer
      model:
        length: int          - sequence length
    """

    def __init__(
        self,
        # Data config
        train_dataset: str = "openwebtext",
        valid_dataset: str = "wikitext103",
        tokenizer_name_or_path: str = "gpt2",
        cache_dir: str = "/tmp/data",
        wrap: bool = True,
        streaming: bool = False,
        insert_train_eos: bool = True,
        insert_valid_eos: bool = True,
        # Loader config
        batch_size: int = 128,
        eval_batch_size: int = 128,
        num_workers: int = 16,
        pin_memory: bool = True,
        # Model config (needed for data preprocessing)
        model_length: int = 1024,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])

        # These will be set during setup()
        self.tokenizer = None
        self._train_dataset = None
        self._valid_dataset = None

    # ------------------------------------------------------------------
    # Configuration class for compatibility with duo dataloader
    # ------------------------------------------------------------------

    class _ConfigAdapter:
        """Adapter to make our config look like the duo config object."""
        pass

    def _make_data_config(self):
        """Build a config object compatible with duo.dataloader.get_tokenizer and get_dataset."""
        cfg = self._ConfigAdapter()

        class DataCfg:
            pass

        class LoaderCfg:
            pass

        class ModelCfg:
            pass

        data_cfg = DataCfg()
        data_cfg.train = self.hparams.train_dataset
        data_cfg.valid = self.hparams.valid_dataset
        data_cfg.tokenizer_name_or_path = self.hparams.tokenizer_name_or_path
        data_cfg.cache_dir = self.hparams.cache_dir
        data_cfg.wrap = self.hparams.wrap
        data_cfg.streaming = self.hparams.streaming
        data_cfg.insert_train_eos = self.hparams.insert_train_eos
        data_cfg.insert_valid_eos = self.hparams.insert_valid_eos
        data_cfg.train_revision = None
        data_cfg.valid_revision = None

        loader_cfg = LoaderCfg()
        loader_cfg.batch_size = self.hparams.batch_size
        loader_cfg.eval_batch_size = self.hparams.eval_batch_size
        loader_cfg.num_workers = self.hparams.num_workers
        loader_cfg.pin_memory = self.hparams.pin_memory

        model_cfg = ModelCfg()
        model_cfg.length = self.hparams.model_length

        cfg.data = data_cfg
        cfg.loader = loader_cfg
        cfg.model = model_cfg

        return cfg

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def prepare_data(self):
        """Download / preprocess data (called once on the main process).

        For non-streaming datasets, this triggers the tokenization and caching
        so that worker processes can load the cached data directly.
        """
        from ._data import dataloader as duo_dataloader

        cfg = self._make_data_config()
        # Load the tokenizer (lightweight, needed for dataset creation)
        self.tokenizer = duo_dataloader.get_tokenizer(cfg)

    def setup(self, stage: typing.Optional[str] = None):
        """Set up train/validation datasets.

        Called on every process in DDP. Each process loads its own shard.
        """
        from ._data import dataloader as duo_dataloader

        cfg = self._make_data_config()

        if self.tokenizer is None:
            self.tokenizer = duo_dataloader.get_tokenizer(cfg)

        if stage == "fit" or stage is None:
            # Load training dataset
            self._train_dataset = duo_dataloader.get_dataset(
                dataset_name=cfg.data.train,
                tokenizer=self.tokenizer,
                mode="train",
                wrap=cfg.data.wrap,
                insert_eos=cfg.data.insert_train_eos,
                cache_dir=cfg.data.cache_dir,
                block_size=cfg.model.length,
                streaming=cfg.data.streaming,
                num_proc=cfg.loader.num_workers,
                revision=cfg.data.train_revision,
            )

            # Load validation dataset
            if cfg.data.valid in ["text8", "lm1b", "ag_news"]:
                validation_split = "test"
            else:
                validation_split = "validation"

            self._valid_dataset = duo_dataloader.get_dataset(
                dataset_name=cfg.data.valid,
                tokenizer=self.tokenizer,
                mode=validation_split,
                wrap=cfg.data.wrap,
                insert_eos=cfg.data.insert_valid_eos,
                cache_dir=cfg.data.cache_dir,
                block_size=cfg.model.length,
                streaming=cfg.data.streaming,
                num_proc=cfg.loader.num_workers,
                revision=cfg.data.valid_revision,
            )

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the training dataloader.

        Uses standard shuffling for non-streaming datasets.
        For DDP, Lightning automatically wraps with DistributedSampler.
        """
        loader = torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=not self.hparams.streaming,
            persistent_workers=self.hparams.num_workers > 0,
        )
        return loader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the validation dataloader."""
        loader = torch.utils.data.DataLoader(
            self._valid_dataset,
            batch_size=self.hparams.eval_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=self.hparams.num_workers > 0,
        )
        return loader
