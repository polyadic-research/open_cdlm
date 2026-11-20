"""Train RePlaid with the shared PyTorch training loop."""

from .. import training


def accumulation_steps(config, world):
    return training.accumulation_steps(config, world, model_name="replaid")


def train(config, *, device, rank, world, resume=None, stop_after_steps=None, init_from=None):
    return training.train(
        config,
        model_name="replaid",
        device=device,
        rank=rank,
        world=world,
        resume=resume,
        stop_after_steps=stop_after_steps,
        init_from=init_from,
    )


def main(argv=None):
    return training.main(argv, model_name="replaid")


if __name__ == "__main__":
    main()
