"""Train LangFlow with the shared PyTorch training loop."""


def main(argv=None):
    from ..training import main as train

    return train(argv, model_name="langflow")


if __name__ == "__main__":
    main()
