import torch

from open_cdlm.cdlm import from_pretrained, generate
from open_cdlm.cdlm.replaid.checkpoint import initialize_ema, save_checkpoint
from open_cdlm.cdlm.replaid.model import RePlaid
from open_cdlm.cdlm.replaid.optim import build_optimizer, build_scheduler


def test_shared_loader_reads_original_checkpoint_with_raw_and_ema(
    tmp_path, tiny_config, local_tokenizer,
):
    model = RePlaid(tiny_config.model)
    optimizer = build_optimizer(model, tiny_config)
    scheduler = build_scheduler(optimizer, tiny_config)
    ema = initialize_ema(model)
    ema["gamma_0"].add_(0.1)
    path = save_checkpoint(
        tmp_path, model, local_tokenizer, tiny_config, optimizer, scheduler, ema,
        1, torch.Generator().manual_seed(42), {}, {},
    )
    raw = from_pretrained("replaid", path, weights="raw")
    averaged = from_pretrained("replaid", path)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, raw.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(averaged.gamma_0, ema["gamma_0"])
    assert generate(averaged, num_samples=2, steps=2).shape == (2, 8)
