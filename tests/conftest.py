import pytest

from guildgpt.config import GuildGPTConfig, RunConfig


@pytest.fixture
def tiny_config():
    return GuildGPTConfig(max_seq_len=4, vocab_size=24, n_layers=1, n_heads=2, hidden_size=8)


@pytest.fixture
def tiny_run(tiny_config):
    return RunConfig(
        model=tiny_config,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        warmup_steps=0,
        wandb_project=None,
    )
