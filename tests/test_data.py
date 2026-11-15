import numpy as np
import pytest

from guildgpt.data import WindowLoader, prepare_data, prompt_split, tokenize_stories


class Tokenizer:
    eos_token_id = 23

    def encode(self, story, add_special_tokens=False):
        return [int(x) for x in story.split()]


def test_split_grouping_and_tokenization(tmp_path):
    records = [
        {"prompt": " Same  prompt ", "story": "1 2"},
        {"prompt": "same prompt", "story": "3 4"},
        {"prompt": "x", "story": ""},
    ]
    assert prompt_split(records[0]["prompt"]) == prompt_split(records[1]["prompt"])
    train, val = tmp_path / "train.bin", tmp_path / "validation.bin"
    counts = tokenize_stories(records, Tokenizer(), train, val)
    chosen = val if counts["validation_stories"] else train
    assert np.fromfile(chosen, dtype=np.uint16).tolist() == [1, 2, 23, 3, 4, 23]


def test_uint16_bounds(tmp_path):
    with pytest.raises(ValueError, match="uint16"):
        tokenize_stories(
            [{"prompt": "p", "story": "65536"}], Tokenizer(), tmp_path / "a", tmp_path / "b"
        )


def test_full_epoch_coverage_and_resume():
    tokens = np.arange(20, dtype=np.uint16)
    loader = WindowLoader(tokens, seq_len=4, batch_size=2, seed=7)
    targets = []
    for i in range(loader.num_batches):
        batch = loader.get_batch(0, i)
        for row in range(2):
            targets.extend(batch.tokens[row, 1:][batch.mask[row].astype(bool)].tolist())
        np.testing.assert_array_equal(batch.tokens, loader.get_batch(0, i).tokens)
    assert sorted(targets) == list(range(1, 20))
    assert len(targets) == len(set(targets))
    assert loader.get_batch(0, 1).tokens.tolist() == loader.get_batch(0, 1).tokens.tolist()
    assert loader.num_batches == 3
    assert not np.array_equal(loader.order(0), loader.order(1))


def test_window_order_is_reused_within_epoch():
    loader = WindowLoader(np.arange(100, dtype=np.uint16), seq_len=4, batch_size=2, seed=7)
    first = loader.order(0)
    loader.get_batch(0, 0)
    loader.get_batch(0, 1)
    assert loader.order(0) is first
    second = loader.order(1)
    assert second is loader.order(1)
    assert not np.array_equal(first, second)


def test_preparation_manifest_is_idempotent_and_detects_stale_input(tmp_path, monkeypatch):
    import datasets
    from transformers import GPT2TokenizerFast

    train_prompt = next(f"p{i}" for i in range(1000) if prompt_split(f"p{i}") == "train")
    val_prompt = next(f"p{i}" for i in range(1000) if prompt_split(f"p{i}") == "validation")

    class Records(list):
        _fingerprint = "fake-fingerprint"

    records = Records(
        [{"prompt": train_prompt, "story": "1 2"}, {"prompt": val_prompt, "story": "3 4"}]
    )
    tokenizer = Tokenizer()
    tokenizer.vocab_size = 24
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: records)
    monkeypatch.setattr(GPT2TokenizerFast, "from_pretrained", lambda *args: tokenizer)
    first = prepare_data(tmp_path, revision="fixed-revision")
    assert first["train_stories"] == first["validation_stories"] == 1
    assert first["train_tokens"] == first["validation_tokens"] == 3
    assert prepare_data(tmp_path, revision="fixed-revision") == first
    with pytest.raises(ValueError, match="differ"):
        prepare_data(tmp_path, revision="changed-revision")
    (tmp_path / "train.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        prepare_data(tmp_path, revision="fixed-revision")
