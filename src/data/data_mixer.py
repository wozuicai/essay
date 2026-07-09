"""
Data mixing utilities for Phase 3 (data mixture ratio experiments).
"""

from datasets import Dataset, concatenate_datasets


def create_mixed_dataset(
    english_data: Dataset,
    target_lang_data: Dataset,
    english_ratio: float,
    total_size: int = 4000,
    seed: int = 42,
) -> Dataset:
    """
    Create a mixed dataset with specified English/target-language ratio.

    Args:
        english_data: English SFT dataset
        target_lang_data: Target language SFT dataset
        english_ratio: Fraction of English samples (0.0 to 1.0)
        total_size: Total number of samples in the mixed dataset
        seed: Random seed for reproducibility

    Returns:
        Shuffled mixed dataset
    """
    assert 0.0 < english_ratio < 1.0, "english_ratio must be strictly between 0 and 1"

    n_english = int(total_size * english_ratio)
    n_target = total_size - n_english

    en_sample = english_data.shuffle(seed=seed).select(range(min(n_english, len(english_data))))
    tgt_sample = target_lang_data.shuffle(seed=seed).select(range(min(n_target, len(target_lang_data))))

    mixed = concatenate_datasets([en_sample, tgt_sample])
    mixed = mixed.shuffle(seed=seed)

    print(f"Mixed dataset: {len(en_sample)} English + {len(tgt_sample)} target = {len(mixed)} total "
          f"(requested ratio={english_ratio:.1%})")
    return mixed
