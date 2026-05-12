import unittest

from open_instruct.data_loader import _merge_env_config
from open_instruct.data_types import EnvConfig, EnvConfigEntry


def _entry(name: str, **kwargs: object) -> EnvConfigEntry:
    return EnvConfigEntry(env_name=name, is_text_env=False, kwargs=kwargs)


def _base(*entries: EnvConfigEntry, max_steps: int = 10) -> EnvConfig:
    return EnvConfig(max_steps=max_steps, env_configs={e.env_name: e for e in entries})


class TestMergeEnvConfig(unittest.TestCase):
    def test_none_sample_returns_base(self):
        base = _base(_entry("counter"))
        self.assertIs(_merge_env_config(base, None), base)

    def test_sample_merges_kwargs(self):
        merged = _merge_env_config(_base(_entry("counter")), {"env_configs": [{"env_name": "counter", "target": 11}]})
        self.assertEqual(merged.env_configs["counter"].kwargs, {"target": 11})

    def test_sample_overrides_max_steps(self):
        merged = _merge_env_config(_base(_entry("counter"), max_steps=10), {"max_steps": 3})
        self.assertEqual(merged.max_steps, 3)

    def test_base_kwargs_preserved_on_merge(self):
        base = _base(_entry("counter", difficulty="hard"))
        merged = _merge_env_config(base, {"env_configs": [{"env_name": "counter", "target": 7}]})
        self.assertEqual(merged.env_configs["counter"].kwargs, {"difficulty": "hard", "target": 7})

    def test_sample_env_name_matches_call_name_alias(self):
        base = _base(_entry("ScholarSearch"), _entry("GeneralSearch"), _entry("Fetch"))
        merged = _merge_env_config(
            base,
            {
                "env_configs": [
                    {"env_name": "scholar_search", "last_time": "2024-01-01"},
                    {"env_name": "general_search", "curr_title": "test"},
                    {"env_name": "fetch", "last_time": "2023-01-01"},
                ]
            },
        )

        self.assertEqual(set(merged.env_configs), {"ScholarSearch", "GeneralSearch", "Fetch"})
        self.assertEqual(merged.env_configs["ScholarSearch"].kwargs, {"last_time": "2024-01-01"})
        self.assertEqual(merged.env_configs["GeneralSearch"].kwargs, {"curr_title": "test"})
        self.assertEqual(merged.env_configs["Fetch"].kwargs, {"last_time": "2023-01-01"})

    def test_sample_env_without_name_raises(self):
        with self.assertRaises(KeyError):
            _merge_env_config(_base(_entry("counter")), {"env_configs": [{"target": 3}]})


if __name__ == "__main__":
    unittest.main()
