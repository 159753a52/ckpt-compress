import subprocess
import sys
import unittest


class TestExperimentLibraryInitialization(unittest.TestCase):
    def test_package_import_is_lightweight_and_public_exports_remain_lazy(self) -> None:
        code = """
import sys
import experiments.lib as lib
assert 'torch' not in sys.modules
assert 'experiments.lib.models' not in sys.modules
assert 'experiments.lib.data' not in sys.modules
assert callable(lib.get_model_type)
assert lib.get_model_type('gpt2-medium') == 'gpt2'
assert 'experiments.lib.models' in sys.modules
print('ok')
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(completed.stdout.strip(), "ok")

    def test_paper_result_public_api_stays_on_the_store_module(self) -> None:
        from experiments.lib import paper_results

        self.assertEqual(
            paper_results.__all__,
            ["JobResultStore", "SCHEMA_VERSION", "SuiteResultStore", "records_from"],
        )


if __name__ == "__main__":
    unittest.main()
