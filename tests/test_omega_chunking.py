import unittest
from unittest.mock import patch

from hmr_backends.omega.runner import (
    build_chunks,
    resolve_chunk_size,
    run_omega_camera_recovery,
)


class OmegaChunkingTests(unittest.TestCase):
    def test_numeric_cli_value(self):
        from run import build_arg_parser

        args = build_arg_parser().parse_args(
            ['--input', 'unused', '--omega_chunk_size', '16']
        )
        self.assertEqual(resolve_chunk_size(args.omega_chunk_size, 1800), 16)

    def test_auto_and_integer_values(self):
        self.assertEqual(resolve_chunk_size('auto', 1800), 500)
        self.assertEqual(resolve_chunk_size('auto', 12), 12)
        self.assertEqual(resolve_chunk_size(16, 1800), 16)
        self.assertEqual(resolve_chunk_size('32', 12), 12)

    def test_invalid_values(self):
        for value in ('invalid', '1.5', '0', '-1', 0, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_chunk_size(value, 1800)

    def test_full_sequence_coverage(self):
        chunks = build_chunks(1800, 16, 8)
        written = []
        for index, (start, end) in enumerate(chunks):
            self.assertLessEqual(end - start, 16)
            if index:
                self.assertEqual(chunks[index - 1][1] - start, 8)
            written.extend(range(start if index == 0 else start + 8, end))
        self.assertEqual(written, list(range(1800)))

    def test_invalid_options_fail_before_model_load(self):
        with patch('hmr_backends.omega.runner.load_omega_model') as load_model:
            with self.assertRaises(ValueError):
                run_omega_camera_recovery(
                    ['unused.jpg'] * 20,
                    '/tmp/egohandkit-invalid-options',
                    chunk_size='8',
                    overlap=8,
                    force=True,
                )
            load_model.assert_not_called()


if __name__ == '__main__':
    unittest.main()
