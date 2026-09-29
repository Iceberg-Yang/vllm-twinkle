"""Offline unit tests only: fake SDKs/dotenv, no downloads or remote validation.

Run: python -B -m unittest discover -s /absolute/path/to/examples -p test_examples.py -v
"""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

if __package__:
    from . import common, inference, legal_eval, main as training
else:
    import common
    import inference
    import legal_eval
    import main as training


EXAMPLES = Path(__file__).resolve().parent
TOKEN = 'unit-test-token-not-a-credential'
CHECKPOINT = 'twinkle://unit-test/weights/checkpoint'


def module(name, **members):
    result = ModuleType(name)
    result.__dict__.update(members)
    return result


# Run imports and CLI help in fresh interpreters, with runtime dependencies and
# network/file access blocked. Never open the actual twinkle/.env, even if present.
IMPORT_GUARD = r'''
import importlib
import importlib.abc
import os
from pathlib import Path
import runpy
import sys

root, action, name = sys.argv[1:]
for key in ('MODELSCOPE_TOKEN', 'MODELSCOPE_API_TOKEN', 'TWINKLE_MODEL_PATH'):
    os.environ.pop(key, None)
before = dict(os.environ)

class RejectRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'dotenv', 'tinker', 'twinkle', 'twinkle_client', 'tqdm',
            'torch', 'transformers', 'datasets', 'modelscope',
        }:
            raise AssertionError('Unexpected runtime import: ' + fullname)

def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        if Path(os.fsdecode(args[0])).name == '.env':
            raise AssertionError('Unexpected .env access')
    if event.startswith('socket.'):
        raise AssertionError('Unexpected network access')
    if event in ('os.putenv', 'os.unsetenv'):
        raise AssertionError('Unexpected environment mutation')

sys.meta_path.insert(0, RejectRuntimeImports())
sys.addaudithook(audit)
sys.path.insert(0, root)
if action == 'help':
    sys.argv = [str(Path(root) / (name + '.py')), '--help']
    try:
        runpy.run_path(sys.argv[0], run_name='__main__')
    except SystemExit as exc:
        assert exc.code == 0, exc.code
    else:
        raise AssertionError('--help did not exit')
else:
    if action == 'package':
        sys.path.insert(0, str(Path(root).parent))
        name = 'examples.' + name
    importlib.import_module(name)
assert dict(os.environ) == before
'''


class ImportAndHelpTests(unittest.TestCase):
    def run_guarded(self, action, name):
        result = subprocess.run(
            [sys.executable, '-I', '-B', '-c', IMPORT_GUARD, str(EXAMPLES), action, name],
            capture_output=True, text=True, encoding='utf-8', timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stderr, '')
        return result.stdout

    def test_imports_are_inert_without_runtime_dependencies(self):
        for name in ('common', 'main', 'inference', 'legal_eval'):
            with self.subTest(module=name):
                self.assertEqual(self.run_guarded('import', name), '')

    def test_package_imports_are_also_inert(self):
        for name in ('common', 'main', 'inference', 'legal_eval'):
            with self.subTest(module=name):
                self.assertEqual(self.run_guarded('package', name), '')

    def test_help_succeeds_without_config_or_sdk(self):
        for name in ('main', 'inference', 'legal_eval'):
            with self.subTest(module=name):
                help_text = self.run_guarded('help', name)
                self.assertIn('usage:', help_text)
                if name == 'inference':
                    self.assertIn('--prompt', help_text)
                if name == 'legal_eval':
                    self.assertIn('--mode {base,lora}', help_text)
                    self.assertIn('--output', help_text)

    def test_invalid_cli_arguments_fail_before_loading_config(self):
        cases = [
            (training, ['--unknown']),
            (inference, ['--prompt']),
            (inference, ['--mode', 'base']),
            (legal_eval, []),
            (legal_eval, ['--mode', 'auto']),
            (legal_eval, ['--mode', 'base', '--output']),
        ]
        for target, argv in cases:
            with self.subTest(module=target.__name__, argv=argv):
                with patch.object(target, 'load_config') as load, contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        target.main(argv)
                self.assertEqual(error.exception.code, 2)
                load.assert_not_called()


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(os.environ, {'MODELSCOPE_TOKEN': TOKEN}, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.load_dotenv = Mock()
        sdk_patch = patch.dict(sys.modules, {'dotenv': module('dotenv', load_dotenv=self.load_dotenv)})
        sdk_patch.start()
        self.addCleanup(sdk_patch.stop)

    def test_explicit_dotenv_path_and_shell_precedence(self):
        def fake_load(*, dotenv_path, override):
            # Simulate dotenv's non-overriding behavior without reading a file.
            self.assertFalse(override)
            for key, value in {'MODELSCOPE_TOKEN': 'file-token', 'TWINKLE_MODEL_PATH': CHECKPOINT}.items():
                os.environ.setdefault(key, value)

        self.load_dotenv.side_effect = fake_load
        config = common.load_config(mode='lora')
        self.load_dotenv.assert_called_once_with(dotenv_path=EXAMPLES.parent / '.env', override=False)
        self.assertEqual(config.api_key, TOKEN)
        self.assertEqual(os.environ['MODELSCOPE_API_TOKEN'], TOKEN)
        self.assertEqual(config.model_path, CHECKPOINT)
        self.assertNotIn(TOKEN, repr(config))

    def test_missing_or_blank_token_fails_even_with_api_token(self):
        os.environ['MODELSCOPE_API_TOKEN'] = 'must-not-be-used-as-a-fallback'
        for value in (None, '', '  \t'):
            with self.subTest(value=value):
                os.environ.pop('MODELSCOPE_TOKEN', None)
                if value is not None:
                    os.environ['MODELSCOPE_TOKEN'] = value
                with self.assertRaisesRegex(ValueError, 'MODELSCOPE_TOKEN'):
                    common.load_config(mode='base')

    def test_default_model_and_fixed_https_url(self):
        os.environ['TWINKLE_BASE_URL'] = 'http://ignored.invalid'
        config = common.load_config(mode='base')
        self.assertEqual(config.base_model, 'Qwen/Qwen3.6-27B')
        self.assertEqual(config.template_model_id, 'ms://Qwen/Qwen3.6-27B')
        self.assertEqual(common.BASE_URL, 'https://www.modelscope.cn/twinkle')

    def test_base_model_normalization(self):
        for value in ('Qwen/test-model', 'ms://Qwen/test-model', '  ms://Qwen/test-model  '):
            with self.subTest(value=value):
                os.environ['TWINKLE_BASE_MODEL'] = value
                config = common.load_config(mode='base')
                self.assertEqual(config.base_model, 'Qwen/test-model')
                self.assertEqual(config.template_model_id, 'ms://Qwen/test-model')

    def test_blank_base_model_fails(self):
        for value in ('', '  ', 'ms://'):
            with self.subTest(value=value):
                os.environ['TWINKLE_BASE_MODEL'] = value
                with self.assertRaisesRegex(ValueError, 'TWINKLE_BASE_MODEL'):
                    common.load_config(mode='base')

    def test_base_ignores_checkpoint_loaded_from_dotenv(self):
        self.load_dotenv.side_effect = lambda **kwargs: os.environ.update(TWINKLE_MODEL_PATH=CHECKPOINT)
        config = common.load_config(mode='base')
        self.assertIsNone(config.model_path)
        self.assertEqual(os.environ['TWINKLE_MODEL_PATH'], CHECKPOINT)

    def test_lora_requires_nonblank_checkpoint(self):
        for value in (None, '', '  '):
            with self.subTest(value=value):
                os.environ.pop('TWINKLE_MODEL_PATH', None)
                if value is not None:
                    os.environ['TWINKLE_MODEL_PATH'] = value
                with self.assertRaisesRegex(ValueError, 'TWINKLE_MODEL_PATH'):
                    common.load_config(mode='lora')

    def test_lora_uses_checkpoint_and_maps_token(self):
        os.environ.update(TWINKLE_MODEL_PATH=f' {CHECKPOINT} ', MODELSCOPE_API_TOKEN='old-token')
        config = common.load_config(mode='lora')
        self.assertEqual(config.model_path, CHECKPOINT)
        self.assertEqual(os.environ['MODELSCOPE_API_TOKEN'], TOKEN)

    def test_invalid_mode_fails_without_dotenv(self):
        with self.assertRaisesRegex(ValueError, 'mode'):
            common.load_config(mode='auto')
        self.load_dotenv.assert_not_called()


class RuntimeUnitTests(unittest.TestCase):
    """Exercise CLI orchestration using fake SDKs, not real training/inference."""

    def setUp(self):
        env_patch = patch.dict(os.environ, {'MODELSCOPE_TOKEN': TOKEN}, clear=True)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        self.output = io.StringIO()
        stdout_patch = contextlib.redirect_stdout(self.output)
        stdout_patch.__enter__()
        self.addCleanup(stdout_patch.__exit__, None, None, None)

        self.service = Mock()
        self.service_factory = Mock(return_value=self.service)
        self.sampler = self.service.create_sampling_client.return_value
        self.sampler.sample.return_value.result.return_value = SimpleNamespace(sequences=[
            SimpleNamespace(tokens=[101]), SimpleNamespace(tokens=[202]),
        ])
        self.trainer = self.service.create_lora_training_client.return_value
        self.trainer.forward_backward.return_value.result.return_value = {'loss': 1.0}
        self.trainer.optim_step.return_value.result.return_value = {'updated': True}
        self.trainer.save_state.return_value.result.return_value = SimpleNamespace(path=CHECKPOINT)

        self.template = Mock()
        self.template.batch_encode.return_value = [{'input_ids': Mock(tolist=Mock(return_value=[11, 22]))}]
        self.template.decode.side_effect = lambda tokens: f'response-{tokens[0]}'
        self.template_factory = Mock(return_value=self.template)
        self.types = SimpleNamespace(
            ModelInput=SimpleNamespace(from_ints=Mock(side_effect=lambda ids: ids)),
            SamplingParams=Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs)),
            AdamParams=Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs)),
        )
        tinker = module('tinker', types=self.types)
        # ServiceClient is exposed only after initialization, as required by the SDK.
        self.init = Mock(side_effect=lambda: setattr(tinker, 'ServiceClient', self.service_factory))
        self.dataset = Mock()
        self.dataset_factory = Mock(return_value=self.dataset)
        self.meta_factory = Mock()
        self.loader_factory = Mock(return_value=[['feature-a'], ['feature-b']])
        self.to_datum = Mock(side_effect=lambda feature: ('datum', feature))
        self.load_dotenv = Mock()
        fakes = {
            'dotenv': module('dotenv', load_dotenv=self.load_dotenv),
            'tinker': tinker,
            'twinkle': module('twinkle', init_tinker_client=self.init),
            'twinkle_client': module('twinkle_client', init_tinker_client=self.init),
            'twinkle.data_format': module('twinkle.data_format', Message=SimpleNamespace, Trajectory=SimpleNamespace),
            'twinkle.template': module('twinkle.template', Qwen3_5Template=self.template_factory),
            'twinkle.dataset': module('twinkle.dataset', Dataset=self.dataset_factory, DatasetMeta=self.meta_factory),
            'twinkle.dataloader': module('twinkle.dataloader', DataLoader=self.loader_factory),
            'twinkle.preprocessor': module('twinkle.preprocessor', Preprocessor=type('Preprocessor', (), {})),
            'twinkle.server': module('twinkle.server'),
            'twinkle.server.common': module('twinkle.server.common', input_feature_to_datum=self.to_datum),
            'tqdm': module('tqdm', tqdm=lambda iterable: iterable),
        }
        sdk_patch = patch.dict(sys.modules, fakes)
        sdk_patch.start()
        self.addCleanup(sdk_patch.stop)

    def assert_connection(self):
        self.init.assert_called_once_with()
        self.service_factory.assert_called_once_with(base_url=common.BASE_URL, api_key=TOKEN)
        self.assertEqual(os.environ['MODELSCOPE_API_TOKEN'], TOKEN)
        self.assertNotIn(TOKEN, self.output.getvalue())

    def test_training_preserves_original_flow_and_processor(self):
        training.main([])
        self.assert_connection()
        self.meta_factory.assert_called_once_with(
            'ms://AI-ModelScope/DISC-Law-SFT', subset_name='default', split='train', data_slice=range(40),
        )
        self.dataset_factory.assert_called_once_with(dataset_meta=self.meta_factory.return_value)
        self.dataset.set_template.assert_called_once_with(
            'Qwen3_5Template', model_id='ms://Qwen/Qwen3.6-27B', max_length=2048, truncation_strategy='delete',
        )
        self.assertEqual([call[0] for call in self.dataset.mock_calls], ['set_template', 'map', 'encode'])
        self.assertEqual(self.dataset.map.call_args.kwargs, {'load_from_cache_file': False})
        self.dataset.encode.assert_called_once_with(batched=True, load_from_cache_file=False)
        self.loader_factory.assert_called_once_with(dataset=self.dataset, batch_size=4)
        self.service.create_lora_training_client.assert_called_once_with(base_model='Qwen/Qwen3.6-27B', rank=16)
        self.assertEqual(self.trainer.forward_backward.call_count, 2)
        for index, feature in enumerate(('feature-a', 'feature-b')):
            self.assertEqual(self.trainer.forward_backward.call_args_list[index].args,
                             ([('datum', feature)], 'cross_entropy'))
        self.assertEqual(self.types.AdamParams.call_count, 2)
        for call in self.types.AdamParams.call_args_list:
            self.assertEqual(call.kwargs, {'learning_rate': 1e-4})
        self.assertEqual(self.trainer.optim_step.call_count, 2)
        self.trainer.save_state.assert_called_once_with('twinkle-lora-0')
        for step in (0, 1):
            self.assertIn(f"step {step}: fwdbwd={{'loss': 1.0}}, optim={{'updated': True}}", self.output.getvalue())

        processor = self.dataset.map.call_args.args[0]
        self.assertEqual(type(processor).__name__, 'LegalSFTProcessor')
        row = {'input': '法律问题', 'output': '法律回答'}
        processor.map_col_to_row = Mock(return_value=[row])
        processor.map_row_to_col = Mock(return_value='converted-columns')
        columns = {'input': [row['input']], 'output': [row['output']]}
        self.assertEqual(processor(columns), 'converted-columns')
        processor.map_col_to_row.assert_called_once_with(columns)
        trajectory = processor.map_row_to_col.call_args.args[0][0]
        self.assertEqual([message.role for message in trajectory.messages], ['system', 'user', 'assistant'])
        self.assertEqual([message.content for message in trajectory.messages], [
            '你是一名严谨的中文法律助手。请基于用户给出的事实和问题进行分析。', '法律问题', '法律回答',
        ])

    def run_inference(self, argv, expected_prompt):
        os.environ.update(TWINKLE_MODEL_PATH=CHECKPOINT, TWINKLE_BASE_MODEL='ms://Qwen/test-model')
        inference.main(argv)
        self.assert_connection()
        self.service.create_sampling_client.assert_called_once_with(
            model_path=CHECKPOINT, base_model='Qwen/test-model',
        )
        self.template_factory.assert_called_once_with(model_id='ms://Qwen/test-model', enable_thinking=False)
        trajectory = self.template.batch_encode.call_args.args[0][0]
        self.assertEqual(trajectory.messages[1].content, expected_prompt)
        self.types.SamplingParams.assert_called_once_with(max_tokens=128, temperature=0.2)
        self.assertEqual(self.sampler.sample.call_args.kwargs['num_samples'], 2)
        self.assertIn('response-101', self.output.getvalue())
        self.assertIn('response-202', self.output.getvalue())

    def test_inference_default_prompt(self):
        self.run_inference([], '你是谁')

    def test_inference_custom_prompt(self):
        self.run_inference(['--prompt', '请解释合同效力'], '请解释合同效力')

    def test_missing_lora_path_fails_before_sdk_initialization(self):
        for target, argv in ((inference, []), (legal_eval, ['--mode', 'lora'])):
            with self.subTest(module=target.__name__), contextlib.redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit) as error:
                    target.main(argv)
                self.assertEqual(error.exception.code, 2)
                self.assertIn('TWINKLE_MODEL_PATH', errors.getvalue())
        self.init.assert_not_called()
        self.service_factory.assert_not_called()

    def test_missing_token_fails_for_all_entrypoints(self):
        os.environ.pop('MODELSCOPE_TOKEN')
        for target, argv in ((training, []), (inference, []), (legal_eval, ['--mode', 'base'])):
            with self.subTest(module=target.__name__), contextlib.redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit) as error:
                    target.main(argv)
                self.assertEqual(error.exception.code, 2)
                self.assertIn('MODELSCOPE_TOKEN', errors.getvalue())
        self.init.assert_not_called()
        self.dataset_factory.assert_not_called()

    def run_evaluation(self, mode):
        os.environ['TWINKLE_MODEL_PATH'] = CHECKPOINT
        output_path = EXAMPLES / 'unit-test-output.json'
        # Mock the write as well, so tests create no output files anywhere.
        with patch.object(Path, 'write_text', autospec=True) as write:
            legal_eval.main(['--mode', mode, '--output', str(output_path)])
        self.assert_connection()
        write.assert_called_once()
        self.assertEqual(write.call_args.args[0], output_path)
        self.assertEqual(write.call_args.kwargs, {'encoding': 'utf-8'})
        serialized = write.call_args.args[1]
        report = json.loads(serialized)
        expected_path = CHECKPOINT if mode == 'lora' else None
        self.service.create_sampling_client.assert_called_once_with(
            model_path=expected_path, base_model='Qwen/Qwen3.6-27B',
        )
        self.template_factory.assert_called_once_with(model_id='ms://Qwen/Qwen3.6-27B', enable_thinking=False)
        self.types.SamplingParams.assert_called_once_with(max_tokens=512, temperature=0.2, top_p=0.9)
        self.assertEqual(self.sampler.sample.call_count, 2)
        calls = self.sampler.sample.call_args_list
        self.assertIs(calls[0].kwargs['sampling_params'], calls[1].kwargs['sampling_params'])
        for call in calls:
            self.assertEqual(call.kwargs['num_samples'], 2)
            self.assertEqual(call.kwargs['prompt'], [11, 22])
            self.assertEqual(vars(call.kwargs['sampling_params']),
                             {'max_tokens': 512, 'temperature': 0.2, 'top_p': 0.9})
        self.assertEqual(set(report), {'mode', 'base_model', 'model_path', 'sampling_params', 'cases'})
        self.assertEqual(report['mode'], mode)
        self.assertEqual(report['base_model'], 'Qwen/Qwen3.6-27B')
        self.assertEqual(report['model_path'], expected_path)
        self.assertEqual(report['sampling_params'],
                         {'max_tokens': 512, 'temperature': 0.2, 'top_p': 0.9, 'num_samples': 2})
        self.assertEqual(len(report['cases']), 2)
        for case, saved in zip(legal_eval.eval_cases, report['cases']):
            self.assertEqual(saved, {**case, 'responses': ['response-101', 'response-202']})
            self.assertIn(case['prompt'], self.output.getvalue())
        self.assertNotIn(TOKEN, serialized)
        self.assertNotIn('api_key', serialized)
        self.assertNotIn('seed', serialized)
        self.assertEqual(self.output.getvalue().count('response-101'), 2)
        self.assertEqual(self.output.getvalue().count('response-202'), 2)
        if mode == 'base':
            self.assertNotIn(CHECKPOINT, serialized)
            self.assertNotIn(CHECKPOINT, self.output.getvalue())

    def test_output_parent_is_created_before_sampling(self):
        with patch.object(Path, 'mkdir') as mkdir, patch.object(Path, 'exists', return_value=False), \
                patch.object(Path, 'write_text'):
            legal_eval.main(['--mode', 'base', '--output', 'outputs/new/base.json'])
        mkdir.assert_called_once_with(parents=True, exist_ok=True)
        self.assertEqual(self.sampler.sample.call_count, 2)

    def test_existing_output_fails_before_remote_initialization(self):
        with patch.object(Path, 'mkdir'), patch.object(Path, 'exists', return_value=True), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                legal_eval.main(['--mode', 'base', '--output', 'outputs/base.json'])
        self.assertEqual(error.exception.code, 2)
        self.init.assert_not_called()
        self.service_factory.assert_not_called()

    def test_base_evaluation_ignores_checkpoint_and_saves_all_responses(self):
        self.run_evaluation('base')

    def test_lora_evaluation_uses_checkpoint_and_saves_all_responses(self):
        self.run_evaluation('lora')

    def test_output_is_optional(self):
        with patch.object(Path, 'write_text') as write:
            legal_eval.main(['--mode', 'base'])
        write.assert_not_called()
        self.assertEqual(self.sampler.sample.call_count, 2)


if __name__ == '__main__':
    unittest.main()
