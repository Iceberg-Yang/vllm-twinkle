"""Standard-library-only tests: all HTTP, framework and result-file I/O is mocked."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager, nullcontext, redirect_stdout
import importlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, mock_open, patch
import urllib.error

import common
import run_serving_client as serving
import summarize_results as offline_summary
import summarize_serving_results as serving_summary


RUNNERS = (
    "run_nanovllm_offline", "run_minisgl_offline",
    "run_transformers_offline", "run_vllm_offline",
)


def offline_args(**overrides):
    args = dict(model="local-model", num_requests=3, input_len=5, output_len=3,
                seed=7, max_model_len=128, max_num_batched_tokens=None,
                max_num_seqs=None, gpu_memory_utilization=None, dtype=None,
                enforce_eager=False, results_path="unused.jsonl", notes="mock only",
                enable_prefix_caching=False, page_size=16, cuda_graph_max_bs=8,
                cache_type="radix")
    return argparse.Namespace(**(args | overrides))


def serving_args(**overrides):
    args = dict(url="http://127.0.0.1:8000/v1/completions", endpoint="completions",
                model="mock-model", num_requests=3, concurrency=2, request_rate=0.0,
                prompt_words=4, max_tokens=4, temperature=0.6, timeout=2.0,
                no_stream=False, results_path="unused.jsonl", notes="mock only")
    return argparse.Namespace(**(args | overrides))


def sse(*events):
    return "".join("data: " + (event if isinstance(event, str) else json.dumps(event))
                   + "\n\n" for event in events).encode()


def completion(text="", finish=None):
    return {"choices": [{"text": text, "finish_reason": finish}]}


def request_result(index=0, ok=True, source="usage.completion_tokens", latency=1.0):
    return dict(request_id=index, ok=ok, latency_s=latency, queue_s=0.2,
                end_to_end_s=latency + 0.2, ttft_s=0.1 if ok else None,
                tpot_s=0.3 if ok else None, tpot_unit_source=source if ok else None,
                output_units=4 if ok else 0, output_unit_source=source if ok else None,
                error=None if ok else f"error-{index}")


class CommonTests(unittest.TestCase):
    def test_nanovllm_discovery_missing_does_not_change_path(self):
        original = list(sys.path)
        with patch.object(common.importlib.util, "find_spec", return_value=None):
            with self.assertRaisesRegex(ImportError, "not installed"):
                common.add_nanovllm_source_to_path()
        self.assertEqual(sys.path, original)

    def test_nanovllm_discovery_installed_does_not_change_path(self):
        original = list(sys.path)
        with patch.object(common.importlib.util, "find_spec", return_value=object()):
            common.add_nanovllm_source_to_path()
        self.assertEqual(sys.path, original)

    def test_metadata_and_legacy_record_fields(self):
        config = common.config_from_args(offline_args(), "nano-vLLM")
        with patch.object(common, "torch_env", return_value={}), \
             patch.object(common, "installed_versions", return_value={"nanovllm": "mock"}), \
             patch.object(common, "peak_memory_gb", return_value=2.0):
            record = common.build_success_record(config, 15, 9, 1.0)
        self.assertNotIn("host", record)
        self.assertNotIn("hostname", record)
        self.assertTrue(record["env"]["python"])
        self.assertEqual(record["env"]["installed_versions"]["nanovllm"], "mock")
        self.assertEqual(record["memory_unit"], "GiB")
        self.assertEqual(record["memory_scope"], "caller_process_torch_allocator")
        for key in ("engine", "error", "peak_memory_gb", "elapsed_s", "output_tokens_per_s"):
            self.assertIn(key, record)
        self.assertEqual(record["config"]["notes"], "mock only")

    def test_installed_versions_only_use_local_metadata(self):
        def version(name):
            if name == "transformers":
                return "mock-version"
            raise common.metadata.PackageNotFoundError(name)
        with patch.object(common.metadata, "version", side_effect=version):
            self.assertEqual(common.installed_versions(), {"transformers": "mock-version"})

    def test_peak_memory_is_caller_gib(self):
        cuda = SimpleNamespace(is_available=lambda: True,
                               max_memory_allocated=lambda: 3 * 1024**3)
        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}):
            self.assertEqual(common.peak_memory_gb(), 3.0)

    def test_common_validation(self):
        invalid = [dict(num_requests=0), dict(input_len=-1), dict(output_len=0),
                   dict(max_model_len=0), dict(max_num_batched_tokens=0),
                   dict(max_num_seqs=-1), dict(gpu_memory_utilization=0),
                   dict(gpu_memory_utilization=1.1), dict(gpu_memory_utilization=float("nan")),
                   dict(gpu_memory_utilization=float("inf")), dict(max_model_len=7),
                   dict(output_len=1, max_model_len=6)]
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                config = common.config_from_args(offline_args(**overrides), "nano-vLLM")
                with self.assertRaises(ValueError):
                    common.validate_config(config)
        common.validate_config(common.config_from_args(offline_args(max_model_len=8), "vLLM"))

    def test_transformers_rejects_unsupported_settings(self):
        for overrides in (dict(max_num_batched_tokens=100), dict(max_num_seqs=4),
                          dict(gpu_memory_utilization=0.9)):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, "does not support"):
                common.validate_config(common.config_from_args(offline_args(**overrides), "Transformers"))

    def test_nanovllm_dtype_aliases_and_mismatch(self):
        for requested in (None, "auto", "float16", "half", "fp16"):
            self.assertEqual(common.resolve_nanovllm_dtype("torch.float16", requested), "float16")
        with self.assertRaisesRegex(ValueError, "cannot override"):
            common.resolve_nanovllm_dtype("torch.bfloat16", "float16")
        with self.assertRaisesRegex(ValueError, "missing model config dtype"):
            common.resolve_nanovllm_dtype(None, None)

    def test_random_warmup_shapes_and_seed(self):
        measured = common.make_random_token_prompts(3, 5, 97, 7)
        warmup = common.make_random_token_prompts(3, 5, 97, 8)
        self.assertNotEqual(measured, warmup)
        self.assertEqual([len(row) for row in warmup], [5, 5, 5])
        self.assertEqual(measured, common.make_random_token_prompts(3, 5, 97, 7))

    def test_jsonl_writes_full_record_in_memory(self):
        for module in (common, serving):
            with self.subTest(module=module.__name__):
                opened = mock_open()
                with patch.object(Path, "mkdir"), patch.object(Path, "open", opened):
                    module.append_jsonl({"error": "a|b", "notes": "中文"}, "unused.jsonl")
                written = opened().write.call_args.args[0]
                self.assertTrue(written.endswith("\n"))
                self.assertEqual(json.loads(written), {"error": "a|b", "notes": "中文"})


class FakeTensor:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]) if rows else 0)

    def __getitem__(self, slices):
        row_slice, col_slice = slices
        return FakeTensor([row[col_slice] for row in self.rows[row_slice]])

    def numel(self):
        return self.shape[0] * self.shape[1]


@contextmanager
def offline_runtime(module, args, generate_error=None, cleanup_error=None):
    """Only mock known runner interfaces; no real framework is imported."""
    tokenizer = MagicMock()
    tokenizer.__len__.return_value = 97
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer_factory = Mock(from_pretrained=Mock(return_value=tokenizer))
    auto_config = Mock(from_pretrained=Mock(return_value=SimpleNamespace(dtype="torch.bfloat16")))
    llm = MagicMock()
    llm.device = "cuda:0"
    llm.dtype = "torch.bfloat16"
    llm.shutdown.side_effect = cleanup_error
    if module.ENGINE == "Transformers":
        def generate(**kwargs):
            return FakeTensor([row + [0] * kwargs["max_new_tokens"]
                               for row in kwargs["input_ids"].rows])
        llm.generate.side_effect = generate
    else:
        outputs = [{"token_ids": [0] * args.output_len} for _ in range(args.num_requests)]
        if module.ENGINE == "vLLM":
            outputs = [SimpleNamespace(outputs=[SimpleNamespace(token_ids=row["token_ids"])])
                       for row in outputs]
        llm.generate.side_effect = [[], outputs]
    if generate_error is not None:
        llm.generate.side_effect = [[], generate_error]
    llm_factory = Mock(return_value=llm)
    model_factory = Mock(from_pretrained=Mock(return_value=llm))
    torch = SimpleNamespace(bfloat16="torch.bfloat16", float16="torch.float16", long="long",
                            tensor=lambda rows, **kwargs: FakeTensor(rows),
                            ones_like=lambda tensor: tensor, inference_mode=nullcontext)
    modules = {
        "torch": torch,
        "transformers": SimpleNamespace(AutoTokenizer=tokenizer_factory, AutoConfig=auto_config,
                                         AutoModelForCausalLM=model_factory),
        "nanovllm": SimpleNamespace(LLM=llm_factory, SamplingParams=SimpleNamespace),
        "vllm": SimpleNamespace(LLM=llm_factory, SamplingParams=SimpleNamespace),
        "minisgl": SimpleNamespace(),
        "minisgl.core": SimpleNamespace(SamplingParams=SimpleNamespace),
        "minisgl.llm": SimpleNamespace(LLM=llm_factory),
    }
    with ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, modules))
        stack.enter_context(patch.object(module, "parse_args", return_value=args))
        stack.enter_context(patch.object(common, "torch_env", return_value={}))
        stack.enter_context(patch.object(common, "installed_versions", return_value={}))
        stack.enter_context(patch.object(common, "peak_memory_gb", return_value=None))
        stack.enter_context(patch.object(module, "print_summary"))
        if hasattr(module, "add_nanovllm_source_to_path"):
            stack.enter_context(patch.object(module, "add_nanovllm_source_to_path"))
        append = stack.enter_context(patch.object(module, "append_jsonl"))
        sync = stack.enter_context(patch.object(module, "synchronize_cuda"))
        reset = stack.enter_context(patch.object(module, "reset_cuda_stats"))
        events = Mock()
        events.attach_mock(llm.generate, "generate")
        events.attach_mock(sync, "sync")
        events.attach_mock(reset, "reset")
        yield SimpleNamespace(llm=llm, factory=llm_factory, model_factory=model_factory,
                              auto_config=auto_config, append=append, events=events)


class OfflineRunnerTests(unittest.TestCase):
    def test_all_runners_match_warmup_shape_seed_and_decode(self):
        args = offline_args()
        for name in RUNNERS:
            module = importlib.import_module(name)
            with self.subTest(runner=name), offline_runtime(module, args) as runtime:
                module.main()
                warmup, measured = runtime.llm.generate.call_args_list
                if module.ENGINE == "Transformers":
                    warmup_prompts = warmup.kwargs["input_ids"].rows
                    measured_prompts = measured.kwargs["input_ids"].rows
                    self.assertEqual(warmup.kwargs["max_new_tokens"], 2)
                    for call in (warmup, measured):
                        self.assertEqual(call.kwargs["top_k"], 0)
                        self.assertEqual(call.kwargs["top_p"], 1.0)
                        self.assertEqual(call.kwargs["temperature"], 0.6)
                        self.assertIsNone(call.kwargs["eos_token_id"])
                        self.assertEqual(call.kwargs["min_new_tokens"], call.kwargs["max_new_tokens"])
                    self.assertEqual(runtime.model_factory.from_pretrained.call_args.kwargs["device_map"],
                                     {"": "cuda:0"})
                else:
                    warmup_prompts, warmup_sampling = warmup.args[:2]
                    measured_prompts, measured_sampling = measured.args[:2]
                    if module.ENGINE == "vLLM":
                        warmup_prompts = [row["prompt_token_ids"] for row in warmup_prompts]
                        measured_prompts = [row["prompt_token_ids"] for row in measured_prompts]
                    self.assertEqual(warmup_sampling.max_tokens, 2)
                    self.assertTrue(warmup_sampling.ignore_eos)
                    self.assertEqual(warmup_sampling.temperature, 0.6)
                    if isinstance(measured_sampling, list):
                        measured_sampling = measured_sampling[0]
                    self.assertEqual(measured_sampling.max_tokens, args.output_len)
                    self.assertTrue(measured_sampling.ignore_eos)
                self.assertEqual(warmup_prompts, common.make_random_token_prompts(3, 5, 97, 8))
                self.assertEqual(measured_prompts, common.make_random_token_prompts(3, 5, 97, 7))
                self.assertEqual([call[0] for call in runtime.events.mock_calls],
                                 ["generate", "sync", "reset", "generate", "sync"])
                record = runtime.append.call_args.args[0]
                self.assertIsNone(record["error"])
                self.assertEqual(record["input_tokens"], 15)
                self.assertEqual(record["output_tokens"], 9)

    def test_all_runners_persist_runtime_failure_before_exit(self):
        for name in RUNNERS:
            module = importlib.import_module(name)
            with self.subTest(runner=name), offline_runtime(
                module, offline_args(), generate_error=RuntimeError("original failure")
            ) as runtime:
                with self.assertRaises(SystemExit) as raised:
                    module.main()
                self.assertEqual(raised.exception.code, 1)
                runtime.append.assert_called_once()
                record = runtime.append.call_args.args[0]
                self.assertIn("original failure", record["error"])
                self.assertEqual(record["failure_stage"], "benchmark")

    def test_all_runners_persist_validation_failure_before_model_load(self):
        for name in RUNNERS:
            module = importlib.import_module(name)
            with self.subTest(runner=name), offline_runtime(module, offline_args(num_requests=0)) as runtime:
                with self.assertRaises(SystemExit) as raised:
                    module.main()
                self.assertEqual(raised.exception.code, 1)
                self.assertEqual(runtime.append.call_args.args[0]["failure_stage"], "validate_config")
                runtime.factory.assert_not_called()
                runtime.model_factory.from_pretrained.assert_not_called()

    def test_transformers_unsupported_config_is_saved(self):
        module = importlib.import_module("run_transformers_offline")
        for overrides in (dict(max_num_seqs=4), dict(max_num_batched_tokens=128),
                          dict(gpu_memory_utilization=0.8)):
            with self.subTest(overrides=overrides), offline_runtime(module, offline_args(**overrides)) as runtime:
                with self.assertRaises(SystemExit):
                    module.main()
                self.assertIn("does not support", runtime.append.call_args.args[0]["error"])
                runtime.model_factory.from_pretrained.assert_not_called()

    def test_mini_cleanup_does_not_mask_original_error(self):
        module = importlib.import_module("run_minisgl_offline")
        with offline_runtime(module, offline_args(), RuntimeError("original"), RuntimeError("cleanup")) as runtime:
            with self.assertRaises(SystemExit) as raised:
                module.main()
            self.assertEqual(raised.exception.code, 1)
            record = runtime.append.call_args.args[0]
            self.assertIn("original", record["error"])
            self.assertIn("cleanup", record["cleanup_error"])
            self.assertEqual(record["failure_stage"], "benchmark")

    def test_mini_cleanup_only_failure_is_saved(self):
        module = importlib.import_module("run_minisgl_offline")
        with offline_runtime(module, offline_args(), cleanup_error=RuntimeError("cleanup")) as runtime:
            with self.assertRaises(SystemExit):
                module.main()
            record = runtime.append.call_args.args[0]
            self.assertEqual(record["failure_stage"], "cleanup")
            self.assertIn("cleanup", record["cleanup_error"])
            self.assertEqual(record["output_tokens"], 9)

    def test_mini_specific_positive_validation(self):
        module = importlib.import_module("run_minisgl_offline")
        for overrides in (dict(page_size=0), dict(cuda_graph_max_bs=-1)):
            with self.subTest(overrides=overrides), offline_runtime(module, offline_args(**overrides)) as runtime:
                with self.assertRaises(SystemExit):
                    module.main()
                self.assertEqual(runtime.append.call_args.args[0]["failure_stage"], "validate_config")

    def test_nano_dtype_mismatch_fails_before_llm(self):
        module = importlib.import_module("run_nanovllm_offline")
        with offline_runtime(module, offline_args(dtype="float16")) as runtime:
            with self.assertRaises(SystemExit):
                module.main()
            runtime.auto_config.from_pretrained.assert_called_once_with("local-model")
            runtime.factory.assert_not_called()
            self.assertIn("cannot override", runtime.append.call_args.args[0]["error"])

    def test_nano_effective_dtype_is_recorded(self):
        module = importlib.import_module("run_nanovllm_offline")
        with offline_runtime(module, offline_args(dtype="bfloat16")) as runtime:
            module.main()
            self.assertEqual(runtime.append.call_args.args[0]["effective_dtype"], "bfloat16")

    def test_vllm_passes_and_logs_prefix_caching_boolean(self):
        module = importlib.import_module("run_vllm_offline")
        for enabled in (False, True):
            with self.subTest(enabled=enabled), offline_runtime(module, offline_args(enable_prefix_caching=enabled)) as runtime:
                module.main()
                self.assertIs(runtime.factory.call_args.kwargs["enable_prefix_caching"], enabled)
                self.assertIs(runtime.append.call_args.args[0]["enable_prefix_caching"], enabled)


class ServingProtocolTests(unittest.TestCase):
    def run_response(self, body, args=None, content_type="text/event-stream", times=None):
        response = io.BytesIO(body)
        response.headers = {"Content-Type": content_type}
        with ExitStack() as stack:
            opener = stack.enter_context(patch.object(serving.urllib.request, "urlopen", return_value=response))
            if times is not None:
                stack.enter_context(patch.object(serving.time, "perf_counter", side_effect=times))
            result = serving.run_one_request(args or serving_args(), 2, submitted_at=8.0 if times else None)
            self.assertEqual(opener.call_args.kwargs["timeout"], (args or serving_args()).timeout)
            return result

    def test_nonstreaming_chat_reads_message_content(self):
        body = json.dumps({"choices": [{"message": {"content": "hello from chat"}}]}).encode()
        result = self.run_response(body, serving_args(endpoint="chat", no_stream=True), "application/json")
        self.assertTrue(result["ok"])
        self.assertEqual(result["output_units"], 3)
        self.assertEqual(result["output_unit_source"], "whitespace_words")
        self.assertIsNone(result["ttft_s"])
        self.assertIsNone(result["tpot_s"])

    def test_nonstreaming_usage_has_priority(self):
        body = json.dumps({"choices": [{"text": "one word"}], "usage": {"completion_tokens": 7}}).encode()
        result = self.run_response(body, serving_args(no_stream=True), "application/json")
        self.assertEqual(result["output_units"], 7)
        self.assertEqual(result["output_unit_source"], "usage.completion_tokens")

    def test_http200_nonstreaming_error_is_failure(self):
        result = self.run_response(b'{"error":{"message":"quota exceeded"}}',
                                   serving_args(no_stream=True), "application/json")
        self.assertFalse(result["ok"])
        self.assertIn("quota exceeded", result["error"])

    def test_streaming_http200_json_error_is_failure(self):
        result = self.run_response(b'{"error":{"message":"overloaded"}}', content_type="application/json")
        self.assertFalse(result["ok"])
        self.assertIn("overloaded", result["error"])

    def test_sse_error_after_partial_output_is_failure(self):
        result = self.run_response(sse(completion("partial"), {"error": {"message": "GPU failed"}}, "[DONE]"))
        self.assertFalse(result["ok"])
        self.assertIn("GPU failed", result["error"])
        self.assertEqual(result["output_units"], 0)
        self.assertIsNone(result["tpot_s"])

    def test_truncated_sse_is_failure(self):
        result = self.run_response(sse(completion("partial")))
        self.assertFalse(result["ok"])
        self.assertIn("Truncated SSE", result["error"])

    def test_usage_without_terminal_marker_is_still_truncated(self):
        result = self.run_response(sse(completion("partial"), {"usage": {"completion_tokens": 8}}))
        self.assertFalse(result["ok"])

    def test_finish_reason_without_done_is_success(self):
        result = self.run_response(sse(completion("finished", "length")))
        self.assertTrue(result["ok"])
        self.assertEqual(result["output_units"], 1)

    def test_done_without_finish_reason_is_success(self):
        result = self.run_response(sse(completion("finished"), "[DONE]"))
        self.assertTrue(result["ok"])

    def test_empty_stream_fails(self):
        self.assertFalse(self.run_response(b"")["ok"])

    def test_empty_finished_stream_has_no_ttft(self):
        result = self.run_response(sse(completion("", "stop"), "[DONE]"))
        self.assertTrue(result["ok"])
        self.assertIsNone(result["ttft_s"])
        self.assertIsNone(result["tpot_s"])
        self.assertEqual(result["output_units"], 0)

    def test_chat_ttft_skips_role_and_empty_chunks(self):
        body = sse({"choices": [{"delta": {"role": "assistant"}}]},
                   {"choices": [{"delta": {"content": ""}}]},
                   {"choices": [{"delta": {"content": "hello world"}}]},
                   {"choices": [{"delta": {"content": " again"}}]}, "[DONE]")
        result = self.run_response(body, serving_args(endpoint="chat"), times=[10.0, 11.0, 14.0])
        self.assertEqual(result["queue_s"], 2.0)
        self.assertEqual(result["ttft_s"], 1.0)
        self.assertEqual(result["latency_s"], 4.0)
        self.assertEqual(result["end_to_end_s"], 6.0)
        self.assertEqual(result["output_units"], 2)
        self.assertEqual(result["tpot_s"], 3.0)
        self.assertEqual(result["tpot_unit_source"], "stream_chunks")

    def test_stream_usage_overrides_chunk_count(self):
        body = sse(completion("one"), completion(" two", "stop"),
                   {"choices": [], "usage": {"completion_tokens": 5}}, "[DONE]")
        result = self.run_response(body, times=[10.0, 11.0, 14.0])
        self.assertEqual(result["output_units"], 5)
        self.assertEqual(result["tpot_s"], 0.75)
        self.assertEqual(result["tpot_unit_source"], "usage.completion_tokens")

    def test_transport_timeout_and_http_error_are_failures(self):
        errors = [TimeoutError("socket timed out"), urllib.error.URLError("unreachable"),
                  urllib.error.HTTPError("http://127.0.0.1", 500, "failed", {}, None)]
        for error in errors:
            with self.subTest(error=error), patch.object(serving.urllib.request, "urlopen", side_effect=error):
                result = serving.run_one_request(serving_args(), 1)
                self.assertFalse(result["ok"])
                self.assertEqual(repr(error), result["error"])
                self.assertGreaterEqual(result["end_to_end_s"], result["latency_s"])

    def test_malformed_json_and_missing_choices_fail(self):
        for body in (b"not json", b"{}", b"[]"):
            with self.subTest(body=body):
                self.assertFalse(self.run_response(body, serving_args(no_stream=True), "application/json")["ok"])
        self.assertFalse(self.run_response(b"data: not json\n\n")["ok"])

    def test_usage_rejects_negative_and_boolean_counts(self):
        for value in (-1, True, "3", 1.2, None):
            self.assertIsNone(serving.extract_usage_completion_tokens({"usage": {"completion_tokens": value}}))
        self.assertEqual(serving.extract_usage_completion_tokens({"usage": {"completion_tokens": 0}}), 0)


class ServingSummaryAndCLITests(unittest.TestCase):
    def test_failure_statistics_keep_all_errors_and_all_latency(self):
        results = [request_result()] + [request_result(i, False, latency=10.0) for i in range(1, 8)]
        record = serving.summarize(serving_args(num_requests=8), results, 12.0)
        self.assertEqual(record["success_count"], 1)
        self.assertEqual(record["error_count"], 7)
        self.assertEqual(record["success_rate"], 0.125)
        self.assertEqual(len(record["errors"]), 7)
        self.assertEqual(len(record["request_results"]), 8)
        self.assertEqual(record["latency_p95"], 1.0)
        self.assertEqual(record["all_latency_p95"], 10.0)
        self.assertEqual(record["end_to_end_p95"], 10.2)
        self.assertEqual(record["queue_p95"], 0.2)
        self.assertEqual(record["latency_percentile_scope"], "successful_requests_only")
        self.assertEqual(record["timeout_scope"], "socket_per_operation_not_total_deadline")
        self.assertEqual(record["timeout_s"], 2.0)

    def test_mixed_units_are_not_added_or_mixed_in_tpot(self):
        results = [request_result(i, source=source) for i, source in enumerate(
            ("usage.completion_tokens", "stream_chunks", "whitespace_words"))]
        record = serving.summarize(serving_args(), results, 2.0)
        self.assertIsNone(record["output_units"])
        self.assertIsNone(record["output_units_per_s"])
        self.assertIsNone(record["tpot_p95"])
        self.assertEqual(record["output_units_by_source"],
                         {"usage.completion_tokens": 4, "stream_chunks": 4, "whitespace_words": 4})
        self.assertEqual(record["output_units_per_s_by_source"]["stream_chunks"], 2.0)

    def test_homogeneous_units_have_explicit_throughput(self):
        record = serving.summarize(serving_args(num_requests=2), [request_result(0), request_result(1)], 2.0)
        self.assertEqual(record["output_units"], 8)
        self.assertEqual(record["output_units_per_s"], 4.0)
        self.assertEqual(record["tpot_p95"], 0.3)

    def test_all_failed_round_has_no_success_percentiles(self):
        record = serving.summarize(serving_args(num_requests=1), [request_result(0, False)], 2.0)
        self.assertEqual(record["success_rate"], 0.0)
        self.assertIsNone(record["latency_p95"])
        self.assertIsNone(record["ttft_p95"])
        self.assertEqual(record["all_latency_p95"], 1.0)

    def test_serving_parameter_validation(self):
        invalid = [dict(num_requests=0), dict(concurrency=0), dict(prompt_words=0),
                   dict(max_tokens=-1), dict(timeout=0), dict(timeout=float("inf")),
                   dict(request_rate=-1), dict(request_rate=float("nan")),
                   dict(temperature=-1), dict(temperature=float("nan")), dict(url="ftp://unused")]
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                serving.validate_args(serving_args(**overrides))
        serving.validate_args(serving_args(temperature=0, request_rate=0))

    def test_cli_failure_is_written_before_exit_and_submit_time_is_passed(self):
        args = serving_args(num_requests=2)
        def worker(args, index, submitted_at):
            self.assertIsInstance(submitted_at, float)
            return request_result(index, ok=index == 0)
        with patch.object(serving, "parse_args", return_value=args), \
             patch.object(serving, "run_one_request", side_effect=worker), \
             patch.object(serving, "append_jsonl") as append, \
             patch.object(serving, "print_summary"):
            with self.assertRaises(SystemExit) as raised:
                serving.main()
            self.assertEqual(raised.exception.code, 1)
            append.assert_called_once()
            self.assertEqual(append.call_args.args[0]["error_count"], 1)

    def test_cli_invalid_args_saved_without_http(self):
        with patch.object(serving, "parse_args", return_value=serving_args(concurrency=0)), \
             patch.object(serving.urllib.request, "urlopen") as opener, \
             patch.object(serving, "append_jsonl") as append, patch.object(serving, "print_summary"):
            with self.assertRaises(SystemExit) as raised:
                serving.main()
            self.assertEqual(raised.exception.code, 1)
            opener.assert_not_called()
            self.assertEqual(append.call_args.args[0]["failure_stage"], "validate_args")

    def test_cli_success_has_no_failure_exit(self):
        with patch.object(serving, "parse_args", return_value=serving_args(num_requests=1)), \
             patch.object(serving, "run_one_request", return_value=request_result()), \
             patch.object(serving, "append_jsonl") as append, patch.object(serving, "print_summary"):
            serving.main()
            self.assertEqual(append.call_args.args[0]["success_rate"], 1.0)

    def test_timeout_help_is_not_total_deadline(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["run_serving_client.py", "--help"]), redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                serving.parse_args()
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("not a strict total request deadline", " ".join(output.getvalue().split()))


class MarkdownSummaryTests(unittest.TestCase):
    def test_offline_errors_default_filter_and_escape(self):
        records = [{"engine": "failed-engine", "error": "bad|path\nnext", "cleanup_error": "cleanup|error"},
                   {"engine": "good-engine", "error": None, "peak_memory_gb": 2.0}]
        markdown = offline_summary.make_markdown(records)
        self.assertIn("failed-engine", markdown)
        self.assertIn("bad\\|path next", markdown)
        self.assertIn("cleanup\\|error", markdown)
        self.assertIn("caller_GiB", markdown)
        self.assertNotIn("failed-engine", offline_summary.make_markdown(records, only_success=True))
        self.assertNotIn("failed-engine", offline_summary.make_markdown(records, False))

    def test_serving_errors_default_filter_and_scope(self):
        records = [{"model": "bad-round", "error_count": 1, "errors": [{"error": "a|b"}], "notes": "x|y"},
                   {"model": "bad-config", "error_count": 0, "error": "invalid config"},
                   {"model": "good-round", "error_count": 0}]
        markdown = serving_summary.make_markdown(records)
        self.assertIn("bad-round", markdown)
        self.assertIn("invalid config", markdown)
        self.assertIn("a\\|b", markdown)
        self.assertIn("x\\|y", markdown)
        self.assertIn("success-only", markdown)
        self.assertIn("lat_p95_s (all)", markdown)
        filtered = serving_summary.make_markdown(records, only_success=True)
        self.assertNotIn("bad-round", filtered)
        self.assertNotIn("bad-config", filtered)
        self.assertIn("good-round", filtered)

    def test_legacy_mixed_unit_throughput_not_printed(self):
        record = {"output_units_per_s": 123456.0,
                  "output_unit_sources": ["stream_chunks", "usage.completion_tokens"]}
        self.assertNotIn("123456", serving_summary.make_markdown([record]))

    def test_missing_files_fail_instead_of_empty_success(self):
        for module in (offline_summary, serving_summary):
            with self.subTest(module=module.__name__), patch.object(Path, "is_file", return_value=False):
                with self.assertRaisesRegex(FileNotFoundError, "Results file not found"):
                    module.load_records("unused.jsonl")

    def test_empty_or_malformed_files_fail(self):
        for module in (offline_summary, serving_summary):
            for text in ("", "\n", "invalid-json", "[]\n"):
                with self.subTest(module=module.__name__, text=text), \
                     patch.object(Path, "is_file", return_value=True), \
                     patch.object(Path, "open", mock_open(read_data=text)):
                    with self.assertRaises(ValueError):
                        module.load_records("unused.jsonl")

    def test_load_jsonl_retains_failed_records(self):
        records = [{"error": "failed"}, {"error": None}]
        for module in (offline_summary, serving_summary):
            with self.subTest(module=module.__name__), patch.object(Path, "is_file", return_value=True), \
                 patch.object(Path, "open", mock_open(read_data="\n".join(json.dumps(r) for r in records))):
                self.assertEqual(module.load_records("unused.jsonl"), records)

    def test_summary_cli_missing_input_exits_before_output(self):
        args = argparse.Namespace(results_path="unused.jsonl", markdown_path="unused.md", only_success=False)
        for module in (offline_summary, serving_summary):
            output = io.StringIO()
            with self.subTest(module=module.__name__), patch.object(module, "parse_args", return_value=args), \
                 patch.object(Path, "is_file", return_value=False), patch.object(Path, "write_text") as write, \
                 redirect_stdout(output):
                with self.assertRaises(SystemExit) as raised:
                    module.main()
                self.assertIn("Cannot summarize results", str(raised.exception))
                write.assert_not_called()
                self.assertEqual(output.getvalue(), "")

    def test_summary_cli_defaults_and_compatibility_flags(self):
        for module in (offline_summary, serving_summary):
            flags = [[], ["--include-errors"], ["--only-success"]]
            if module is offline_summary:
                flags.append(["--include-failures"])
            for flag in flags:
                output = io.StringIO()
                with self.subTest(module=module.__name__, flag=flag), \
                     patch.object(sys, "argv", ["summary"] + flag), \
                     patch.object(module, "load_records", return_value=[{"engine": "bad", "model": "bad", "error": "known-failure"}]), \
                     redirect_stdout(output):
                    module.main()
                self.assertEqual("known-failure" in output.getvalue(), flag != ["--only-success"])


if __name__ == "__main__":
    unittest.main()
