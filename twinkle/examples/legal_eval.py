"""Compare explicit base/LoRA modes on two legal prompts (not a scored benchmark)."""

import argparse
import json
from pathlib import Path

if __package__:
    from .common import BASE_URL, load_config
else:
    from common import BASE_URL, load_config


eval_cases = [
    {
        'name': 'case_summary',
        'prompt': (
            '请阅读以下案情，并用“案情摘要、争议焦点、分析思路、结论”四个小标题回答：\n'
            '甲向乙借款10万元，双方通过微信约定月利率2%，乙当天转账。到期后甲只归还2万元，'
            '并称双方没有签订纸质借条，所以剩余款项不应继续偿还。乙准备起诉。'
        ),
    },
    {
        'name': 'legal_qa',
        'prompt': (
            '房屋租赁合同到期后，承租人继续居住并按月支付租金，出租人也一直收取。'
            '半年后出租人突然要求承租人三天内搬离。请分析双方之间的法律关系以及承租人可以如何应对。'
        ),
    },
]


def sample_one(sampling_client, template, case, *, sampling_params, num_samples):
    from tinker import types
    from twinkle.data_format import Message, Trajectory

    trajectory = Trajectory(messages=[
        Message(role='system', content='你是一名严谨的中文法律助手。请基于用户给出的事实和问题进行分析。'),
        Message(role='user', content=case['prompt']),
    ])
    input_feature = template.batch_encode([trajectory], add_generation_prompt=True)[0]
    prompt = types.ModelInput.from_ints(input_feature['input_ids'].tolist())
    result = sampling_client.sample(
        prompt=prompt, sampling_params=sampling_params, num_samples=num_samples,
    ).result()
    return [template.decode(seq.tokens) for seq in result.sequences]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('base', 'lora'), required=True,
                        help='base ignores checkpoints; lora requires TWINKLE_MODEL_PATH.')
    parser.add_argument('--output', type=Path, help='Optional JSON output path; parent directories are created.')
    args = parser.parse_args(argv)
    try:
        config = load_config(mode=args.mode)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.output.exists():
                parser.error('Output already exists; choose a new experiment filename.')
        except OSError as exc:
            parser.error(f'Cannot prepare output directory: {exc}')

    from tinker import types
    from twinkle import init_tinker_client
    from twinkle.template import Qwen3_5Template

    init_tinker_client()
    from tinker import ServiceClient

    service_client = ServiceClient(base_url=BASE_URL, api_key=config.api_key)
    sampling_client = service_client.create_sampling_client(
        model_path=config.model_path, base_model=config.base_model,
    )
    template = Qwen3_5Template(model_id=config.template_model_id, enable_thinking=False)
    sampling_params = {'max_tokens': 512, 'temperature': 0.2, 'top_p': 0.9}
    # Two samples match the original experiment, not a permanent worker constraint.
    num_samples = 2
    params = types.SamplingParams(**sampling_params)
    report = {
        'mode': args.mode,
        'base_model': config.base_model,
        'model_path': config.model_path,
        'sampling_params': {**sampling_params, 'num_samples': num_samples},
        'cases': [],
    }

    print(f'Mode: {args.mode}')
    print(f'Base model: {config.base_model}')
    if config.model_path:
        print(f'Model path: {config.model_path}')

    for case in eval_cases:
        print(f'\n===== {case["name"]} =====')
        print(f'Prompt: {case["prompt"]}')
        responses = sample_one(sampling_client, template, case,
                               sampling_params=params, num_samples=num_samples)
        report['cases'].append({'name': case['name'], 'prompt': case['prompt'], 'responses': responses})
        for i, response in enumerate(responses):
            print(f'\n--- response {i} ---')
            print(response)

    if args.output is not None:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'Wrote results to {args.output}')


if __name__ == '__main__':
    main()
