"""Sample from a LoRA checkpoint supplied through TWINKLE_MODEL_PATH."""

import argparse

if __package__:
    from .common import BASE_URL, load_config
else:
    from common import BASE_URL, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt', default='你是谁', help='User prompt (default: %(default)s).')
    args = parser.parse_args(argv)
    try:
        config = load_config(mode='lora')
    except ValueError as exc:
        parser.error(str(exc))

    from tinker import types
    from twinkle.data_format import Message, Trajectory
    from twinkle.template import Qwen3_5Template
    from twinkle import init_tinker_client

    init_tinker_client()
    from tinker import ServiceClient

    service_client = ServiceClient(base_url=BASE_URL, api_key=config.api_key)
    sampling_client = service_client.create_sampling_client(
        model_path=config.model_path, base_model=config.base_model,
    )
    print(f'Using model {config.base_model}')
    template = Qwen3_5Template(model_id=config.template_model_id, enable_thinking=False)
    trajectory = Trajectory(messages=[
        Message(role='system', content='You are a helpful assistant.'),
        Message(role='user', content=args.prompt),
    ])
    input_feature = template.batch_encode([trajectory], add_generation_prompt=True)[0]
    prompt = types.ModelInput.from_ints(input_feature['input_ids'].tolist())
    params = types.SamplingParams(max_tokens=128, temperature=0.2)

    # Two samples reproduce the original experiment, not a permanent worker constraint.
    print('Sampling...')
    result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=2).result()
    print('Responses:')
    for i, seq in enumerate(result.sequences):
        print(f'{i}: {repr(template.decode(seq.tokens))}')


if __name__ == '__main__':
    main()
