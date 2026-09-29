"""Run the original 40-row legal LoRA SFT experiment explicitly from the CLI."""

import argparse

if __package__:
    from .common import BASE_URL, load_config
else:
    from common import BASE_URL, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        config = load_config(mode='base')
    except ValueError as exc:
        parser.error(str(exc))

    from tqdm import tqdm
    from tinker import types
    from twinkle_client import init_tinker_client
    from twinkle.dataloader import DataLoader
    from twinkle.dataset import Dataset, DatasetMeta
    from twinkle.data_format import Message, Trajectory
    from twinkle.preprocessor import Preprocessor
    from twinkle.server.common import input_feature_to_datum

    class LegalSFTProcessor(Preprocessor):
        """Convert DISC-Law-SFT rows into Twinkle chat trajectories."""

        def __call__(self, rows):
            rows = self.map_col_to_row(rows)
            rows = [self.preprocess(row) for row in rows]
            return self.map_row_to_col(rows)

        def preprocess(self, row):
            return Trajectory(messages=[
                Message(role='system', content='你是一名严谨的中文法律助手。请基于用户给出的事实和问题进行分析。'),
                Message(role='user', content=row['input']),
                Message(role='assistant', content=row['output']),
            ])

    # Use the first 40 input rows; encoding deletes samples longer than 2048 tokens.
    dataset = Dataset(dataset_meta=DatasetMeta(
        'ms://AI-ModelScope/DISC-Law-SFT', subset_name='default', split='train',
        data_slice=range(40),
    ))
    dataset.set_template('Qwen3_5Template', model_id=config.template_model_id,
                         max_length=2048, truncation_strategy='delete')
    dataset.map(LegalSFTProcessor(), load_from_cache_file=False)
    dataset.encode(batched=True, load_from_cache_file=False)
    dataloader = DataLoader(dataset=dataset, batch_size=4)

    # Initialize Tinker client before importing ServiceClient.
    init_tinker_client()
    from tinker import ServiceClient

    service_client = ServiceClient(base_url=BASE_URL, api_key=config.api_key)
    training_client = service_client.create_lora_training_client(base_model=config.base_model, rank=16)

    for epoch in range(1):
        for step, batch in tqdm(enumerate(dataloader)):
            input_datum = [input_feature_to_datum(input_feature) for input_feature in batch]
            fwdbwd_future = training_client.forward_backward(input_datum, 'cross_entropy')
            optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))
            fwdbwd_result = fwdbwd_future.result()
            optim_result = optim_future.result()
            print(f'Epoch {epoch}, step {step}: fwdbwd={fwdbwd_result}, optim={optim_result}')

        result = training_client.save_state(f'twinkle-lora-{epoch}').result()
        print(f'Saved checkpoint for epoch {epoch} to {result.path}')


if __name__ == '__main__':
    main()
