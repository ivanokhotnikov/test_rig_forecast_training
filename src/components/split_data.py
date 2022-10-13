from kfp.v2.dsl import Dataset, Input, Output, component


@component(
    base_image='python:3.10-slim',
    packages_to_install=['pandas'],
)
def split_data(
    train_data_size: float,
    processed_data: Input[Dataset],
    train_data: Output[Dataset],
    test_data: Output[Dataset],
) -> None:
    """
    Split processed data into train and test data.
    
    Args:
        train_data_size (float): Train-test split
        processed_data (Input[Dataset]): Processed dataset
        train_data (Output[Dataset]): Train dataset
        test_data (Output[Dataset]): Test dataset
    """
    import pandas as pd

    processed_df = pd.read_csv(
        processed_data.path + '.csv',
        index_col=False,
        header=0,
    )
    train_df = processed_df.loc[:int(len(processed_df) * train_data_size)]
    test_df = processed_df.loc[int(len(processed_df) * train_data_size):]
    train_df.to_csv(
        train_data.path + '.csv',
        index=False,
    )
    test_df.to_csv(
        test_data.path + '.csv',
        index=False,
    )
