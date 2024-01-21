import pandas as pd


def string_to_continuous(income_data: list,
                         classes: list) -> pd.DataFrame:

    """This function wraps model responses into pd.DataFrame format"""

    df = pd.DataFrame(0, columns=classes, index=range(len(income_data)))
    df[classes] = df.apply(lambda x: [1 if col in income_data[x.name] else 0 for col in classes],
                           axis=1,
                           result_type='expand')

    return df
