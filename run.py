import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from person import Customer, Fraudster
from config import n_customers, n_fraudsters, n_sessions, \
    fraud_session_rate, data_output_dir, save_formats
from pathlib import Path


# set seed
seed = 123
random.seed(seed)
np.random.seed(seed)


# generate sessions
def generate_sessions(customers, fraudsters):
    """
    Generate sessions for customers and fraudsters.

    Parameters
    ----------
    customers : list of Customer
    fraudsters : list of Fraudster
    """

    sessions = []
    for i in tqdm(range(n_sessions)):
        # fraud case
        if np.random.uniform() < fraud_session_rate:
            user = np.random.choice(fraudsters)
        # legit case
        else:
            user = np.random.choice(customers)
        session = user.make_session()
        sessions.append(session)
    return sessions


def flatten_sessions(sessions):
    """
    Flatten a list of sessions into a dataframe.

    Parameters
    ----------
    sessions : list of dict
        Each dict is a session.
    """

    dfs = []
    for session in sessions:
        dfs.append(pd.DataFrame.from_records(session))
    return pd.concat(dfs)


def generate_customer_info_table(customers):
    """
    Generate a table of customer info.

    Parameters
    ----------
    customers : list of Customer
    """

    records = [c.make_customer_info_record() for c in customers]
    df = pd.DataFrame.from_records(records)
    return df


def run():
    """
    Generate mock data and save to disk.
    """
    # generate customers (legit and fraud)
    customers = [Customer(i) for i in range(n_customers)]
    fraudsters = [Fraudster() for _ in range(n_fraudsters)]

    # generate sessions
    sessions = generate_sessions(customers, fraudsters)
    # flatten sessions into a dataframe
    df = flatten_sessions(sessions)
    # generate customer info table
    cust_df = generate_customer_info_table(customers)

    # save to disk
    path = Path(data_output_dir)
    path.mkdir(parents=True, exist_ok=True)
    if 'csv' in save_formats:
        # save to csv
        df.to_csv(path / "flat_sessions.csv", index=False)
        cust_df.to_csv(path / "customer_info.csv", index=False)
    if 'json' in save_formats:
        # save to json
        df.to_json(path / "flat_sessions.json", orient="records", lines=True)
        cust_df.to_json(path / "customer_info.json", orient="records", lines=True)


if __name__ == "__main__":
    run()
