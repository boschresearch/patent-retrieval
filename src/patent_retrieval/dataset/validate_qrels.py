# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

import os
from pathlib import Path

import hydralette as hl
import sqlalchemy as sqla
import sqlmodel as sqlm
from pyrootutils import setup_root
from tqdm import tqdm

from patent_retrieval import utils as utils, dataset as dataset

root = setup_root(__file__)
logger = utils.get_logger(__name__)

cfg = hl.Config(
    test_topics_path=Path(os.environ["CLEF_IP_LOCATION"]) / "02_topics" / "test-pac" / "relass_clef-ip-2011-PAC.txt",
    db_path=Path(os.environ["CLEF_IP_LOCATION"]) / "patents_v4.db",
    valid_topics_path=Path(os.environ["CLEF_IP_LOCATION"]) / "02_topics" / "test-pac" / "relass_clef-ip-2011-PAC_v100.txt",
)

sqlm.SQLModel.metadata.clear()


def main(cfg: hl.Config) -> None:
    """Validate qrels against the cleaned patent database and save valid qrels to a new file."""
    df = utils.load_topics_df(cfg.test_topics_path)

    engine = sqlm.create_engine(f"sqlite:///{cfg.db_path}")

    with sqlm.Session(engine) as session:
        statement = sqlm.select(dataset.Patent.number)
    
        candidates = session.exec(statement).all()
        logger.info(f"Found {len(candidates)} candidates in the database")
    
    filtered_df = df[df['candidate'].isin(candidates)]
    missing_candidates = set(df['candidate']) - set(filtered_df['candidate'])
    logger.info(f"Found {filtered_df['candidate'].nunique()} candidates in the qrels that are present in the database")
    logger.error(f"Not Found - {len(missing_candidates)} Missing candidates: {missing_candidates}")
    # exclude language column if it exists
    filtered_df.iloc[:, :3].to_csv(cfg.valid_topics_path, sep="\t", index=False, header=False)
    logger.info(f"Saved valid qrels to {cfg.valid_topics_path}")

if __name__ == "__main__":
    cfg.apply()
    main(cfg)
