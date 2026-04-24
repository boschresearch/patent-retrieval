# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

import multiprocessing as mp
import os
import queue
from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Generator, Iterable, Literal, TypeVar, overload
from tqdm import tqdm
import hydralette as hl
import sqlalchemy as sqla
import sqlmodel as sqlm
from lxml import etree as ET
from mpire import WorkerPool
from pyrootutils import setup_root
import pandas as pd
from patent_retrieval import dataset as dataset, utils as utils

root = setup_root(__file__)
logger = utils.get_logger(__name__)



cfg = hl.Config(
    document_collection_dir=Path(os.environ["CLEF_IP_LOCATION"]) / "01_document_collection" / "document_collection_pac",
    test_topics_dir=Path(os.environ["CLEF_IP_LOCATION"]) / "02_topics" / "test-pac",
    train_topics_dir=Path(os.environ["CLEF_IP_LOCATION"]) / "training-pac",
    db_path=Path(os.environ["CLEF_IP_LOCATION"]) / "patents_v4.db",
    test_topics_path=Path(os.environ["CLEF_IP_LOCATION"])/ "02_topics"/ "test-pac"/ "relass_clef-ip-2011-PAC_abs.txt",
    output_db_path=Path(os.environ["CLEF_IP_LOCATION"]) / "patents_v4_en.db",
    subset_topics_path=Path(os.environ["CLEF_IP_LOCATION"])/ "02_topics"/ "test-pac"/ "relass_clef-ip-2011-PAC_abs_en.txt"
)

def create_subset_db(cfg: hl.Config) -> None:
    engine = sqlm.create_engine(f"sqlite:///{cfg.db_path}")
    included_patents = []
    with sqlm.Session(engine) as session:
        num_rows = session.exec(
                    sqlm.select(sqlm.func.count()).select_from(dataset.Patent)
                    .where(dataset.Patent.language == "EN") 
                ).one()
        logger.info(f"found EN patents: {num_rows}")
        offset = 0
        batch_size=100000

        while offset<num_rows:
            
            batch = session.exec(
                sqlm.select(dataset.Patent)
                .where(dataset.Patent.language == "EN")
                .offset(offset)
                .limit(batch_size)
            )

            logger.info(offset)
            included_patents.extend(batch)
            offset += batch_size


    logger.info(f"Total EN patents:{len(included_patents)}")

    subset_clean = [p.__class__(**p.dict()) for p in included_patents]

    subset_engine = sqlm.create_engine(f"sqlite:///{cfg.output_db_path}")

    BATCH_SIZE = 10000

    with sqlm.Session(subset_engine) as subset_session:
        num_rows = subset_session.exec(
                    sqlm.select(sqlm.func.count()).select_from(dataset.Patent)
                    .where(dataset.Patent.language == "EN") 
                ).scalar()
        if num_rows == len(subset_clean):
            logger.info("Subset database already up to date, skipping insertion")
            return
        elif num_rows > 0:
            logger.info("Subset database already contains some patents, deleteing existing entries before inserting")
            subset_session.exec(sqlm.delete(dataset.Patent))
            subset_session.commit()
        else:

            logger.info("Inserting patents into subset database")
            for i in tqdm(range(0, len(subset_clean), BATCH_SIZE), desc="Inserting"):
                batch = subset_clean[i:i+BATCH_SIZE]
                subset_session.add_all(batch)
                subset_session.commit()
            return


    logger.info("Writing to db done")
#n=10000 # size of subset
def main(cfg: hl.Config) -> None:
    #create_subset_db(cfg)

    test_topics_df = utils.load_topics_df(cfg.test_topics_path)
    test_topics_df["topic_language"] = test_topics_df['topic'].apply(lambda x: dataset.parse_patent([dataset.find_patent_file(x)])[0].language)
    en_topics_df = test_topics_df[test_topics_df['topic_language'] == "EN"]
    logger.info(f"Found {en_topics_df['topic'].nunique()} English topics")
    engine = sqlm.create_engine(f"sqlite:///{cfg.output_db_path}")

    with sqlm.Session(engine) as session:
        statement = sqlm.select(dataset.Patent.number)
    
        candidates = session.exec(statement).all()
        logger.info(f"Found {len(candidates)} candidates in the subset database")

    filtered_en_topics_df = en_topics_df[en_topics_df['candidate'].isin(candidates)]
    logger.info(f"Found {filtered_en_topics_df['candidate'].nunique()} English candidates in the qrels")
    filtered_en_topics_df.to_csv(cfg.subset_topics_path, sep="\t", index=False, header=False)
    logger.info(f"Saved subset qrels to {cfg.subset_topics_path}")


if __name__ == "__main__":
    cfg.apply()
    main(cfg)
