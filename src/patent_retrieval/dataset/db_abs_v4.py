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
from patent_retrieval import utils as utils, dataset as dataset

root = setup_root(__file__)
logger = utils.get_logger(__name__)



cfg = hl.Config(
    document_collection_dir=Path(os.environ["CLEF_IP_LOCATION"]) / "01_document_collection" / "document_collection_pac",
    test_topics_dir=Path(os.environ["CLEF_IP_LOCATION"]) / "02_topics" / "test-pac",
    train_topics_dir=Path(os.environ["CLEF_IP_LOCATION"]) / "training-pac",
    db_path=Path(os.environ["CLEF_IP_LOCATION"]) / "patents_v3.db"
)

engine = sqlm.create_engine(f"sqlite:///{cfg.db_path}")
included_patents = []
with sqlm.Session(engine) as session:
    num_rows = session.exec(
                sqlm.select(sqlm.func.count()).select_from(dataset.Patent)
                .where(
                    sqlm.or_(
                        dataset.Patent.abstract_de != None,
                        dataset.Patent.abstract_en != None,
                        dataset.Patent.abstract_fr != None
                    )
                ) 
            ).one()
    logger.info(f"found {num_rows} patents without abstract in DE, EN, FR")
    offset = 0
    batch_size=100000

    while offset<num_rows:
        
        batch = session.exec(
            sqlm.select(dataset.Patent)
            .where(
                sqlm.or_(
                        dataset.Patent.abstract_de != None,
                        dataset.Patent.abstract_en != None,
                        dataset.Patent.abstract_fr != None
                    )
            )
            .offset(offset)
            .limit(batch_size)
        ).all()

        logger.info(offset)
        included_patents.extend(batch)
        offset += batch_size


logger.info(f"Total patents:{len(included_patents)}")

subset_clean = [p.__class__(**p.dict()) for p in included_patents]


subset_engine = sqlm.create_engine(f"sqlite:///{cfg.db_path}"[:-5]+f"v4.db")
sqlm.SQLModel.metadata.create_all(subset_engine)

BATCH_SIZE = 10000

with sqlm.Session(subset_engine) as subset_session:
    for i in tqdm(range(0, len(subset_clean), BATCH_SIZE), desc="Inserting"):
        batch = subset_clean[i:i+BATCH_SIZE]
        subset_session.add_all(batch)
        subset_session.commit()


logger.info("Writing to db done")