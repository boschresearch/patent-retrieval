# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

import os
from pathlib import Path

import hydralette as hl
import sqlalchemy as sqla
import sqlmodel as sqlm
from pyrootutils import setup_root
from tqdm import tqdm

from patent_retrieval import utils, dataset

root = setup_root(__file__)
logger = utils.get_logger(__name__)

cfg = hl.Config(
    db_path=Path(os.environ["CLEF_IP_LOCATION"]) / "clef_ip2011.db",
    output_db_path=Path(os.environ["CLEF_IP_LOCATION"]) / "clef_ip2011_cleaned.db",
)

sqlm.SQLModel.metadata.clear()


def reassign_language_tags(session: sqlm.Session) -> tuple[int, int]:
    """Reassign language for patents tagged 'XX' based on available content.
    
    If all EN columns are populated -> 'EN', else try DE, then FR.
    Patents with no complete language set are deleted.
    
    Returns (updated_count, deleted_count).
    """
    results = session.exec(
        sqlm.select(dataset.Patent).where(dataset.Patent.language == "XX")
    ).all()
    logger.info(f"Found {len(results)} patents with language 'XX'")

    updated = 0
    deleted = 0

    for patent in tqdm(results, desc="Reassigning language tags"):
        if patent.title_en and patent.abstract_en and patent.claims_en and patent.description_en:
            patent.language = "EN"
            updated += 1
            session.add(patent)
        elif patent.title_de and patent.abstract_de and patent.claims_de and patent.description_de:
            patent.language = "DE"
            updated += 1
            session.add(patent)
        elif patent.title_fr and patent.abstract_fr and patent.claims_fr and patent.description_fr:
            patent.language = "FR"
            updated += 1
            session.add(patent)
        else:
            session.delete(patent)
            deleted += 1

    session.commit()
    return updated, deleted


def drop_no_claims(session: sqlm.Session) -> int:
    """Delete patents where claims are missing in all languages.
    
    Returns number of deleted rows.
    """
    results = session.exec(
        sqlm.select(dataset.Patent).where(
            sqla.and_(
                sqla.or_(dataset.Patent.claims_en == None, dataset.Patent.claims_en == []),
                sqla.or_(dataset.Patent.claims_de == None, dataset.Patent.claims_de == []),
                sqla.or_(dataset.Patent.claims_fr == None, dataset.Patent.claims_fr == []),
            )
        )
    ).all()

    for patent in tqdm(results, desc="Dropping patents with no claims"):
        session.delete(patent)

    session.commit()
    return len(results)


def drop_no_abstract(session: sqlm.Session) -> int:
    """Delete patents where abstract is missing in all languages.
    
    Returns number of deleted rows.
    """
    results = session.exec(
        sqlm.select(dataset.Patent).where(
            sqla.and_(
                sqla.or_(dataset.Patent.abstract_en == None, dataset.Patent.abstract_en == ""),
                sqla.or_(dataset.Patent.abstract_de == None, dataset.Patent.abstract_de == ""),
                sqla.or_(dataset.Patent.abstract_fr == None, dataset.Patent.abstract_fr == ""),
            )
        )
    ).all()

    for patent in tqdm(results, desc="Dropping patents with no abstract"):
        session.delete(patent)

    session.commit()
    return len(results)


def copy_db(input_engine: sqla.Engine, output_engine: sqla.Engine, batch_size: int = 10_000) -> int:
    """Copy all patents from input DB to output DB. Returns total count."""
    sqlm.SQLModel.metadata.create_all(output_engine, tables=[dataset.Patent.__table__])

    with sqlm.Session(input_engine) as src_session:
        total = src_session.exec(
            sqlm.select(sqlm.func.count()).select_from(dataset.Patent)
        ).one()
        logger.info(f"Source DB has {total} patents")

        offset = 0
        copied = 0
        while offset < total:
            batch = src_session.exec(
                sqlm.select(dataset.Patent).offset(offset).limit(batch_size)
            ).all()
            # detach from source session
            detached = [p.__class__(**p.model_dump()) for p in batch]

            with sqlm.Session(output_engine) as dst_session:
                dst_session.add_all(detached)
                dst_session.commit()

            copied += len(batch)
            offset += batch_size
            logger.info(f"Copied {copied}/{total} patents")

    return copied


def main(cfg: hl.Config) -> None:
    input_engine = sqlm.create_engine(f"sqlite:///{cfg.db_path}")
    output_engine = sqlm.create_engine(f"sqlite:///{cfg.output_db_path}")

    # Step 1: Copy source DB to new output DB
    logger.info(f"Copying {cfg.db_path} -> {cfg.output_db_path}")
    total = copy_db(input_engine, output_engine)
    logger.info(f"Copied {total} patents to output DB")

    # Step 2: Clean the output DB
    with sqlm.Session(output_engine) as session:
        logger.info("Step 1/3: Reassigning language tags...")
        updated, deleted = reassign_language_tags(session)
        logger.info(f"  -> Updated {updated}, deleted {deleted} patents")

        logger.info("Step 2/3: Dropping patents with no claims...")
        n_no_claims = drop_no_claims(session)
        logger.info(f"  -> Deleted {n_no_claims} patents")

        logger.info("Step 3/3: Dropping patents with no abstract...")
        n_no_abstract = drop_no_abstract(session)
        logger.info(f"  -> Deleted {n_no_abstract} patents")

    # Final count
    with sqlm.Session(output_engine) as session:
        final_count = session.exec(
            sqlm.select(sqlm.func.count()).select_from(dataset.Patent)
        ).one()
    logger.info(f"Done. Output DB has {final_count} patents (started with {total})")


if __name__ == "__main__":
    cfg.apply()
    main(cfg)
