# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

from itertools import islice
import multiprocessing as mp
import os
import queue
from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Generator, Iterable, Literal, Optional, TypeVar, overload
import re
import hydralette as hl
import sqlalchemy as sqla
import sqlmodel as sqlm
from lxml import etree as ET
from mpire import WorkerPool
from pyrootutils import setup_root
import patent_retrieval

root = setup_root(__file__)


logger = patent_retrieval.utils.get_logger(__name__)

cfg = hl.Config(
    document_collection_dir=Path(os.environ["CLEF_IP_LOCATION"]) / "01_document_collection" / "document_collection_pac",
    test_topics_dir=Path(os.environ["CLEF_IP_LOCATION"]) / "02_topics" / "test-pac",
    train_topics_dir=Path(os.environ["CLEF_IP_LOCATION"]) / "training-pac",
    db_path=Path(os.environ["CLEF_IP_LOCATION"]) / "patents_clefip.db"
)

sqlm.SQLModel.metadata.clear()

class Patent(sqlm.SQLModel, table=True):
    number: str = sqlm.Field(primary_key=True, index=True)
    kinds: list[str] = sqlm.Field(sa_column=sqlm.Column(sqlm.JSON))
    jurisdiction: str
    application_date: datetime
    publication_date: datetime
    language: str | None
    valid: int | None
    
    title_en: str | None
    title_de: str | None
    title_fr: str | None

    abstract_en: str | None
    abstract_de: str | None
    abstract_fr: str | None

    claims_en: list[str] = sqlm.Field(sa_column=sqlm.Column(sqlm.JSON))
    claims_de: list[str] = sqlm.Field(sa_column=sqlm.Column(sqlm.JSON))
    claims_fr: list[str] = sqlm.Field(sa_column=sqlm.Column(sqlm.JSON))

    description_en: list[str] = sqlm.Field(sa_column=sqlm.Column(sqlm.JSON))
    description_de: list[str] = sqlm.Field(sa_column=sqlm.Column(sqlm.JSON))
    description_fr: list[str] = sqlm.Field(sa_column=sqlm.Column(sqlm.JSON))

    ipc_classes: list[str] = sqlm.Field(sa_column=sqlm.Column(sqlm.JSON))


    def __repr__(self):
        fields = [
            f"{key}={repr(value)[:100]}"
            if not isinstance(value, list)
            else f"{key}={len(value)} items" 
            for key, value in self.model_dump().items()
        ]
        return f"Patent(\n\t{'\n\t'.join(fields)}\n)"
    
    def __str__(self):
        return repr(self)


def get_all_text(element):
    text = element.text or ""
    #print(type(element))
    for child in element:
        text += get_all_text(child)

        text += child.tail or ""
    
    text = re.sub(r'EPO\s*<DP\s+n="[^"]*"\s*/>',"",text)
    return re.sub(r'\s+', ' ', text).strip() if text else ""


T = TypeVar("T")
DT = TypeVar("DT")

@overload
def first_non_none(
    iterable: Iterable[T | None], 
    default_factory: Callable[[], DT] = ...,
    raise_empty: Literal[True] = True
) -> T: ...

@overload
def first_non_none(
    iterable: Iterable[T | None], 
    default_factory: Callable[[], DT] = ...,
    raise_empty: Literal[False] = False
) -> T | DT: ...

def first_non_none(
    iterable: Iterable[T | None], 
    default_factory: Callable[[], DT]=lambda: None,
    raise_empty: bool = False
) -> T | DT:
    try:
        return next((item for item in iterable if item))
    except StopIteration:
        if raise_empty:
            raise
        return default_factory()
    
    
def parse_patent(patent_files: list[Path]) -> tuple[Patent, dict[str, int]]:
    patent_kinds = [file.stem[-2:] for file in patent_files]
    # process document variants by name --> revised version overwrites initial application
    patent_contents = dict(
        sorted(
            {file: file.read_bytes() for file in patent_files}.items(),
            key=lambda item: item[0].name
        )
    )

    trees = []
    for file, content in patent_contents.items():
        try:
            trees.append(ET.fromstring(content))
        except ET.XMLSyntaxError as e:
            logger.exception(f"Failed to parse {file}: {e}")
            continue

    if not trees:
        raise ValueError(f"No valid XML files found in {patent_files}")

    def get_date(tree: ET._Element, app_or_ref: str) -> datetime | None:
        date_elem = tree.find(f'.//{app_or_ref}/document-id/date')
        return (
            datetime.strptime(date_elem.text, '%Y%m%d')
            if date_elem is not None and date_elem.text is not None
            else None
        )
    
    def get_language(tree: ET._Element) -> str | None:
        return tree.get("lang") if tree.get("lang") is not None else None 

    def get_title(tree: ET._Element, lang: str) -> str | None:
        title_elem = tree.find(f'.//invention-title[@lang="{lang}"]')
        return title_elem.text if title_elem is not None else None

    def get_abstract(tree: ET._Element, lang: str) -> str | None:
        abstract_elem = tree.find(f'.//abstract[@lang="{lang}"]')
        return get_all_text(abstract_elem) if abstract_elem is not None else None

    def get_claims(tree: ET._Element, lang: str) -> list[str]:

        
        claims_elems = tree.findall(f'.//claims[@lang="{lang}"]/claim')
        
        if len(claims_elems)==1:

            claims = re.split(r'(?<!\d)\.|(?<=\d)\.(?=\d)', re.sub(r'EPO\s*<DP\s+n="[^"]*"\s*/>',"",get_all_text(claims_elems[0]).replace("\n", " ").strip()))
            claims = [claim.strip()+"."for claim in claims if len(claim.split(" "))>6 ]
            return [re.sub(r'\s+', ' ', claim).strip() for claim in claims if claim is not None]
        else:
            return [re.sub(r'\s+', ' ',get_all_text(claim)).strip() for claim in claims_elems if claim is not None]

    def get_ipc_classes(tree: ET._Element) -> list[str]:
        ipc_elems = tree.findall('.//classification-ipcr')
        return [elem.text.strip() for elem in ipc_elems if elem is not None and elem.text is not None]
            
    def get_description(tree: ET._Element, lang: str) -> list[str]:
        desc_paras = tree.findall(f'.//description[@lang="{lang}"]/p')
        return [get_all_text(para) for para in desc_paras if para is not None]

    patent = Patent(
        number=patent_files[0].stem[:-3],
        kinds=patent_kinds,
        jurisdiction=patent_files[0].stem[:2],
        language=first_non_none(get_language(tree) for tree in trees),
        application_date=first_non_none((get_date(tree, 'application-reference') for tree in trees), raise_empty=True),
        publication_date=first_non_none((get_date(tree, 'publication-reference') for tree in trees), raise_empty=True),
        title_en=first_non_none(get_title(tree, 'EN') for tree in trees),
        title_de=first_non_none(get_title(tree, 'DE') for tree in trees),
        title_fr=first_non_none(get_title(tree, 'FR') for tree in trees),
        abstract_en=first_non_none(get_abstract(tree, 'EN') for tree in trees),
        abstract_de=first_non_none(get_abstract(tree, 'DE') for tree in trees),
        abstract_fr=first_non_none(get_abstract(tree, 'FR') for tree in trees),
        claims_en=first_non_none((get_claims(tree, 'EN') for tree in trees), list),
        claims_de=first_non_none((get_claims(tree, 'DE') for tree in trees), list),
        claims_fr=first_non_none((get_claims(tree, 'FR') for tree in trees), list),
        description_en=first_non_none((get_description(tree, 'EN') for tree in trees), list),
        description_de=first_non_none((get_description(tree, 'DE') for tree in trees), list),
        description_fr=first_non_none((get_description(tree, 'FR') for tree in trees), list),
        ipc_classes=first_non_none((get_ipc_classes(tree) for tree in trees), list),
    )

    stats = {
        f"jurisdiction/{patent.jurisdiction}": 1,
        **{f"kind/{kind}": 1 for kind in patent.kinds},
    }

    if not any([patent.title_en, patent.title_de, patent.title_fr]):
        stats["missing/title"] = 1
    if not any([patent.abstract_en, patent.abstract_de, patent.abstract_fr]):
        stats["missing/abstract"] = 1
    if not any([patent.description_en, patent.description_de, patent.description_fr]):
        stats["missing/description"] = 1
    if not any([patent.claims_en, patent.claims_de, patent.claims_fr]):
        stats["missing/claims"] = 1

    complete_languages = []
    if all([patent.title_en, patent.abstract_en, patent.description_en, patent.claims_en]):
        stats["complete/en"] = 1
        complete_languages.append("en")
    if all([patent.title_de, patent.abstract_de, patent.description_de, patent.claims_de]):
        stats["complete/de"] = 1
        complete_languages.append("de")
    if all([patent.title_fr, patent.abstract_fr, patent.description_fr, patent.claims_fr]):
        stats["complete/fr"] = 1
        complete_languages.append("fr")
    if not complete_languages:
        complete_languages = ["none"]
    stats[f"complete/{','.join(complete_languages)}"] = 1

    return patent, stats

def parse_candidate_path(patent_id: str): 

    if patent_id[:2] == 'EP':
        pad = '000000' if patent_id[3] == '0' else '000001'
        return f"{patent_id[:2]}/{pad}/{patent_id[4:6]}/{patent_id[6:8]}/{patent_id[8:10]}/{patent_id}*.xml"
    else:
        return f"{patent_id[:2]}/00{patent_id[3:7]}/{patent_id[7:9]}/{patent_id[9:11]}/{patent_id[11:13]}/{patent_id}*.xml"

def find_patent_file(identifier: str) -> Optional[Path]:
    # first try the test topics PAC_topics/files (same approach as get_patent)

    topic_glob = list(cfg.test_topics_dir.glob(f"PAC_topics/files/{identifier}*.xml"))

    if topic_glob:
        return topic_glob[0]
    # then try the document collection dirs (document_collection_dir is a tuple)
    try:
        topic_glob = list(cfg.document_collection_dir.glob(parse_candidate_path(identifier)))
        if not topic_glob:
            logger.warning(f"No XML file found for {identifier} in document collection")
            return None
        else:
            return topic_glob[0]
    except Exception as e:
        logger.exception(f"Error finding XML for {identifier}: {e}")
        return None

def get_independent_claims(claims: list[str]) -> list[str]:
    independent_claims = []
    for i,claim in enumerate(claims):

        regex = (
            r"\baccording to\b|"
            r"\bas claimed in\b|"
            r"\bthe method of claim\b|"
            r"(?<!^)\bclaim \d+\b|"       # "claim X" (but not at start)
            r"\bselon l['’]une\b|"        # "selon l'une" (handles straight ' or curly ’ apostrophe)
            r"\bselon la\b|"
            r"(?<!^)\brevendication \d+\b|" # French: "revendication X" (not at start)
            r"\brevendications\b|"    # French: plural
            r"(?<!^)\bAnspruch \d+\b|"    # German: "Anspruch X" (not at start)
            r"\bAnsprüche\b|"         # German: plural
            r"\bnach Anspruch\b"
        )

        if i==0 or not re.search(regex, claim.lower(), re.IGNORECASE):
            independent_claims.append(claim)
            
    return independent_claims
    
def extract_query_text(
    patent: Patent,
    search_columns: list[str],
    kclaims: Optional[int] = None,
    independent_only: bool = False,
    desc_max_tokens: Optional[int] = None,
    # special_chars: list[str] = list(r'\()+^`:{}"[]~!*') + ["'"],
) -> str:

    #def get_attr_text(patent, attr):
      #  return getattr(patent, f"{attr}_{patent.language.lower()}", None) or ""
    def get_attr_text(patent, attr):
        
        value = getattr(patent, f"{attr}_{patent.language.lower()}", None)
        if value==[] or value is None:
            value = getattr(patent, f"{attr}_en", None)

        # If it's a list → return an iterator (generator)
       # if isinstance(value, list):
          #  return (item for item in value)  # generator, not list

        # Otherwise (e.g., string)
        return value if value is not None else ""

    title = (
        f"{'Title: ' + get_attr_text(patent,"title")}\n\n"
        if "title" in search_columns
        else ""
    )
    ipc_classes = (
        f"{'IPC Classes: ' + ', '.join(patent.ipc_classes)}\n\n"
        if "ipc" in search_columns and patent.ipc_classes
        else ""
    )
    abstract = (
        f"{'Abstract: ' + get_attr_text(patent,"abstract")}\n\n"
        if "abstract" in search_columns
        else ""
    )

    list_claims = list(islice(get_attr_text(patent,"claims"), 0, kclaims))
    if independent_only:
        list_claims = get_independent_claims(list_claims)
    #"\n".join(get_attr_text(patent,"claims")[:int(cfg.claims) if cfg.claims.isdigit() else None])
    claims = (
        f"{'Claims: \n -' + "\n -".join(list_claims)}\n\n"
        if "claims" in search_columns
        else ""
    )

    del list_claims

    description = (
        f"{'Description: \n' + " ".join(get_attr_text(patent,"description"))}"
        if "description" in search_columns
        else ""
    )
    if desc_max_tokens is not None and len(description.split()) > desc_max_tokens:
        description = " ".join(description.split()[:desc_max_tokens]) + "..."

    query = title + abstract + claims + description

    # for char in special_chars:
    #     query = query.replace(char, rf"\{char}")
    return query

def db_writer_process(
    write_queue: queue.Queue, 
    db_path: Path, 
    batch_size: int = 10_000
):
    
    def remove_seen(batch: list[Patent]) -> Generator[Patent, None, None]:
        for patent in batch:
            if patent.number in seen_patents:
                logger.warning(f"Patent {patent.number} already seen, skipping.")
                continue
            seen_patents.add(patent.number)
            yield patent

    engine = sqlm.create_engine(f"sqlite:///{db_path}")
    sqlm.SQLModel.metadata.create_all(engine)
    seen_patents = set()

    batch: list[Patent] = []
    with sqlm.Session(engine) as session:
        while True:
            try:
                patent = write_queue.get(timeout=100)
            except (TimeoutError, queue.Empty):
                if batch:
                    session.exec(sqla.insert(Patent), params=remove_seen(batch))  # type: ignore
                    session.commit()
                break

            batch.append(patent)
            if len(batch) >= batch_size:
                session.exec(sqla.insert(Patent), params=remove_seen(batch))  # type: ignore
                session.commit()
                batch = []


def parse_and_save_patent(patent_dir: Path, write_queue: queue.Queue):
    try:
        patent, stats = parse_patent(list(patent_dir.glob("*.xml")))
    except Exception as e:
        logger.error(f"Failed to parse patent in {patent_dir}: {e}")
        return {"error": 1}
    write_queue.put(patent)
    return stats


def main(cfg: hl.Config) -> None:
    pbar = patent_retrieval.utils.RichTableProgress(total=1_768_641, print_every=10_000)
    stats = defaultdict(int)

    mp.set_start_method("spawn", force=True)    
    manager = mp.Manager()
    write_queue = manager.Queue(maxsize=30_000)
    
    writer_process = mp.Process(
        target=db_writer_process, 
        args=(write_queue, cfg.db_path)
    )
    writer_process.start()

    try:
        with WorkerPool(n_jobs=24, start_method="spawn") as pool:
            for stats_ in pool.imap(
                partial(parse_and_save_patent, write_queue=write_queue), 
                cfg.document_collection_dir.glob("*/*/*/*/*"), 
                chunk_size=32
            ):
                for key, value in stats_.items():
                    stats[key] += value
                pbar.update(data=stats)
            pbar.update(data=stats)

    finally:
        writer_process.join()
        manager.shutdown()


if __name__ == "__main__":
    cfg.apply()
    with (
        (root / "dataset" / "clef-ip-parsing_v4_2.log").open("w") as fp,
        patent_retrieval.utils.redirect_stdout_stderr(fp)
    ):
        main(cfg)