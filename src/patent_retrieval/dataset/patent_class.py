# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

import os
from pathlib import Path
from tqdm import tqdm
import sqlalchemy as sqla
import sqlmodel as sqlm
from lxml import etree as ET
from mpire import WorkerPool
from pyrootutils import setup_root
import pandas as pd
import re
from patent_retrieval import utils as utils, dataset as dataset


from typing import Optional, Set
print("Setting up environment...")
root = setup_root(".")
db_path=Path(os.environ["CLEF_IP_LOCATION"]) / "clef_ip2011_cleaned.db"
test_topics_path=Path(os.environ["CLEF_IP_LOCATION"])/ "02_topics"/ "test-pac"/ "relass_clef-ip-2011-PAC_abs.txt"
document_collection_dir=Path(os.environ["CLEF_IP_LOCATION"]) / "01_document_collection" / "document_collection_pac"

# build maps of patent id -> set of "first two words" ipcr classes for topics and documents

def parse_candidate_path(patent_id: str): 

    if patent_id[:2] == 'EP':
        return f"{patent_id[:2]}/{"000000" if patent_id[3] == '0' else "000001"}/{patent_id[4:6]}/{patent_id[6:8]}/{patent_id[8:10]}/{patent_id}*.xml"
    else:
        return f"{patent_id[:2]}/00{patent_id[3:7]}/{patent_id[7:9]}/{patent_id[9:11]}/{patent_id[11:13]}/{patent_id}*.xml"


def find_patent_file(identifier: str) -> Optional[Path]:
    # first try the test topics PAC_topics/files (same approach as get_patent)

    topic_glob = list(test_topics_path.parent.glob(f"PAC_topics/files/{identifier}*.xml"))

    if topic_glob:
        return topic_glob[0]
    # then try the document collection dirs (document_collection_dir is a tuple)

    topic_glob = list(document_collection_dir.glob(parse_candidate_path(identifier)))
    #print(parse_candidate_path(identifier))
    #print(topic_glob)
    return topic_glob[0]

def extract_ipcr_set(p: Path) -> Set[str]:
    try:
        content = p.read_bytes()
        tree = ET.fromstring(content)
    except Exception:
        return set()
    ipcr_texts = [ (e.text or "").strip() for e in tree.findall(".//classification-ipcr") ]
    # normalize whitespace and keep only first two tokens of each classification string
    result = set()
    for t in ipcr_texts:
        if not t:
            continue
        norm = ' '.join(t.split())
        parts = norm.split()
        #key = " ".join(parts[:1]) if parts else ""
        #key = norm.split()[0][:-1]
        key = " ".join(parts[:2]) if parts else ""
        if key:
            result.add(key)
    return result
print("Testing topic file resolution...")
df = pd.read_csv("/home/alm3rng/patent-retrieval/embeddings/runs/patQwen3-emb-4b-v2_db-v4_all-topics_abstract-claims_aysm_top1000/results.csv")
# group by topic and keep up to 500 unique 'number' values per topic
df = df.groupby('topic').head(500).reset_index(drop=True)
# unique ids to resolve
topic_ids = pd.Index(df['topic'].unique())
doc_ids = pd.Index(df['number'].unique())

def process_patent_id(patent_id: str) -> tuple[str, Set[str]]:
    """Helper function to process a single patent ID."""
    f = dataset.find_patent_file(patent_id)
    if f is None:
        return (patent_id, set())
    return (patent_id, extract_ipcr_set(f))

# resolve topics using multiprocessing
with WorkerPool(n_jobs=os.cpu_count()) as pool:
    topic_results = list(tqdm(
        pool.imap(process_patent_id, topic_ids),
        total=len(topic_ids),
        desc="resolve topic files"
    ))
true_dict = utils.load_true_docs(test_topics_path)
topic_ipcrs = dict(topic_results)
missing_topics = [tid for tid, ipcrs in topic_results if not ipcrs]



def parse_ipc_classes_field(ipc_field) -> Set[str]:
    """Parse an `ipc_classes` DB field and return a set of first-two-token keys."""
    if ipc_field is None:
        return set()
    
    result = set()
    print("Parsing IPC classes field...type is", type(ipc_field))
    for p in ipc_field:
        p = (p or "").strip()
        if not p:
            continue
        norm = " ".join(p.split())
        tokens = norm.split()
        key = " ".join(tokens[:2]) if tokens else ""
        if key:
            result.add(key)
    return result

print("Resolving document IPC classes from database...")
doc_ipcrs = {}
batch_size = 10000
doc_id_list = list(doc_ids)
engine = sqlm.create_engine(f"sqlite:///{db_path}")
with sqlm.Session(engine) as session:
    for i in range(0, len(doc_id_list), batch_size):
        batch = doc_id_list[i : i + batch_size]
        statement = sqlm.select(dataset.Patent.number, dataset.Patent.ipc_classes).where(dataset.Patent.number.in_(batch))
        rows = session.exec(statement).all()
        for number, ipc in rows:
            doc_ipcrs[str(number)] = parse_ipc_classes_field(ipc)

# Ensure all requested doc_ids exist in the mapping (missing -> empty set)
for did in doc_id_list:
    doc_ipcrs.setdefault(str(did), set())

missing_docs = [did for did, ipcrs in doc_ipcrs.items() if not ipcrs]

print(f"missing topic files: {len(missing_topics)}, missing doc files: {len(missing_docs)}")

# map ipcr sets back onto the dataframe and compute common classes
df['topic_ipcrs'] = df['topic'].map(topic_ipcrs)
df['doc_ipcrs'] = df['number'].map(doc_ipcrs)


df['common_ipcrs'] = df.apply(
    lambda r: (r['topic_ipcrs'] or set()).intersection(r['doc_ipcrs'] or set()),
    axis=1
)

df['n_common_ipcrs'] = df['common_ipcrs'].apply(len)

df['tp'] = df.apply(
    lambda r: r['number'] in true_dict.get(r['topic'], []),
    axis=1,
)

df.to_csv("patent_classification_analysis_B00X 000.csv", index=False)