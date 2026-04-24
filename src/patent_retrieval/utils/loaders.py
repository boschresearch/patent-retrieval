# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

from pathlib import Path
import pandas as pd
import os
from pyrootutils import setup_root
import re

root = setup_root(__file__)


def load_topics(path: Path) -> list[str]:
    df = pd.read_csv(
        path, sep="\t", header=None, names=["topic", "candidate", "score"]
    )
    return df.topic.unique().tolist()


def load_topics_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="\t", header=None, names=["topic", "candidate", "score"]
    )
    return df

def load_true_docs(path: Path) -> dict:
    df = pd.read_csv(
        path, sep="\t", header=None, names=["topic", "candidate", "score"]
    )
    return df.groupby('topic')['candidate'].apply(list).to_dict()

def load_retreived_docs(path: Path,k:int=300) -> dict:
    df = pd.read_csv(
        path, sep=",", skiprows=1, names=["topic", "candidate", "score"]
    )
    return df.groupby('topic')['candidate'].apply(lambda x: list(x)[:k]).to_dict()

def get_patent_path(patent_id: Path,path:Path=None) -> dict:

    def _parse_candidate_path(patent_id: str): 

        if patent_id[:2] == 'EP':
            return f"{patent_id[:2]}/{"000000" if patent_id[3] == '0' else "000001"}/{patent_id[4:6]}/{patent_id[6:8]}/{patent_id[8:10]}/{patent_id}*.xml"
        else:
            return f"{patent_id[:2]}/00{patent_id[3:7]}/{patent_id[7:9]}/{patent_id[9:11]}/{patent_id[11:13]}/{patent_id}*.xml"
    try:
        if path is None:
            path = Path(os.getenv("TEST_TOPICS_PATH"))
        topic_glob = list(path.parent.glob(f"PAC_topics/files/{patent_id}*.xml"))

        if topic_glob:
            return topic_glob[0]
            #return dataset.parse_patent([topic_glob[0]])[0]
        
        # then try the document collection dirs (document_collection_dir is a tuple)
        path = Path(os.getenv("DOCUMENT_COLLECTION_DIR")) 
        topic_glob = list(path.glob(_parse_candidate_path(patent_id)))
        return topic_glob[0] if topic_glob else None
        #print(parse_candidate_path(identifier))
        #print(topic_glob)
        #return dataset.parse_patent([topic_glob[0]])[0]
    except Exception as e:
        print(f"Error finding XML for {patent_id}: {e}")
        return None
    
def read_topics(topics_path: Path | int | str) -> list[str]:
    # If input is a number (n), use topics/ntopics.txt
    if isinstance(topics_path, int) or (isinstance(topics_path, str) and topics_path.isdigit()):
        path = Path(root) /"src"/"patent_retrieval"/"dataset"/"topics"/ f"{int(topics_path)}topics.txt"
    else:
        path = Path(topics_path)
    text = path.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]

def read_md_prompt(prompt_id, file_path):

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Find the main block
        pattern = rf'<prompt id="{re.escape(prompt_id)}">(.*?)</prompt>'
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            return None

        body = match.group(1)

        # Extract tags
        sys_match = re.search(r'<system>(.*?)</system>', body, re.DOTALL)
        usr_match = re.search(r'<user>(.*?)</user>', body, re.DOTALL)

        return {
            "system": sys_match.group(1).strip() if sys_match else "",
            "user": usr_match.group(1).strip() if usr_match else ""
        }
    except FileNotFoundError:
        return None