# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

from pathlib import Path
import os

import streamlit as st
import sqlmodel as sqlm

from patent_retrieval import utils, dataset, encoder

os.environ["HTTP_PROXY"] = "http://rb-proxy-sl.bosch.com:8080"
os.environ["HTTPS_PROXY"] = "http://rb-proxy-sl.bosch.com:8080"

MODEL_NAME = "patQwen3-emb-4b-v2"
TOKENIZER_PATH = "/home/alm3rng/patent-retrieval/finetuning/runs/optuna/trial_11/checkpoint-95"
BASE_URL = "http://localhost:59749/v1"
TYPE = "dense"
BACKEND = "openai"
STORE_TYPE = "faiss"

st.title("Patent Search")

if 'encoder' not in st.session_state:
    st.session_state.encoder = None

# Sidebar for configuration
st.sidebar.header("Configuration")
index_path = st.sidebar.text_input("FAISS Index Path", "")

db_path = os.environ.get("CLEF_IP_LOCATION", "") + "/patents_v4.db"
test_topics_path = Path(os.environ["CLEF_IP_LOCATION"]) / "02_topics" / "test-pac" / "relass_clef-ip-2011-PAC_abs.txt"

if st.sidebar.button("Load Index"):
    try:
        st.session_state.encoder = encoder.get_encoder(
            type=TYPE, backend=BACKEND, store_type=STORE_TYPE,
            model_name=MODEL_NAME, tokenizer=TOKENIZER_PATH,
            base_url=BASE_URL, index_dir=index_path,
        )
        st.session_state.encoder.load_index(path=index_path)
        st.sidebar.success(f"Loaded {len(st.session_state.encoder.get_indices())} patents")
    except Exception as e:
        st.sidebar.error(f"Error loading index: {e}, index_path: {index_path}")


def prepend_instruct(query: str = "") -> str:
    return f"Instruct: Given a patent, perform a prior art search and identify relevant existing patents. \nQuery:{query}"

if st.session_state.encoder is not None:
    search_mode = st.radio("Search by:", ["Query", "Patent ID"])
    search_columns = ["title", "abstract", "claims", "description"]
    if search_mode == "Query":
        query = st.text_area("Enter your query:", height=100)
        patent_id = st.text_input("Enter a patent ID:")
    else:
        patent_id = st.text_input("Enter a patent ID:")
        search_columns = st.multiselect(
            "Select search columns:",
            search_columns,
            default=["title", "abstract"],
        )
        query = None

    k = st.slider("Number of results:", min_value=1, max_value=1000, value=10)

    if st.button("Search"):
        true_dict = utils.load_true_docs(test_topics_path)
        evaluate = patent_id in true_dict

        if query:
            with st.spinner("Searching..."):
                results = st.session_state.encoder.search(prepend_instruct(query=query), k=k, text=True)

                if evaluate:
                    relevant_docs = true_dict.get(patent_id, [])
                    tp_count = sum(1 for patent, score in results if patent.metadata['number'] in relevant_docs)
                    st.success(f"Found {tp_count} out of {len(relevant_docs)} in top {k} results.")
                    recall = tp_count / len(relevant_docs) if relevant_docs else 0
                    st.info(f"Recall@{k}: {recall:.4f}")

                st.subheader(f"Top {k} Results:")
                for i, (patent, score) in enumerate(results):
                    tp = evaluate and patent.metadata['number'] in true_dict[patent_id]
                    expander_label = f"Doc {i+1}. Patent ID: {patent.metadata['number']} - Similarity Score: {score:.4f}"
                    if tp:
                        expander_label += " ✅"
                    with st.expander(expander_label):
                        st.write(f"**Patent:** \n {patent.page_content}")

        elif patent_id:
            with st.spinner("Searching by Patent ID..."):
                parts = patent_id.split("-")
                if len(parts) == 3:
                    patent_text = dataset.extract_query_text(
                        dataset.parse_patent([dataset.find_patent_file(patent_id)])[0],
                        search_columns=search_columns,
                    )
                elif len(parts) == 2:
                    engine = sqlm.create_engine(f"sqlite:///{db_path}")
                    with sqlm.Session(engine) as session:
                        patent = session.exec(
                            sqlm.select(dataset.Patent).where(dataset.Patent.number == patent_id)
                        ).first()
                        patent_text = dataset.extract_query_text(patent, search_columns=search_columns) if patent else ""
                else:
                    patent_text = ""

                if patent_text:
                    st.write("**Query Patent Text:**")
                    st.text_area("Patent content being searched:", patent_text, height=200, disabled=True)
                    results = st.session_state.encoder.search(prepend_instruct(query=patent_text), k=k, text=True)

                    if evaluate:
                        relevant_docs = true_dict.get(patent_id, [])
                        tp_count = sum(1 for patent, score in results if patent.metadata['number'] in relevant_docs)
                        st.success(f"Found {tp_count} out of {len(relevant_docs)} in top {k} results.")
                        recall = tp_count / len(relevant_docs) if relevant_docs else 0
                        st.info(f"Recall@{k}: {recall:.4f}")

                    st.subheader(f"Top {k} Results for Patent ID {patent_id}:")
                    for i, (patent, score) in enumerate(results):
                        tp = evaluate and patent.metadata['number'] in true_dict[patent_id]
                        expander_label = f"Doc {i+1}. Patent ID: {patent.metadata['number']} - Similarity Score: {score:.4f}"
                        if tp:
                            expander_label += " ✅"
                        with st.expander(expander_label):
                            st.write(f"**Patent:** \n {patent.page_content}")
                else:
                    st.error(f"Patent ID {patent_id} not found in dataset.")
        else:
            st.warning("Please enter a query")
else:
    st.info("Please load a FAISS index from the sidebar")
