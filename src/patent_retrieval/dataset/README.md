# Dataset Preparation

This directory contains scripts to prepare the CLEF-IP patent dataset.

## Prerequisites

Set the environment variable `CLEF_IP_LOCATION` pointing to the root directory containing the CLEF-IP data:

```bash
export CLEF_IP_LOCATION=/path/to/clef_ip_data
```

Expected structure:
```
$CLEF_IP_LOCATION/
  ├── 01_document_collection/document_collection_pac/     # Raw patent documents
  ├── 02_topics/test-pac/                                  # Test topics & qrels
  └── training-pac/                                        # Training topics
```

## Scripts

### 1. Parse CLEF-IP Dataset
**File:** `parse_clef_ip.py`

Parses XML patent documents and stores them in SQLite database.

**What it does:**
- Reads raw XML documents from configured source
- Extracts patent metadata (number, jurisdiction, dates)
- Parses multilingual content (EN, DE, FR): titles, abstracts, claims, descriptions
- Extracts IPC classification codes
- Stores in structured SQLite database

**Run:**
```bash
python parse_clef_ip.py
```

Configure input/output paths by updating the `cfg` variables in the script.

### 2. Clean Database
**File:** `clean_db.py`

Cleans and validates the parsed database.

**What it does:**
- Copies all patents from input database to output database
- Reassigns patents tagged 'XX' to correct language (EN/DE/FR) based on available content
- Removes patents with no claims in any language
- Removes patents with no abstract in any language

**Run:**
```bash
python clean_db.py
```

Configure input/output database paths by updating the `cfg` variables in the script.

### 3. Validate QRELs
**File:** `validate_qrels.py`

Filters qrels to match patents in the cleaned database.

**What it does:**
- Loads qrels from TSV file
- Checks which candidates are present in the cleaned database
- Logs missing candidates
- Saves filtered qrels with only valid candidates

**Run:**
```bash
python validate_qrels.py
```

Configure database path and input/output qrels paths by updating the `cfg` variables in the script.

## Full Pipeline

```bash
export CLEF_IP_LOCATION=/path/to/clef_ip
cd src/patent_retrieval/dataset

# Step 1: Parse raw documents
python parse_clef_ip.py

# Step 2: Clean database
python clean_db.py

# Step 3: Validate qrels
python validate_qrels.py
```
