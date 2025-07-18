# Plot-Political-Compass

This repository contains Python programs for building a political compass prediction system. It covers data acquisition, AI-assisted data labeling, deep learning model fine-tuning, and user Input visualization.

### Repository Structure and File Descriptions

* **`data.py`**:
    * **Purpose**: Acquires raw political ideology data.
    * **Method**: Web scrapes `polcompball.wikitide.org`. Extracts ideology names, main article text, and key infobox details (e.g., influences).
    * **Output**: `political_ideologies_structured_data.jsonl` (raw text content).

* **`ratings.py`**:
    * **Purpose**: Quantifies ideology extremism.
    * **Method**: Uses gemini api to assign an "extremism rating" (0-100%) to each ideology. An "extremism rating" is a scalar value indicating an ideology's deviation from centrism. It leverages predefined ratings and contextual information from related ideologies in an influence graph (`ideology_influence_graph.json`) for context (RAG). Prioritizes rating ideologies with more already-rated neighbors, simulating human expert reasoning through graph-based active learning.
    * **Input**: `political_ideologies_structured_data.jsonl`, `ideology_influence_graph.json`.
    * **Output**: `tierlistgraph.json` (ideologies with extremism ratings), `tier1_ideologies.json`, `tier2_ideologies.json`, `tier3_ideologies.json`, `tier4_ideologies.json` (ideologies categorized by extremism tiers).

* **`coordinates.py`**:
    * **Purpose**: Assigns precise political compass (x, y) coordinates.
    * **Method**: Employs gemini api for coordinate assignment. Uses RAG, providing the llm with the ideology's description, alignment, extremism rating, and coordinates of nearby (by extremism rating) related ideologies. RAG grounds the llm's Output in existing data, improving accuracy and consistency. It enforces coordinate boundaries `[-100.0, 100.0]` and ensures 100% extremism ratings result in at least one coordinate at `+/-100.0`.
    * **Input**: `political_ideologies_structured_data.jsonl`, `ideology_influence_graph.json`, `tierlistgraph.json`, `tier[1-4]_ideologies.json`.
    * **Output**: `political_ideologies_coordinates.jsonl` (final dataset with assigned coordinates).
    * **Temporary File**: `temp_tier_order.json` (for processing order).

* **`train_distilbert.py`**:
    * **Purpose**: Fine-tunes a deep learning model for text-to-coordinate regression.
    * **Method**: Fine-tunes `distilbert-base-uncased`, a transformer-based neural network. distilbert is a smaller, efficient bert-like model chosen for its balance of semantic understanding and computational efficiency. It learns to map ideology text descriptions (`article_body`) to their assigned (x, y) coordinates. Training uses `num_train_epochs=20`, `max_length=512`, `fp16` for memory efficiency, and checkpointing for resume capability. it performs a regression task, directly Outputting two numerical values.
    * **Input**: `political_ideologies_coordinates.jsonl`.
    * **Output**: `distilbert_checkpoints/final_model/` (contains the fine-tuned distilbert model and tokenizer).

* **`plot.py`**:
    * **Purpose**: Predicts user Input coordinates and visualizes them on a political compass.
    * **Method**: Loads the fine-tuned distilbert model. For user-provided text, it generates a semantic embedding (vector representation) using the fine-tuned model. It then applies k-nearest neighbors (KNN) to find the `k` most semantically similar known ideologies (from `political_ideologies_coordinates.jsonl`) in the embedding space. KNN ensures predictions are grounded within the established political compass space. The user's predicted coordinates are a weighted average of these `k` neighbors' actual coordinates.
    * **Input**: `political_ideologies_coordinates.jsonl`, `distilbert_checkpoints/final_model/`.
    * **Output**: `user_compass_prediction.html` (interactive html visualization).

* **`requirements.txt`**: Lists all python package dependencies.

### Data Files (Generated by Programs)

* **`political_ideologies_structured_data.jsonl`**: Raw scraped data, including ideology names, article bodies, and infobox details. (generated by `data.py`).
* **`ideology_influence_graph.json`**: Graph data representing influence relationships between ideologies. this graph is crucial for `ratings.py` and `coordinates.py` To propagate information from known to unknown nodes. (generated by a prior, unlisted program; Input for `ratings.py` and `coordinates.py`).
* **`tierlistgraph.json`**: Mapping of ideologies to their assigned extremism ratings. (generated by `ratings.py`).
* **`tier[1-4]_ideologies.json`**: Individual json files for ideologies categorized by extremism tier. (generated by `ratings.py`).
* **`political_ideologies_coordinates.jsonl`**: The core dataset containing each ideology's name, article body, infobox details, and its final (x, y) political compass coordinates. (generated by `coordinates.py`).
* **`concept_manifestos.jsonl`**: llm-generated affirmative and negative manifestos for fine-grained political concepts. (generated by `programx_concept_manifesto_generator.py`, a separate script used for question design, not included in this repository).

### Missing Files (Not Included in Repository due to Size)

* `distilbert_checkpoints/`: This directory, containing the fine-tuned distilbert model and tokenizer, is generated by `train_distilbert.py` during training. It is typically too large for direct repository inclusion.

### Training `train_distilbert.py` in Google Colab

`train_distilbert.py` was developed and primarily executed in a google colab environment. This is due to the necessity of **gpu access** for efficient transformer model fine-tuning, which colab provides free of charge. Local machines often lack the required gpu vram.

To run this project in google colab:
1.  **Upload Files**: place all `.py` and `.jsonl`/`.json` files (excluding `distilbert_checkpoints/`) into a google drive folder.
2.  **Mount Google Drive**: in your colab notebook, execute `from google.colab import drive; drive.mount('/content/drive')`.
3.  **Ddjust `base_path`**: modify the `base_path` variable in each script to `base_path = "/content/drive/MyDrive/YourPoliticalCompassDataFolder/"`.
4.  **Install Dependencies**: run `!pip install -r requirements.txt` in a colab cell.
5.  **Execute Scripts**: run programs sequentially (e.g., `!python data.py`). for `ratings.py` and `coordinates.py`, re-run the cell if api rate limits are encountered. `train_distilbert.py` will automatically resume training from checkpoints if the colab session disconnects.


### Install Requirements:
   ```python
   pip install -r requirements.txt
   ```
