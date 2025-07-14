import json
import os
import random
import time
from collections import defaultdict
from tqdm import tqdm
import requests

# configuration
input_jsonl_file = "political_ideologies_structured_data.jsonl"
input_graph_file = "ideology_influence_graph.json"
tierlist_graph_file = "tierlistgraph.json"
political_coordinates_file = "political_ideologies_coordinates.jsonl"

tier_files = [
    "tier1_ideologies.json",
    "tier2_ideologies.json",
    "tier3_ideologies.json",
    "tier4_ideologies.json"
]
temp_tier_order_file = "temp_tier_order.json"

# gemini api configuration
gemini_api_key = ""
gemini_api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generatecontent"

def get_llm_coordinates(prompt_text, max_retries=5, initial_delay=1):
    # call gemini api for coordinates
    headers = {
        'content-type': 'application/json',
    }
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
        "generationconfig": {
            "responsemimetype": "application/json",
            "responseschema": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "minimum": -100.0, "maximum": 100},
                    "y": {"type": "number", "minimum": -100, "maximum": 100}
                },
                "required": ["x", "y"]
            }
        }
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(f"{gemini_api_url}?key={gemini_api_key}", headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()

            if result and result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
                json_string_from_llm = result["candidates"][0]["content"]["parts"][0]["text"]
                parsed_llm_output = json.loads(json_string_from_llm)
                x_coord = parsed_llm_output.get("x")
                y_coord = parsed_llm_output.get("y")

                if isinstance(x_coord, (int, float)) and isinstance(y_coord, (int, float)):
                    return round(x_coord, 1), round(y_coord, 1)
                else:
                    print(f"warning: llm returned invalid coordinate format: x={x_coord}, y={y_coord}. full output: {parsed_llm_output}")
                    return None, None
            else:
                print(f"warning: llm response structure unexpected: {result}")
                return None, None
        except requests.exceptions.httperror as e:
            if e.response.status_code == 503 and attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"api request error (503 service unavailable): {e}. retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries}).")
                time.sleep(delay)
            else:
                print(f"api request error: {e}")
                return None, None
        except requests.exceptions.requestexception as e:
            print(f"network/connection error: {e}. retrying in {initial_delay} seconds (attempt {attempt + 1}/{max_retries}).")
            time.sleep(initial_delay)
        except json.jsondecodeerror as e:
            print(f"json decode error from llm response: {e}. raw response: {response.text if 'response' in locals() else 'n/a'}")
            return None, None
        except Exception as e:
            print(f"unexpected error during llm call: {e}")
            return None, None
    
    print(f"failed to get valid coordinates after {max_retries} attempts.")
    return None, None

def run_coordinate_rater():
    # load necessary data
    print("loading required data for coordinate rating...")
    try:
        ideology_details_map = {}
        with open(input_jsonl_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="loading descriptions"):
                data = json.loads(line.strip())
                ideology_details_map[data["ideology_name"]] = {
                    "description": data["article_body"],
                    "alignment": data["infobox_details"].get("alignment", "n/a")
                }

        base_graph = {}
        with open(input_graph_file, 'r', encoding='utf-8') as f:
            base_graph = json.load(f)

        tierlist_graph_data = {}
        with open(tierlist_graph_file, 'r', encoding='utf-8') as f:
            tierlist_graph_data = json.load(f)

    except filenotfounderror as e:
        print(f"error: required input file not found: {e.filename}. ensure previous programs were run successfully.")
        return
    except json.jsondecodeerror:
        print(f"error: could not decode json from an input file: {e}. ensure files are valid json.")
        return
    except Exception as e:
        print(f"unexpected error during data loading: {e}")
        return

    # initialize or load political_ideologies_coordinates.jsonl
    coordinates_data = {}
    if os.path.exists(political_coordinates_file):
        print(f"\nloading existing coordinates from '{political_coordinates_file}' to resume progress...")
        try:
            with open(political_coordinates_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    coordinates_data[data["ideology_name"]] = data
            print(f"resuming: {sum(1 for d in coordinates_data.values() if d.get('coordinates') and d['coordinates']['x'] is not None)} ideologies already have coordinates.")
        except json.jsondecodeerror:
            print(f"error: '{political_coordinates_file}' is corrupted. starting fresh for coordinates.")
            coordinates_data = {}
        except Exception as e:
            print(f"unexpected error loading coordinates: {e}. starting fresh.")
            coordinates_data = {}

    if not coordinates_data:
        print("\ninitializing new political_ideologies_coordinates.jsonl...")
        with open(input_jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                ideology_name = data["ideology_name"]
                coordinates_data[ideology_name] = data
                coordinates_data[ideology_name]["coordinates"] = {"x": None, "y": None}

        for fixed_ideology in ["apoliticism", "centrism"]:
            if fixed_ideology in coordinates_data:
                coordinates_data[fixed_ideology]["coordinates"] = {"x": 0.0, "y": 0.0}
                print(f"set '{fixed_ideology}' coordinates to (0.0, 0.0).")

        # save initial state
        with open(political_coordinates_file, 'w', encoding='utf-8') as f:
            for ideology_name in sorted(coordinates_data.keys()):
                f.write(json.dumps(coordinates_data[ideology_name], ensure_ascii=False) + '\n')
        print(f"initial '{political_coordinates_file}' created.")

    # build reverse graph for parent lookup
    reverse_base_graph = defaultdict(set)
    for parent, children in base_graph.items():
        for child in children:
            reverse_base_graph[child].add(parent)

    total_ideologies_to_rate = len(coordinates_data) - sum(1 for fixed_ideology in ["apoliticism", "centrism"] if fixed_ideology in coordinates_data)
    coordinates_assigned_count = sum(1 for d in coordinates_data.values() if d.get('coordinates') and d['coordinates']['x'] is not None and d['ideology_name'] not in ["apoliticism", "centrism"])

    print("\nstarting coordinate assignment loop...")
    progress_bar = tqdm(total=total_ideologies_to_rate, initial=coordinates_assigned_count, desc="coordinate progress")

    for tier_file in tier_files:
        print(f"\n--- processing tier: {tier_file} ---")
        tier_path = tier_file

        tier_ideologies_data = {}
        try:
            with open(tier_path, 'r', encoding='utf-8') as f:
                tier_ideologies_data = json.load(f)
        except filenotfounderror:
            print(f"warning: tier file '{tier_path}' not found. skipping this tier.")
            continue
        except json.jsondecodeerror:
            print(f"error: could not decode json from '{tier_path}'. skipping this tier.")
            continue

        sorted_tier_ideologies_with_ratings = sorted(
            tier_ideologies_data.items(),
            key=lambda item: item[1].get("extremism rating", -1),
            reverse=False
        )

        temp_tier_order_list = [[name, data.get("extremism rating")] for name, data in sorted_tier_ideologies_with_ratings]
        try:
            with open(temp_tier_order_file, 'w', encoding='utf-8') as f:
                json.dump(temp_tier_order_list, f, indent=2, ensure_ascii=False)
            print(f"created '{temp_tier_order_file}' for current tier (with ratings).")
        except Exception as e:
            print(f"error saving '{temp_tier_order_file}': {e}")
            continue

        while True:
            unrated_in_tier = [
                name for name, _ in temp_tier_order_list
                if coordinates_data.get(name, {}).get("coordinates", {}).get("x") is None and name not in ["apoliticism", "centrism"]
            ]

            if not unrated_in_tier:
                print(f"all ideologies in tier '{tier_file}' have been rated. moving to next tier.")
                break

            best_candidate = None
            max_rated_related = -1
            candidate_pool = []

            for node_name in unrated_in_tier:
                if node_name in ["apoliticism", "centrism"]:
                    continue

                related_neighbors = set(reverse_base_graph.get(node_name, []))
                related_neighbors.update(base_graph.get(node_name, []))

                rated_related_count = 0
                for neighbor_name in related_neighbors:
                    if coordinates_data.get(neighbor_name, {}).get("coordinates", {}).get("x") is not None:
                        rated_related_count += 1

                if rated_related_count > max_rated_related:
                    max_rated_related = rated_related_count
                    candidate_pool = [node_name]
                elif rated_related_count == max_rated_related and rated_related_count > 0:
                    candidate_pool.append(node_name)

            if not candidate_pool:
                print(f"no unrated ideologies in tier '{tier_file}' found with rated related ideologies. cannot propagate further in this tier.")
                break

            best_candidate = random.choice(candidate_pool)

            # prepare rag data
            current_ideology_info = ideology_details_map.get(best_candidate, {})
            current_description = current_ideology_info.get("description", "no description available.")
            current_alignment = current_ideology_info.get("alignment", "n/a")
            current_extremism_rating = tierlist_graph_data.get(best_candidate, {}).get("extremism rating")

            related_info_for_llm = []
            current_band_limit = 5
            initial_band_limit = current_band_limit

            while len(related_info_for_llm) <= 3 and current_band_limit <= 100:
                temp_related_info = []

                for potential_related_name, potential_related_data in coordinates_data.items():
                    if potential_related_name == best_candidate:
                        continue

                    potential_related_coords = potential_related_data.get("coordinates")
                    potential_related_extremism_rating = tierlist_graph_data.get(potential_related_name, {}).get("extremism rating")

                    if (potential_related_coords and potential_related_coords["x"] is not None and
                            potential_related_extremism_rating is not None and
                            current_extremism_rating is not None and
                            abs(potential_related_extremism_rating - current_extremism_rating) <= current_band_limit):

                        potential_related_details = ideology_details_map.get(potential_related_name, {})
                        potential_related_description = potential_related_details.get("description", "no description available.")
                        potential_related_alignment = potential_related_details.get("alignment", "n/a")

                        relation_type = []
                        if potential_related_name in reverse_base_graph.get(best_candidate, []):
                            relation_type.append("parent")
                        if best_candidate in reverse_base_graph.get(potential_related_name, []):
                            relation_type.append("child")

                        relation_str = "/".join(relation_type) if relation_type else "general related"

                        temp_related_info.append(
                            f"related ideology ({relation_str}): {potential_related_name}\n"
                            f"description: {potential_related_description}\n"
                            f"alignment: {potential_related_alignment}\n"
                            f"extremism rating: {potential_related_extremism_rating}%\n"
                            f"assigned coordinates: ({potential_related_coords['x']}, {potential_related_coords['y']})\n"
                        )

                related_info_for_llm = temp_related_info

                if len(related_info_for_llm) <= 3:
                    if current_band_limit == 100:
                        break
                    current_band_limit += 1

            if current_band_limit > initial_band_limit and len(related_info_for_llm) > 3:
                print(f"note: for '{best_candidate}', extremism rating search band increased to +/- {current_band_limit}% to find more than 3 related ideologies ({len(related_info_for_llm)} found).")

            if len(related_info_for_llm) <= 3:
                print(f"warning: selected candidate '{best_candidate}' has only {len(related_info_for_llm)} suitable rated related ideologies. search band expanded up to +/- {current_band_limit}% to find them. proceeding with available data.")

            # construct llm prompt
            prompt = (
                "you are an ai specialized in political ideologies and their placement on a political compass. "
                "your task is to assign (x, y) coordinates to a given ideology based on its description, alignment, "
                "and the coordinates of its related ideologies. the coordinates should be in the range [-100.0, 100.0], "
                "with one decimal place. aim for precise values, not just whole numbers, reflecting the nuanced position on the compass.\n\n"
                "political compass axes definitions:\n"
                "- **x-axis (economic/cultural):**\n"
                "  - more market control, cultural, conservative, prone to destructive populism, less equality = higher x value (towards +100.0).\n"
                "  - more equality, more government/state control over economy, least value for culture, least religious influence in politics = lower x value (towards -100.0).\n"
                "- **y-axis (authority):**\n"
                "  - more authoritarian, militant, controlling, centralized power, less concern for individual life/environment = higher y value (towards +100.0).\n"
                "  - more individualistic, stateless, striving for least segregation = lower y value (towards -100.0).\n\n"
                "extremism rating guidance:\n"
                "the 'extremism rating' (0-100%) indicates the *absolute distance* of the (x, y) point from the center (0,0). "
                "it is not a vector. for example, an ideology with a 70% extremism rating should be roughly 70 units away from (0,0) in euclidean distance. "
                "specifically:\n"
                "- ideologies with 0% extremism (apoliticism, centrism) are fixed at (0.0, 0.0).\n"
                "- ideologies with 100% extremism should have at least one coordinate (x or y) at 100.0 or -100.0 (e.g., (100.0, 50.0) or (-70.0, -100.0)).\n\n"
                "analyze the following information:\n\n"
                f"current ideology to rate: {best_candidate}\n"
                f"current ideology description: {current_description}\n"
                f"current ideology alignment: {current_alignment}\n"
                f"current ideology extremism rating: {current_extremism_rating}%\n\n"
                "information about its rated related ideologies (neighbors with similar extremism ratings, providing contextual coordinates):\n"
            )
            if related_info_for_llm:
                prompt += "\n".join(related_info_for_llm)
            else:
                prompt += "no sufficiently close related ideologies with assigned coordinates were found to provide additional context.\n"

            prompt += (
                "\nbased on the descriptions, alignments, and extremism ratings of the current and related ideologies, "
                "determine the (x, y) coordinates for the current ideology. "
                "ensure the coordinates are within [-100.0, 100.0] and rounded to one decimal place. "
                "if the current ideology's extremism rating is 100%, ensure either x or y is 100.0 or -100.0. "
                "output only a json object with 'x' and 'y' keys. example: {\"x\": 25.5, \"y\": -10.2}"
            )

            print(f"\n--- rating coordinates for '{best_candidate}' (llm call {coordinates_assigned_count + 1}/{total_ideologies_to_rate}) ---")
            x_coord, y_coord = get_llm_coordinates(prompt)

            if x_coord is not None and y_coord is not None:
                # apply 100% extremism constraint
                if current_extremism_rating == 100:
                    if abs(x_coord) < 100.0 and abs(y_coord) < 100.0:
                        max_abs = max(abs(x_coord), abs(y_coord))
                        if max_abs > 0:
                            scale_factor = 100.0 / max_abs
                            x_coord = round(x_coord * scale_factor, 1)
                            y_coord = round(y_coord * scale_factor, 1)
                            x_coord = max(-100.0, min(100.0, x_coord))
                            y_coord = max(-100.0, min(100.0, y_coord))
                        else:
                            x_coord = 100.0
                            y_coord = 0.0

                    if abs(x_coord) != 100.0 and abs(y_coord) != 100.0:
                        if abs(x_coord) >= abs(y_coord):
                            x_coord = 100.0 if x_coord > 0 else -100.0
                        else:
                            y_coord = 100.0 if y_coord > 0 else -100.0

                coordinates_data[best_candidate]["coordinates"] = {"x": x_coord, "y": y_coord}
                coordinates_assigned_count += 1
                progress_bar.update(1)

                print(f"assigned coordinates ({x_coord}, {y_coord}) with rating {current_extremism_rating}% to '{best_candidate}'.")

                # dynamically write updated data
                updated_lines = []
                with open(political_coordinates_file, 'r', encoding='utf-8') as f_read:
                    for line in f_read:
                        line_data = json.loads(line.strip())
                        if line_data["ideology_name"] == best_candidate:
                            updated_lines.append(json.dumps(coordinates_data[best_candidate], ensure_ascii=False))
                        else:
                            updated_lines.append(line.strip())

                with open(political_coordinates_file, 'w', encoding='utf-8') as f_write:
                    for line_content in updated_lines:
                        f_write.write(line_content + '\n')

            else:
                print(f"failed to get valid coordinates for '{best_candidate}'. skipping for this run.")
                time.sleep(2)

            time.sleep(0.5)

    progress_bar.close()
    print(f"\ncoordinate assignment process completed for this run. total coordinates assigned: {coordinates_assigned_count}")
    print(f"final '{political_coordinates_file}' saved.")

    # clean up temp file
    if os.path.exists(temp_tier_order_file):
        os.remove(temp_tier_order_file)
        print(f"removed temporary file: '{temp_tier_order_file}'.")

    # final check for all coordinates
    all_coordinates_assigned = True
    for name, data in coordinates_data.items():
        if name not in ["apoliticism", "centrism"] and (data.get("coordinates") is None or data["coordinates"]["x"] is None):
            all_coordinates_assigned = False
            break

    if all_coordinates_assigned:
        print("\nall ideologies have had coordinates assigned. ready for final visualization!")
    else:
        print("\nnot all ideologies have had coordinates assigned yet. run the program again to continue.")

if __name__ == "__main__":
    print("starting program 6: ideology coordinate rater")
    run_coordinate_rater()
