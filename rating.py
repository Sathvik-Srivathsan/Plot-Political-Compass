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

predefined_ratings = {
    "ingsoc": 100, "hive-mind collectivism": 100, "anarcho-nihilism": 100,
    "social darwinism": 100, "communalism": 100,
    "anarcho-egoism": 100, "anarcho-jazzism": 100, "anti-realism": 100,
    "avaritionism": 100, "benefactorism": 100, "death worship": 100,
    "fordism": 100, "fully automated gay space communism": 100,
    "ismism": 100, "kraterocracy": 100, "neo-bolshevism": 100,
    "ochlocracy": 100, "post-humanism": 100, "primalism": 100,
    "senatorialism": 100, "soulism": 100, "illegalism": 100, "illuminatism": 100,
    "necrocracy": 95,
    "posadism": 90,
    "naziism": 80, "esoteric fascism": 80, "national bolshevism": 80, "jewish-nazism": 80,
    "maoism": 78, "juche": 78,
    "strasserism": 75, "fascism": 75,
    "islamic theocracy": 70, "anarcho-transhumanism": 70, "pinochetism": 70,
    "anarcho-capitalism": 60, "anarcho-communism": 60, "trotskyism": 60,
    "integralism": 55,
    "transhumanism": 50, "geolibertarianism": 50,
    "longism": 40,
    "mercantilism": 35,
    "social democracy": 30, "constitutional monarchism": 30,
    "georgism": 20, "technocracy": 20,
    "anti-authoritarianism": 10,
    "anti-extremeism": 5, "neoliberalism": 5, "horseshoe centrism": 5, "religious rejectionism": 5,
    "apoliticism": 0, "centrism": 0
}

# gemini api configuration
gemini_api_key = ""
gemini_api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generatecontent"

def get_llm_rating(prompt_text, max_retries=5, initial_delay=1):
    # call gemini api for rating
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
                    "rating": {"type": "number", "minimum": 0, "maximum": 100}
                },
                "required": ["rating"]
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
                rating = parsed_llm_output.get("rating")

                if isinstance(rating, (int, float)) and 0 <= rating <= 100:
                    return int(round(rating))
                else:
                    print(f"warning: llm returned invalid rating format: {rating}. full output: {parsed_llm_output}")
                    return None
            else:
                print(f"warning: llm response structure unexpected: {result}")
                return None
        except requests.exceptions.httperror as e:
            if e.response.status_code == 503 and attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"api request error (503 service unavailable): {e}. retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries}).")
                time.sleep(delay)
            else:
                print(f"api request error: {e}")
                return None
        except requests.exceptions.requestexception as e:
            print(f"network/connection error: {e}. retrying in {initial_delay} seconds (attempt {attempt + 1}/{max_retries}).")
            time.sleep(initial_delay)
        except json.jsondecodeerror as e:
            print(f"json decode error from llm response: {e}. raw response: {response.text if 'response' in locals() else 'n/a'}")
            return None
        except Exception as e:
            print(f"unexpected error during llm call: {e}")
            return None
    
    print(f"failed to get a valid rating after {max_retries} attempts.")
    return None

def classify_and_save_tiers(tierlist_graph_data, base_graph):
    # classify and save tiers
    print("\nclassifying ideologies into tiers and saving subgraphs...")

    tier_definitions = {
        "tier1": {"min": 0, "max": 24, "data": defaultdict(dict), "filename": "tier1_ideologies.json"},
        "tier2": {"min": 25, "max": 49, "data": defaultdict(dict), "filename": "tier2_ideologies.json"},
        "tier3": {"min": 50, "max": 74, "data": defaultdict(dict), "filename": "tier3_ideologies.json"},
        "tier4": {"min": 75, "max": 100, "data": defaultdict(dict), "filename": "tier4_ideologies.json"},
    }

    # map ideology to tier
    ideology_to_tier_map = {}
    for ideology_name, details in tierlist_graph_data.items():
        rating = details.get("extremism rating")
        if rating is None:
            print(f"warning: ideology '{ideology_name}' has no rating, skipping tier classification.")
            continue

        for tier_name, tier_info in tier_definitions.items():
            if tier_info["min"] <= rating <= tier_info["max"]:
                ideology_to_tier_map[tier_name].setdefault(ideology_name, {"extremism rating": rating, "influences": []})
                ideology_to_tier_map[ideology_name] = tier_name
                break

    # populate tier-specific graphs
    for ideology_name, details in tierlist_graph_data.items():
        current_tier_name = ideology_to_tier_map.get(ideology_name)
        if not current_tier_name:
            continue

        current_tier_data_dict = tier_definitions[current_tier_name]["data"]

        current_tier_data_dict[ideology_name]["extremism rating"] = details["extremism rating"]
        current_tier_data_dict[ideology_name]["influences"] = []

        original_influences_from_base = base_graph.get(ideology_name, [])
        for influenced_ideology in original_influences_from_base:
            if ideology_to_tier_map.get(influenced_ideology) == current_tier_name:
                current_tier_data_dict[ideology_name]["influences"].append(influenced_ideology)

        current_tier_data_dict[ideology_name]["influences"].sort()

    # save each tier's graph
    for tier_name, tier_info in tier_definitions.items():
        output_filename = tier_info["filename"]
        tier_graph = dict(sorted(tier_info["data"].items()))

        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(tier_graph, f, indent=2, ensure_ascii=False)
            print(f"saved {len(tier_graph)} ideologies to '{output_filename}'")
        except Exception as e:
            print(f"error saving '{output_filename}': {e}")


def run_extremism_rater():
    # load base graph
    print(f"loading base graph from '{input_graph_file}'...")
    try:
        with open(input_graph_file, 'r', encoding='utf-8') as f:
            base_graph = json.load(f)
    except filenotfounderror:
        print(f"error: base graph file '{input_graph_file}' not found.")
        print("ensure program 2 was run successfully.")
        return
    except json.jsondecodeerror:
        print(f"error: could not decode json from '{input_graph_file}'.")
        return
    except Exception as e:
        print(f"unexpected error loading base graph: {e}")
        return

    # load ideology descriptions
    print(f"loading ideology descriptions from '{input_jsonl_file}'...")
    ideology_details_map = {}
    try:
        with open(input_jsonl_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="loading descriptions"):
                data = json.loads(line.strip())
                ideology_details_map[data["ideology_name"]] = data["article_body"]
    except filenotfounderror:
        print(f"error: descriptions file '{input_jsonl_file}' not found.")
        print("ensure program 1 was run successfully.")
        return
    except json.jsondecodeerror:
        print(f"error: could not decode json from '{input_jsonl_file}'.")
        return
    except Exception as e:
        print(f"unexpected error loading descriptions: {e}")
        return

    # initialize or load tierlist graph
    tierlist_graph_data = {}
    rated_ideologies_count = 0

    if os.path.exists(tierlist_graph_file):
        print(f"\nloading existing tierlist graph from '{tierlist_graph_file}' to resume progress...")
        try:
            with open(tierlist_graph_file, 'r', encoding='utf-8') as f:
                tierlist_graph_data = json.load(f)
            for node_data in tierlist_graph_data.values():
                if node_data.get("extremism rating") is not None:
                    rated_ideologies_count += 1
            print(f"resuming: {rated_ideologies_count} ideologies already rated.")
        except json.jsondecodeerror:
            print(f"error: '{tierlist_graph_file}' is corrupted. starting fresh.")
            tierlist_graph_data = {}
        except Exception as e:
            print(f"unexpected error loading tierlist graph: {e}. starting fresh.")
            tierlist_graph_data = {}

    if not tierlist_graph_data:
        print("\ninitializing new tierlist graph...")
        for ideology_name in base_graph.keys():
            tierlist_graph_data[ideology_name] = {
                "extremism rating": None,
                "influences": base_graph[ideology_name]
            }

        # apply predefined ratings
        for ideology, rating in predefined_ratings.items():
            if ideology in tierlist_graph_data:
                tierlist_graph_data[ideology]["extremism rating"] = rating
                rated_ideologies_count += 1
        print(f"initialized with {rated_ideologies_count} predefined ratings.")
        try:
            with open(tierlist_graph_file, 'w', encoding='utf-8') as f:
                json.dump(tierlist_graph_data, f, indent=2, ensure_ascii=False)
            print(f"initial '{tierlist_graph_file}' saved.")
        except Exception as e:
            print(f"error saving initial '{tierlist_graph_file}': {e}")

    # build reverse graph
    reverse_graph = defaultdict(set)
    for parent, data in tierlist_graph_data.items():
        for child in data.get("influences", []):
            reverse_graph[child].add(parent)

    total_ideologies = len(tierlist_graph_data)

    print("\nstarting rating assignment loop...")
    progress_bar = tqdm(total=total_ideologies, initial=rated_ideologies_count, desc="rating progress")

    while True:
        unrated_nodes = [
            name for name, data in tierlist_graph_data.items()
            if data.get("extremism rating") is None
        ]

        if not unrated_nodes:
            print("\nall ideologies have been rated. stopping.")
            break

        best_candidate = None
        max_rated_related = -1
        candidate_pool = []

        for node_name in unrated_nodes:
            parents = reverse_graph.get(node_name, [])
            children = base_graph.get(node_name, [])

            rated_related_count = 0

            unique_related_neighbors = set()

            for parent in parents:
                unique_related_neighbors.add(parent)
            for child in children:
                unique_related_neighbors.add(child)

            for related_node in unique_related_neighbors:
                if tierlist_graph_data.get(related_node, {}).get("extremism rating") is not None:
                    rated_related_count += 1

            if rated_related_count > max_rated_related:
                max_rated_related = rated_related_count
                candidate_pool = [node_name]
            elif rated_related_count == max_rated_related and rated_related_count > 0:
                candidate_pool.append(node_name)

        if not candidate_pool:
            print("\nno unrated ideologies found with rated related ideologies. cannot propagate further. stopping.")
            break

        best_candidate = random.choice(candidate_pool)

        # prepare rag data
        current_node_description = ideology_details_map.get(best_candidate, "no description available.")

        related_info_for_llm = []

        all_neighbors = set(reverse_graph.get(best_candidate, []))
        all_neighbors.update(base_graph.get(best_candidate, []))

        for neighbor_name in all_neighbors:
            neighbor_rating = tierlist_graph_data.get(neighbor_name, {}).get("extremism rating")
            if neighbor_rating is not None:
                neighbor_description = ideology_details_map.get(neighbor_name, "no description available.")

                is_parent = neighbor_name in reverse_graph.get(best_candidate, [])
                is_child = neighbor_name in base_graph.get(best_candidate, [])

                relation_type = []
                if is_parent:
                    relation_type.append("parent")
                if is_child:
                    relation_type.append("child")

                relation_str = "/".join(relation_type) if relation_type else "related"

                related_info_for_llm.append(
                    f"related ideology ({relation_str}): {neighbor_name}\n"
                    f"description: {neighbor_description}\n"
                    f"extremism rating: {neighbor_rating}%\n"
                )

        if not related_info_for_llm:
            print(f"warning: selected candidate '{best_candidate}' has no *currently* rated related ideologies. skipping for this iteration.")
            continue

        # llm prompt
        prompt = (
            "you are an ai specialized in political ideologies. your task is to assign an 'extremism rating' "
            "from 0 to 100 to a given ideology. the rating should reflect how far the ideology is from centrism, "
            "regardless of direction (left or right, authoritarian or libertarian). "
            "here are the precise definitions for the boundaries:\n"
            "- a rating of 0% is reserved only for 'apoliticism' and 'centrism'. any other ideology must have a rating > 0%. so that means an ideology with low rating implies it closer to the ideology of centrism.\n"
            "- a rating of 100% represents an 'off-the-compass' or purely theoretical extreme, rarely achievable by realistic ideologies. so the higher the rating the more extreme it is, and if it is very very high it implies it is almost fictional and unrealistic amounts of extremism.\n\n"
            "analyze the following information:\n\n"
            f"current ideology to rate: {best_candidate}\n"
            f"current ideology description: {current_node_description}\n\n"
            "information about its rated related ideologies (parents and children, providing contextual values):\n"
        )
        prompt += "\n".join(related_info_for_llm)
        prompt += (
            "\nbased on the descriptions and extremism ratings of its related ideologies, and considering the "
            "description of the current ideology, determine its 'extremism rating' (0-100). "
            "output only a json object with a single key 'rating' and its numerical value. "
            "example: {\"rating\": 50}"
        )

        print(f"\n--- rating '{best_candidate}' ---")
        new_rating = get_llm_rating(prompt)

        if new_rating is not None:
            tierlist_graph_data[best_candidate]["extremism rating"] = new_rating
            rated_ideologies_count += 1
            progress_bar.update(1)
            print(f"assigned rating {new_rating}% to '{best_candidate}'.")

            # save progress after each successful rating
            try:
                with open(tierlist_graph_file, 'w', encoding='utf-8') as f:
                    json.dump(tierlist_graph_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"error saving progress to '{tierlist_graph_file}': {e}")
        else:
            print(f"failed to get a valid rating for '{best_candidate}'. skipping for this run.")
            time.sleep(2)

        time.sleep(0.5)

    progress_bar.close()
    print(f"\nrating process completed for this run. total rated ideologies: {rated_ideologies_count}")
    print(f"final '{tierlist_graph_file}' saved.")

    # tier classification and saving
    all_rated = all(data.get("extremism rating") is not None for data in tierlist_graph_data.values())
    if all_rated:
        classify_and_save_tiers(tierlist_graph_data, base_graph)
    else:
        print("\nnot all ideologies have been rated yet. tier classification skipped for this run.")

if __name__ == "__main__":
    print("starting program 5: ideology extremism rater")
    run_extremism_rater()
