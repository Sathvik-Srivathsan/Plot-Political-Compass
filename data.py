import requests
from bs4 import BeautifulSoup
from bs4.element import NavigableString
import json
import time
from tqdm import tqdm
import re
import os

# configuration
base_url = "https://polcompball.wikitide.org"
list_of_ideologies_url = f"{base_url}/wiki/List_of_Ideologies"
output_jsonl_file = "political_ideologies_structured_data.jsonl"

default_headers = {
    'user-agent': 'mozilla/5.0 (windows nt 10.0; win64; x64) applewebkit/537.36 (khtml, like gecko) chrome/109.0.0.0 safari/537.36'
}

excluded_article_section_ids = {
    'how_to_draw', 'gallery', 'further_information', 'see_also',
    'references', 'external_links', 'notes', 'bibliography',
    'navigation_templates', 'citations', 'navbox', 'references_and_notes',
    'see_also', 'external_links', 'videos',
    'user', 'talk', 'draft', 'sandbox', 'test', 'archive', 'log', 'redirect',
    'discussions', 'polls', 'surveys', 'quizzes', 'templates', 'categories',
    'files', 'media', 'special', 'help', 'project', 'portal', 'forum', 'blog',
    'news', 'announcements', 'updates', 'community', 'guidelines', 'policies',
    'rules', 'about', 'contact', 'privacy', 'terms', 'disclaimer', 'copyright',
    'donations', 'merchandise', 'press', 'partnerships', 'jobs', 'developers',
    'api', 'irc', 'discord', 'subpage',
}

# helper functions
def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\[.*?\]', '', text)
    return text

def get_ideology_titles_from_list_page(list_page_url):
    # fetch ideology titles
    s = requests.Session()
    s.headers.update(default_headers)

    titles = []
    print(f"fetching ideology titles from: {list_page_url}...")

    try:
        response = s.get(list_page_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        content_div = soup.find('div', class_='mw-parser-output')

        if content_div:
            for link in content_div.find_all('a', href=True):
                href = link.get('href')
                title_text = link.get('title')

                if href and title_text:
                    if href.startswith('/wiki/file:'):
                        continue

                    if '/' in title_text:
                        continue

                    if title_text in ["list of ideologies", "community:how to edit on the wiki", "list of ideology icons"]:
                        continue

                    clean_title = title_text.replace('_', ' ').strip()

                    if clean_title not in titles:
                        titles.append(clean_title)
        else:
            print("could not find 'mw-parser-output' div on the list page.")

    except requests.exceptions.requestexception as e:
        print(f"http error fetching ideology list from {list_page_url}: {e}. exiting.")
        exit()
    except Exception as e:
        print(f"unexpected error parsing ideology list from {list_page_url}: {e}. exiting.")
        exit()

    print(f"found {len(titles)} ideology titles from list page.")
    return titles

def scrape_and_parse_ideology(title):
    # fetch and parse ideology page
    page_url = f"{base_url}/wiki/{title.replace(' ', '_')}"

    try:
        response = requests.get(page_url, headers=default_headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # article body section
        article_body_text = ""
        content_div = soup.find('div', class_='mw-parser-output')

        if content_div:
            current_section_is_excluded = False
            for element in content_div.children:
                if element.name == 'h2':
                    section_id_span = element.find('span', class_='mw-headline')
                    if section_id_span and section_id_span.get('id') in excluded_article_section_ids:
                        current_section_is_excluded = True
                        continue
                    else:
                        current_section_is_excluded = False

                if not current_section_is_excluded:
                    if element.name in ['p', 'ul', 'ol', 'li', 'h3']:
                        article_body_text += element.get_text(separator=" ", strip=True) + "\n"

            article_body_text = clean_text(article_body_text)

        # infobox details section
        infobox_data = {}
        infobox = soup.find('aside', class_='portable-infobox')

        if infobox:
            for item in infobox.find_all('div', class_='pi-item'):
                label_tag = item.find('h3', class_='pi-data-label')
                value_tag = item.find('div', class_='pi-data-value')

                if label_tag and value_tag:
                    label_text_for_comparison = clean_text(label_tag.get_text()).lower().replace(':', '').strip()

                    if label_text_for_comparison in ["influenced by", "influenced"]:
                        items = []
                        for a_tag in value_tag.find_all('a', href=True):
                            if a_tag['href'].startswith('/wiki/file:'):
                                continue

                            item_text = clean_text(a_tag.get_text())

                            if item_text:
                                items.append(item_text)

                        if label_text_for_comparison == "influenced by":
                            infobox_data["influenced_by"] = items
                        else:
                            infobox_data["influenced"] = items
                    else:
                        value_parts = []
                        for content in value_tag.contents:
                            if isinstance(content, NavigableString):
                                stripped_text = str(content).strip()
                                if stripped_text:
                                    value_parts.append(stripped_text)
                            elif content.name == 'a':
                                value_parts.append(content.get_text(strip=True))
                        value = clean_text(" ".join(value_parts))

                        if label_text_for_comparison == "aliases":
                            infobox_data["aliases"] = value
                        elif label_text_for_comparison == "alignment(s)":
                            infobox_data["alignment"] = value
                        elif label_text_for_comparison == "likes":
                            infobox_data["likes"] = value
                        elif label_text_for_comparison == "dislikes":
                            infobox_data["dislikes"] = value

        return {
            "ideology_name": title,
            "article_body": article_body_text,
            "infobox_details": infobox_data
        }

    except requests.exceptions.requestexception as e:
        print(f"http error scraping '{title}' ({page_url}): {e}")
        return None
    except Exception as e:
        print(f"unexpected error parsing '{title}' ({page_url}): {e}")
        return None

# main scraping logic
def run_scraper(output_file=output_jsonl_file):
    all_titles = get_ideology_titles_from_list_page(list_of_ideologies_url)

    # remove existing file
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"removed existing file: {output_file}")

    with open(output_file, 'a', encoding='utf-8') as f:
        print(f"\nstarting to scrape and save data to '{output_file}'...")
        for title in tqdm(all_titles):
            ideology_data = scrape_and_parse_ideology(title)

            if ideology_data:
                f.write(json.dumps(ideology_data, ensure_ascii=False) + '\n')

            time.sleep(0.05)

    print(f"\nscraping complete! data saved to '{output_file}'")

if __name__ == "__main__":
    print("starting program 1: structured ideology data scraper")
    run_scraper()

    # print sample data
    print("\n--- sample of saved data (first 3 entries) ---")
    try:
        with open(output_jsonl_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                print(json.dumps(json.loads(line), indent=2, ensure_ascii=False))
                print("-" * 30)
    except filenotfounderror:
        print(f"error: output file '{output_jsonl_file}' not found.")
    except Exception as e:
        print(f"error reading sample from file: {e}")
