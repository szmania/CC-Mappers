import os
import json
import hashlib
import openai # Assuming OpenAI API is installed and configured

# Define the cache directory relative to this file
CACHE_DIR = os.path.join(os.path.dirname(__file__), '.cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def get_llm_response(prompt):
    """
    Retrieves a response from the LLM, utilizing a caching layer for performance and cost savings.

    Args:
        prompt (str): The prompt string to send to the LLM.

    Returns:
        dict: The parsed JSON response from the LLM, or an error dictionary if an issue occurs.
    """
    cache_key = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    cache_file_path = os.path.join(CACHE_DIR, cache_key + '.json')

    # Cache Hit: If the file exists, read its content and return
    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'r', encoding='utf-8') as f:
            print(f"Cache hit for prompt hash: {cache_key[:8]}...")
            return json.load(f)
    
    # Cache Miss: Proceed to make the API call
    print(f"Cache miss for prompt hash: {cache_key[:8]}... Calling LLM...")
    try:
        # Initialize OpenAI client (ensure OPENAI_API_KEY is set in environment)
        client = openai.OpenAI() 
        response = client.chat.completions.create(
            model="gpt-4o", # Using gpt-4o as a capable model for JSON output
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides JSON output."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"} # Crucial for structured output
        )
        raw_response_content = response.choices[0].message.content
        parsed_response = json.loads(raw_response_content)

        # Save the raw JSON response to the cache file
        with open(cache_file_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_response, f, indent=4)
        
        return parsed_response
    except openai.APIError as e:
        print(f"OpenAI API error: {e}")
        return {"error": f"OpenAI API error: {e}"}
    except json.JSONDecodeError as e:
        print(f"Error decoding LLM response JSON: {e}")
        return {"error": f"LLM response was not valid JSON: {e}"}
    except Exception as e:
        print(f"An unexpected error occurred during LLM call: {e}")
        return {"error": f"Unexpected error: {e}"}

def create_roster_correction_prompt(culture, flagged_units, cultural_unit_pool):
    """
    Generates a prompt string for the LLM to correct core roster units.

    Args:
        culture (str): The faction's culture.
        flagged_units (list): A list of unit keys identified as potentially incorrect.
        cultural_unit_pool (list): A list of valid unit keys for the given culture.

    Returns:
        str: The formatted prompt string.
    """
    prompt = f"""
    You are an expert game data editor. Your task is to correct unit keys in a game roster based on cultural affinity.
    The faction's culture is '{culture}'.
    The following units are currently in the faction's core roster but are flagged as potentially incorrect: {flagged_units}.
    Here is a list of valid unit keys for the '{culture}' culture: {cultural_unit_pool}.

    For each flagged unit, determine the most appropriate replacement from the `cultural_unit_pool`.
    If a flagged unit is already culturally appropriate (i.e., its key is in the cultural_unit_pool or starts with an allowed cultural prefix), keep it as is.
    If a flagged unit is not found in the `cultural_unit_pool` and no obvious replacement exists, suggest a generic equivalent if possible, or indicate 'NO_REPLACEMENT'.

    Provide the output as a JSON object where keys are the original flagged unit keys and values are their corrected unit keys.

    Example:
    Input:
    culture: 'roman'
    flagged_units: ['barbarian_spearmen', 'roman_legionaries']
    cultural_unit_pool: ['roman_legionaries', 'roman_archers', 'roman_cavalry']

    Output:
    {{
        "barbarian_spearmen": "roman_legionaries",
        "roman_legionaries": "roman_legionaries"
    }}
    """
    return prompt

def create_maa_remapping_prompt(culture, maa_roster, cultural_unit_pool, global_unit_pool):
    """
    Generates a prompt string for the LLM to remap the MenAtArm roster.

    Args:
        culture (str): The faction's culture.
        maa_roster (list): The current list of MenAtArm unit entries (dictionaries).
        cultural_unit_pool (list): A list of valid unit keys for the given culture.
        global_unit_pool (list): A comprehensive list of all possible unit keys in the game.

    Returns:
        str: The formatted prompt string.
    """
    prompt = f"""
    You are an expert game data editor. Your task is to remap a faction's 'MenAtArm' roster based on cultural affinity and global availability.
    The faction's culture is '{culture}'.
    The current 'MenAtArm' roster for this faction is: {maa_roster}.
    Here is a list of valid unit keys for the '{culture}' culture: {cultural_unit_pool}.
    Here is a comprehensive list of all possible units in the game (global pool): {global_unit_pool}.

    Your goal is to create a new, culturally appropriate 'MenAtArm' roster.
    The new roster should consist of 5 distinct unit entries.
    For each entry, you need to provide:
    1.  `key`: The unit key (must be from `cultural_unit_pool` if possible, otherwise from `global_unit_pool`).
    2.  `min_rank`: The minimum rank required for this unit to appear (integer, e.g., 0, 1, 2, 3, 4).
    3.  `max_rank`: The maximum rank for this unit (integer, e.g., 0, 1, 2, 3, 4).
    4.  `min_amount`: The minimum amount of this unit (integer, e.g., 1, 2, 3).
    5.  `max_amount`: The maximum amount of this unit (integer, e.g., 1, 2, 3).

    Prioritize units from the `cultural_unit_pool`. If the cultural pool is too small, supplement with appropriate generic units from the `global_unit_pool`.
    Ensure a good progression of units across ranks.
    The `min_rank` and `max_rank` should cover the range 0-4 across the 5 units, with some overlap.
    The `min_amount` and `max_amount` should generally be small (1-3).
    Ensure that `min_rank` <= `max_rank` and `min_amount` <= `max_amount` for all entries.
    Ensure all `key` values are present in the `global_unit_pool`.

    Provide the output as a JSON object containing a single key 'men_at_arms' which is a list of 5 dictionaries, each representing a unit entry with 'key', 'min_rank', 'max_rank', 'min_amount', 'max_amount'.

    Example:
    Input:
    culture: 'roman'
    maa_roster: []
    cultural_unit_pool: ['roman_spearmen', 'roman_archers', 'roman_cavalry']
    global_unit_pool: ['roman_spearmen', 'roman_archers', 'roman_cavalry', 'generic_swordsmen', 'generic_archers', 'roman_legionaries']

    Output:
    {{
        "men_at_arms": [
            {{ "key": "roman_spearmen", "min_rank": 0, "max_rank": 1, "min_amount": 1, "max_amount": 2 }},
            {{ "key": "roman_archers", "min_rank": 0, "max_rank": 2, "min_amount": 1, "max_amount": 2 }},
            {{ "key": "roman_cavalry", "min_rank": 1, "max_rank": 3, "min_amount": 1, "max_amount": 1 }},
            {{ "key": "generic_swordsmen", "min_rank": 2, "max_rank": 4, "min_amount": 1, "max_amount": 2 }},
            {{ "key": "roman_legionaries", "min_rank": 3, "max_rank": 4, "min_amount": 1, "max_amount": 1 }}
        ]
    }}
    """
    return prompt
