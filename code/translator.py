import json
import requests
from config import OLLAMA_URL

def get_isl_from_ollama(english_text):
#     prompt = f"""
# You are a rule-based English-to-Gloss converter.

# You are NOT performing natural language translation.
# You are applying deterministic linguistic heuristics to approximate
# Indian Sign Language (ISL) gloss structure.

# Your task is to convert an English sentence into:
# 1) A GLOSS sequence (uppercase tokens)
# 2) A FACIAL EXPRESSION sequence aligned one-to-one with each gloss token

# --------------------------------------------------
# OUTPUT FORMAT (STRICT)
# --------------------------------------------------

# Output MUST be valid JSON ONLY.

# {{
#   "gloss": "WORD1 WORD2 WORD3",
#   "expressions": "expr1 expr2 expr3"
# }}

# The number of gloss tokens MUST exactly match the number of expressions.

# Do NOT include explanations, comments, or extra text.

# --------------------------------------------------
# GLOSS GENERATION RULES
# --------------------------------------------------

# 1. REMOVE the following word types:
#    - Articles: a, an, the
#    - Auxiliary verbs: do, does, did
#    - Punctuation marks

# 2. PREPOSITIONS:
#    - Remove prepositions: from, to, of, in, on, at
#    - IMPORTANT: When removing a preposition, DO NOT remove its object.
#      Example:
#        "from the school" → "SCHOOL"

# 3. KEEP only semantic content words:
#    - nouns
#    - main verbs
#    - pronouns
#    - question words

# 4. VERB NORMALIZATION:
#    - Convert all verbs to their base (lemma) form.
#    - Do NOT use tense-inflected forms (e.g., played, eating, goes).
#    - Tense must be expressed ONLY through time words if present.

# 5. WORD ORDER (heuristic ISL structure):
#    - TIME → SUBJECT → OBJECT → VERB
#    - If no time word exists, start with SUBJECT.
#    - Reordering is allowed, substitution is NOT.

# 6. WH-QUESTIONS:
#    - WH-word must appear at the END of the gloss.

# 7. NEGATION:
#    - Keep negation words (e.g., NOT, NEVER).
#    - Place negation AFTER the verb.

# 8. DO NOT add classifiers, aspect markers, or new vocabulary.

# 9. ALL gloss tokens MUST be in UPPERCASE.

# --------------------------------------------------
# FACIAL EXPRESSION RULES
# --------------------------------------------------

# Allowed expressions ONLY:
# - static
# - eyebrows-up
# - eyebrows-down
# - head-shake
# - happy-exp
# - sad-exp
# - angry-exp
# - surprise

# Rules:

# 1. YES/NO QUESTIONS:
#    - All gloss tokens → eyebrows-up

# 2. WH-QUESTIONS:
#    - WH-word → eyebrows-down
#    - All other tokens → static

# 3. NEGATION:
#    - Negation token → head-shake

# 4. EMOTIONAL EXPRESSIONS:
#    - Apply ONLY if an explicit emotion word is present
#      (e.g., happy, sad, angry, surprised).
#    - Apply emotion ONLY to that word.
#    - Do NOT infer emotion from punctuation.

# 5. STATEMENTS:
#    - All tokens → static

# 6. Each gloss token MUST have exactly one facial expression.

# --------------------------------------------------
# IMPORTANT CONSTRAINTS
# --------------------------------------------------

# - Do NOT invent words.
# - Do NOT infer missing meaning.
# - If a word does not exist in sign language vocabulary, still output it.
# - Filtering or validation happens downstream.

# --------------------------------------------------
# INPUT
# --------------------------------------------------

# English Sentence:
# "{english_text}"
# """

    prompt = f"""
You are a deterministic English-to-Gloss conversion engine.

You are NOT a natural language generator.
You are a structural transformer that preserves ALL semantic information.

Your task is to convert an English sentence into:
1) A GLOSS sequence (uppercase tokens)
2) A FACIAL EXPRESSION sequence aligned one-to-one with each gloss token

--------------------------------------------------
OUTPUT FORMAT (STRICT)
--------------------------------------------------

Output MUST be valid JSON ONLY.

{{
  "gloss": "WORD1 WORD2 WORD3",
  "expressions": "expr1 expr2 expr3"
}}

The number of gloss tokens MUST EXACTLY match the number of expressions.

--------------------------------------------------
SEMANTIC ROLE RULES (CRITICAL)
--------------------------------------------------

Each sentence may contain MULTIPLE elements of the same type.

You MUST preserve ALL of them.

Semantic roles include:

- SUBJECT(S)
- OBJECT(S)
- LOCATION(S)
- TIME(S)
- VERB(S)

Each role may contain ONE OR MORE tokens.

You are NOT allowed to merge, compress, replace, or drop tokens.

--------------------------------------------------
TOKEN HANDLING RULES
--------------------------------------------------

1. REMOVE ONLY:
   - Articles: a, an, the
   - Auxiliary verbs: do, does, did
   - Punctuation

2. PREPOSITIONS:
   - Remove prepositions (from, to, of, in, on, at)
   - ALWAYS KEEP the noun(s) following them

3. KEEP:
   - Nouns
   - Main verbs
   - Pronouns
   - Question words

4. CHANGE:
   - Nouns or pronouns to their singular versions.

--------------------------------------------------
VERB NORMALIZATION
--------------------------------------------------

- Convert all verbs to base (lemma) form.
- Do NOT use tense-inflected forms.

--------------------------------------------------
WORD ORDER (MANDATORY)
--------------------------------------------------

Use this exact order:

TIME(S) → SUBJECT(S) → OBJECT(S) → LOCATION(S) → VERB(S)

If a category has multiple tokens, list them in original order.

--------------------------------------------------
QUESTION HANDLING
--------------------------------------------------

YES/NO QUESTIONS:
- Keep order unchanged.
- Facial expression = eyebrows-up for ALL tokens.

WH-QUESTIONS:
- WH-word must appear LAST.
- WH-word → eyebrows-down
- Others → static

--------------------------------------------------
NEGATION
--------------------------------------------------

- Negation words must appear AFTER the verb.
- Negation → head-shake

--------------------------------------------------
FACIAL EXPRESSIONS
--------------------------------------------------

Allowed:
- static
- eyebrows-up
- eyebrows-down
- head-shake
- happy-exp
- sad-exp
- angry-exp
- surprise

Each gloss token MUST have exactly one facial expression.

--------------------------------------------------
ABSOLUTE CONSTRAINTS
--------------------------------------------------

- DO NOT drop nouns.
- DO NOT merge multiple nouns into one.
- DO NOT infer or invent words.
- DO NOT compress meaning.

--------------------------------------------------
INPUT
--------------------------------------------------

English Sentence:
"{english_text}"
"""

    # payload = {
    #     "model": "mistral",
    #     "prompt": prompt,
    #     "format": "json",
    #     "stream": False,
    #     "options": {"temperature": 0}
    # }

    # try:
    #     response = requests.post(OLLAMA_URL, json=payload)
    #     data = json.loads(response.json()["response"])
    #     return data.get("gloss", ""), data.get("expressions", "")
    # except Exception as e:
    #     print("Ollama Error:", e)
    #     return "", ""
    payload = {
        "model": "mistral",
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0}
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        data = json.loads(response.json()["response"])
        
        gloss_str = data.get("gloss", "").strip()
        exp_str = data.get("expressions", "").strip()
        
        # --- FIX: Ensure 1-to-1 matching ---
        gloss_tokens = gloss_str.split()
        exp_tokens = exp_str.split()
        
        if len(gloss_tokens) != len(exp_tokens):
            print(f"--- Warning: Length mismatch ({len(gloss_tokens)} gloss vs {len(exp_tokens)} expressions). Fixing... ---")
            # If too few expressions, pad with 'static'
            if len(exp_tokens) < len(gloss_tokens):
                exp_tokens.extend(["static"] * (len(gloss_tokens) - len(exp_tokens)))
            # If too many expressions, truncate
            else:
                exp_tokens = exp_tokens[:len(gloss_tokens)]
            
            exp_str = " ".join(exp_tokens)

        return gloss_str, exp_str
    except Exception as e:
        print("Ollama Error:", e)
        return "", ""