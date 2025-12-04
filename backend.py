# backend.py

from claims_normalizer import ClaimsNormalizer

# TODO: replace these with your real values
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"  # <- your model API URL
API_KEY = "AIzaSyD4a4I1Hh35JCqv4Vd5lbj4zh-eFc0eetU"                  # <- your API key

# Create the normalizer, enabling the external API
normalizer = ClaimsNormalizer(
    use_api=True,
    api_url=API_URL,
    api_key=API_KEY,
    use_spacy=True,   # optional
    use_ml=False      # set True if you later train ML
)

# Example claim text
claim_text = "Water leak in kitchen damaged the floor and cabinets. Estimate $3,500 in repairs needed."

result = normalizer.normalize(claim_text)

print("Predicted source:", result.predicted_source)
print("Result dict:\n", result.to_dict())
