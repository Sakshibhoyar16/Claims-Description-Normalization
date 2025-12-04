# claims_normalizer.py
"""
Improved Claims Normalizer
- Rule-based extraction (keywords + regex)
- Optional spaCy NER for robust entity extraction (fallback to regex)
- Optional ML classifier (TF-IDF + LogisticRegression) to predict loss_type and severity
- Optional external API-based prediction for loss_type and severity
- Evaluation helpers and batch processing
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import re
import json
import logging
import os

import requests  # for calling external API

# ML libs (optional)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Optional spaCy (if installed)
try:
    import spacy
    _SPACY = spacy.load("en_core_web_sm")
except Exception:
    _SPACY = None

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LossType(Enum):
    WATER_DAMAGE = "Water Damage"
    FIRE_DAMAGE = "Fire Damage"
    THEFT = "Theft"
    VANDALISM = "Vandalism"
    COLLISION = "Collision"
    STORM_DAMAGE = "Storm Damage"
    WIND_DAMAGE = "Wind Damage"
    HAIL_DAMAGE = "Hail Damage"
    LIGHTNING = "Lightning"
    EARTHQUAKE = "Earthquake"
    FLOOD = "Flood"
    SMOKE_DAMAGE = "Smoke Damage"
    ELECTRICAL_DAMAGE = "Electrical Damage"
    FROZEN_PIPE = "Frozen Pipe"
    LIABILITY = "Liability"
    MEDICAL = "Medical"
    OTHER = "Other"


class Severity(Enum):
    MINOR = "Minor"
    MODERATE = "Moderate"
    MAJOR = "Major"
    CATASTROPHIC = "Catastrophic"


@dataclass
class NormalizedClaim:
    """
    Structured representation of a normalized claim.
    """
    loss_types: List[str]
    severity: str
    affected_assets: List[str]
    confidence_score: float
    extracted_entities: Dict[str, List[str]]
    raw_text: str
    predicted_loss: Optional[str] = None
    predicted_severity: Optional[str] = None
    predicted_source: Optional[str] = None  # "api", "ml", "rule"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert dataclass to a serializable dictionary.
        """
        return asdict(self)


class ClaimsNormalizer:
    """
    Claims description normalizer using rule-based extraction with optional
    spaCy NER, optional ML classifiers, and optional external API.
    """

    def __init__(
        self,
        use_spacy: bool = True,
        use_ml: bool = False,
        use_api: bool = False,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the normalizer.

        Args:
            use_spacy: Enable spaCy NER if the model is available.
            use_ml: Enable ML-based prediction once models are trained.
            use_api: Enable external API-based prediction for loss_type and severity.
            api_url: URL of the external API endpoint
                     (or env var CLAIMS_API_URL).
            api_key: API key/token for the external API
                     (or env var CLAIMS_API_KEY).
        """
        self.use_spacy = use_spacy and (_SPACY is not None)
        self.use_ml = use_ml
        self.use_api = use_api

        # external API info
        self.api_url = api_url or os.getenv("CLAIMS_API_URL")
        self.api_key = api_key or os.getenv("CLAIMS_API_KEY")

        if self.use_api and (not self.api_url or not self.api_key):
            logger.warning(
                "use_api=True but api_url or api_key not set. "
                "API predictions will be skipped."
            )

        self._initialize_patterns()
        self._initialize_keywords()

        # ML-related attributes (initialized lazily if train_ml is called)
        self.loss_encoder = LabelEncoder()
        self.sev_encoder = LabelEncoder()
        self.loss_pipeline: Optional[Pipeline] = None
        self.sev_pipeline: Optional[Pipeline] = None

    # ---------- Initialization helpers ----------

    def _initialize_patterns(self) -> None:
        """
        Initialize regex patterns for low-level entity extraction
        (dates, times, money, locations, vehicles).
        """
        self.patterns = {
            "monetary": r"\$[\d,]+(?:\.\d{2})?|\d+\s*(?:dollars?|USD)",
            "date": r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
            "time": r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b",
            "location": r"\b(?:room|kitchen|bedroom|bathroom|garage|basement|attic|living room|dining room|hallway|porch|yard|driveway)\b",
            "vehicle": r"\b(?:\d{4}\s+)?(?:car|truck|van|suv|SUV|sedan|vehicle|auto|automobile|motorcycle|bike)\b",
        }

    def _initialize_keywords(self) -> None:
        """
        Initialize keyword dictionaries for loss type, severity, and assets.
        """
        self.loss_type_keywords: Dict[LossType, List[str]] = {
            LossType.WATER_DAMAGE: [
                "water", "leak", "pipe", "plumbing", "flood", "moisture",
                "wet", "drip", "overflow", "burst pipe", "water damage",
                "submerged",
            ],
            LossType.FIRE_DAMAGE: [
                "fire", "burn", "flame", "smoke", "scorch", "ignite",
                "blaze", "combustion", "burnt",
            ],
            LossType.THEFT: [
                "theft", "stolen", "burglary", "robbery", "break-in",
                "burglar", "took", "missing", "stole",
            ],
            LossType.VANDALISM: [
                "vandalism", "vandalized", "graffiti", "damaged",
                "destruction", "malicious", "defaced",
            ],
            LossType.COLLISION: [
                "collision", "crash", "accident", "hit", "impact",
                "rear-end", "rear ended", "side-swipe", "sideswipe",
                "totaled",
            ],
            LossType.STORM_DAMAGE: [
                "storm", "storm damage", "hurricane", "tornado",
                "severe weather", "tropical storm", "cyclone", "tree fell",
            ],
            LossType.WIND_DAMAGE: [
                "wind", "windstorm", "gust", "blown", "roof damage",
            ],
            LossType.HAIL_DAMAGE: [
                "hail", "hailstorm", "hailstone", "ice pellet",
            ],
            LossType.LIGHTNING: [
                "lightning", "lightning strike", "electrical storm",
                "struck by lightning",
            ],
            LossType.EARTHQUAKE: [
                "earthquake", "seismic", "tremor", "quake",
            ],
            LossType.FLOOD: [
                "flood", "flooding", "flash flood", "inundation",
                "submerged",
            ],
            LossType.SMOKE_DAMAGE: [
                "smoke", "smoke damage", "soot", "smoky",
            ],
            LossType.ELECTRICAL_DAMAGE: [
                "electrical", "electric", "power surge", "short circuit",
                "wiring", "outlet",
            ],
            LossType.FROZEN_PIPE: [
                "frozen", "frozen pipe", "ice", "freeze",
            ],
            LossType.LIABILITY: [
                "liability", "injury", "injured", "slip", "fall",
                "personal injury", "bodily injury",
            ],
            LossType.MEDICAL: [
                "medical", "hospital", "ambulance", "emergency room",
                "doctor", "injury", "pain",
            ],
        }

        self.severity_keywords: Dict[Severity, List[str]] = {
            Severity.MINOR: [
                "minor", "small", "slight", "superficial", "cosmetic",
                "tiny", "little", "scratch", "dent",
            ],
            Severity.MODERATE: [
                "moderate", "medium", "considerable", "significant",
                "noticeable", "damaged", "broken",
            ],
            Severity.MAJOR: [
                "major", "severe", "extensive", "substantial", "serious",
                "large", "heavy", "destroyed", "ruined",
            ],
            Severity.CATASTROPHIC: [
                "catastrophic", "catastrophe", "total loss", "totaled",
                "complete destruction", "devastating", "irreparable",
                "uninhabitable",
            ],
        }

        self.asset_keywords: Dict[str, List[str]] = {
            "Property": [
                "house", "home", "property", "building", "residence",
                "dwelling",
            ],
            "Vehicle": [
                "car", "truck", "vehicle", "auto", "automobile", "van",
                "suv", "SUV", "motorcycle",
            ],
            "Roof": [
                "roof", "shingles", "roofing", "ceiling",
            ],
            "Window": [
                "window", "glass", "windshield",
            ],
            "Door": [
                "door", "entrance", "doorway",
            ],
            "Appliance": [
                "appliance", "refrigerator", "washer", "dryer",
                "dishwasher", "stove", "oven",
            ],
            "Furniture": [
                "furniture", "couch", "sofa", "table", "chair", "bed",
                "desk",
            ],
            "Electronics": [
                "tv", "television", "computer", "laptop", "phone",
                "electronics", "stereo",
            ],
            "Flooring": [
                "floor", "flooring", "carpet", "hardwood", "tile",
                "laminate",
            ],
            "Wall": [
                "wall", "drywall", "paint", "wallpaper",
            ],
            "Personal Property": [
                "jewelry", "clothes", "clothing", "belongings", "items",
            ],
        }

    # ---------- Entity extraction ----------

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities using optional spaCy NER plus regex-based patterns.
        """
        entities: Dict[str, List[str]] = {}

        if self.use_spacy:
            doc = _SPACY(text)
            for ent in doc.ents:
                entities.setdefault(ent.label_, []).append(ent.text)

        for name, pattern in self.patterns.items():
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if matches:
                entities.setdefault(name, []).extend(
                    [m.strip() for m in matches if m]
                )

        for k in list(entities.keys()):
            entities[k] = sorted(list(set(entities[k])))

        return entities

    # ---------- Rule-based loss detection ----------

    def _detect_loss_types_rule(self, text: str) -> List[LossType]:
        """
        Detect candidate loss types using keyword frequency.
        """
        scores: Dict[LossType, int] = {}
        low = text.lower()

        for loss_type, keywords in self.loss_type_keywords.items():
            score = sum(low.count(k) for k in keywords)
            if score > 0:
                scores[loss_type] = score

        if not scores:
            return [LossType.OTHER]

        max_score = max(scores.values())
        picks = [lt for lt, sc in scores.items() if sc >= max_score * 0.7]
        return picks or [LossType.OTHER]

    # ---------- Severity ----------

    def _determine_severity_rule(
        self, text: str, entities: Dict[str, List[str]]
    ) -> Severity:
        """
        Determine severity from text and entities using keyword scores
        and monetary amount thresholds.
        """
        scores = {s: 0 for s in Severity}
        low = text.lower()

        for s, keywords in self.severity_keywords.items():
            for k in keywords:
                if k in low:
                    scores[s] += 2

        if "monetary" in entities:
            for m in entities["monetary"]:
                amt = self._extract_amount(m)
                if amt is None:
                    continue
                if amt < 1000:
                    scores[Severity.MINOR] += 1
                elif amt < 5000:
                    scores[Severity.MODERATE] += 1
                elif amt < 20000:
                    scores[Severity.MAJOR] += 1
                else:
                    scores[Severity.CATASTROPHIC] += 1

        if any(
            w in low
            for w in ["total", "complete", "entire", "whole", "all", "everything", "uninhabitable"]
        ):
            scores[Severity.MAJOR] += 2
            scores[Severity.CATASTROPHIC] += 1

        best = max(scores.items(), key=lambda x: x[1])
        return best[0] if best[1] > 0 else Severity.MODERATE

    # ---------- Assets ----------

    def _extract_assets(self, text: str) -> List[str]:
        """
        Detect which asset categories are mentioned in the text.
        Uses word-boundary matching so 'car' won't match 'cart'.
        """
        found = set()
        tl = text.lower()

        for asset_name, kws in self.asset_keywords.items():
             for k in kws:
                # match 'car' or 'cars', 'truck' or 'trucks', etc.
                pattern = r"\b" + re.escape(k.lower()) + r"s?\b"
                if re.search(pattern, tl):
                    found.add(asset_name)
                    break  # no need to check more keywords for this asset~

        return sorted(list(found))


    # ---------- Utility helpers ----------

    def _extract_amount(self, value_str: str) -> Optional[float]:
        """
        Extract numeric amount from a monetary string (e.g. '$3,500').
        """
        try:
            nums = re.findall(r"[\d,]+(?:\.\d{1,2})?", value_str)
            if not nums:
                return None
            return float(nums[0].replace(",", ""))
        except Exception:
            return None

    def _calculate_confidence(
        self,
        loss_types: List[LossType],
        severity: Severity,
        assets: List[str],
        entities: Dict[str, List[str]],
    ) -> float:
        """
        Heuristic confidence score based on how much signal we extracted.
        """
        conf = 0.0

        if loss_types and loss_types[0] != LossType.OTHER:
            conf += 0.35

        if severity != Severity.MODERATE:
            conf += 0.2

        if assets:
            conf += min(0.3, 0.08 * len(assets))

        if entities:
            conf += min(0.15, 0.03 * len(entities))

        return round(min(conf, 1.0), 2)

    # ---------- External API prediction (optional) ----------

    def _predict_api(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Call Google Gemini API (Google AI Studio) to predict loss_type and severity.

        We prompt the model to respond ONLY with JSON:
        {
          "loss_type": "...",
          "severity": "..."
        }
        """
        if not self.use_api or not self.api_url or not self.api_key:
            return None, None

        try:
            # 1️⃣ Build prompt for Gemini
            prompt = (
                "You are an insurance claims classification model.\n"
                "Given the following insurance claim description, classify:\n"
                "- loss_type: one of "
                "[Water Damage, Fire Damage, Theft, Vandalism, Collision, "
                "Storm Damage, Wind Damage, Hail Damage, Lightning, "
                "Earthquake, Flood, Smoke Damage, Electrical Damage, "
                "Frozen Pipe, Liability, Medical, Other].\n"
                "- severity: one of [Minor, Moderate, Major, Catastrophic].\n\n"
                "Respond ONLY with a JSON object in this exact format:\n"
                '{"loss_type": "...", "severity": "..."}\n\n'
                f"Claim description: {text}"
            )

            # 2️⃣ Gemini REST request body
            body = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ]
            }

            # 3️⃣ Full URL with API key in query params
            url = f"{self.api_url}?key={self.api_key}"

            headers = {
                "Content-Type": "application/json",
            }

            resp = requests.post(url, headers=headers, json=body, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            # 4️⃣ Extract model output
            candidates = data.get("candidates", [])
            if not candidates:
                return None, None

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                return None, None

            model_text = parts[0].get("text", "").strip()
            if not model_text:
                return None, None

            # 5️⃣ Parse JSON from the model output
            try:
                parsed = json.loads(model_text)
            except json.JSONDecodeError:
                logger.error(f"Gemini response not valid JSON: {model_text}")
                return None, None

            loss = parsed.get("loss_type")
            sev = parsed.get("severity")

            if isinstance(loss, str):
                loss = loss.strip()
            if isinstance(sev, str):
                sev = sev.strip()

            return loss, sev

        except Exception as e:
            logger.error(f"API prediction failed: {e}")
            return None, None

    # ---------- ML training (optional) ----------

    def train_ml(
        self,
        texts: List[str],
        loss_labels: List[str],
        severity_labels: List[str],
    ) -> None:
        """
        Train two supervised text classifiers for loss type and severity.
        """
        self.loss_encoder.fit(loss_labels)
        self.sev_encoder.fit(severity_labels)

        self.loss_pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )
        self.loss_pipeline.fit(texts, self.loss_encoder.transform(loss_labels))

        self.sev_pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000)),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )
        self.sev_pipeline.fit(
            texts, self.sev_encoder.transform(severity_labels)
        )

        self.use_ml = True
        logger.info("ML models trained and ready.")

    def predict_ml(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Run ML-based prediction for loss type and severity if models are enabled.
        """
        if not self.use_ml or self.loss_pipeline is None or self.sev_pipeline is None:
            return None, None

        loss_idx = self.loss_pipeline.predict([text])[0]
        sev_idx = self.sev_pipeline.predict([text])[0]

        loss_label = self.loss_encoder.inverse_transform([loss_idx])[0]
        sev_label = self.sev_encoder.inverse_transform([sev_idx])[0]

        return loss_label, sev_label

    # ---------- Main normalize APIs ----------

    def normalize(self, claim_text: str) -> NormalizedClaim:
        """
        Normalize a single claim description into structured fields.
        Priority: API → ML → rule-based.
        """
        claim_raw = claim_text.strip()

        # 1. Entities
        entities = self._extract_entities(claim_raw)

        # 2. Rule-based predictions
        loss_types_rule = self._detect_loss_types_rule(claim_raw)
        severity_rule = self._determine_severity_rule(claim_raw, entities)
        assets = self._extract_assets(claim_raw)

        # 3. Try API, then ML
        pred_loss: Optional[str] = None
        pred_sev: Optional[str] = None
        source = "rule"

        if self.use_api:
            api_loss, api_sev = self._predict_api(claim_raw)
            if api_loss or api_sev:
                pred_loss = api_loss
                pred_sev = api_sev
                source = "api"

        if source == "rule" and self.use_ml:
            ml_loss, ml_sev = self.predict_ml(claim_raw)
            if ml_loss or ml_sev:
                pred_loss = ml_loss
                pred_sev = ml_sev
                source = "ml"

        # 4. Final outputs (start with rule-based)
        final_loss = [lt.value for lt in loss_types_rule]
        final_severity = severity_rule.value

        # override with ML / API if available
        if pred_loss:
            final_loss = [pred_loss]
        if pred_sev:
            final_severity = pred_sev

        # 5. Confidence score
        conf = self._calculate_confidence(
            loss_types_rule, severity_rule, assets, entities
        )

        return NormalizedClaim(
            loss_types=final_loss,
            severity=final_severity,
            affected_assets=assets,
            confidence_score=conf,
            extracted_entities=entities,
            raw_text=claim_raw,
            predicted_loss=pred_loss,
            predicted_severity=pred_sev,
            predicted_source=source,
        )

    def batch_normalize(self, claims: List[str]) -> List[NormalizedClaim]:
        """
        Normalize a batch/list of claim descriptions.
        """
        return [self.normalize(c) for c in claims]

    # ---------- Evaluation helper ----------

    def evaluate(
        self, texts: List[str], true_loss: List[str], true_sev: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate current configuration (rule-based, ML, or API-enhanced).
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        preds_loss: List[str] = []
        preds_sev: List[str] = []

        for t in texts:
            r = self.normalize(t)
            preds_loss.append(
                r.predicted_loss
                if r.predicted_loss
                else (r.loss_types[0] if r.loss_types else "Other")
            )
            preds_sev.append(
                r.predicted_severity if r.predicted_severity else r.severity
            )

        metrics = {
            "loss": {
                "accuracy": float(accuracy_score(true_loss, preds_loss)),
                "f1_macro": float(
                    f1_score(true_loss, preds_loss, average="macro", zero_division=0)
                ),
                "precision_macro": float(
                    precision_score(
                        true_loss,
                        preds_loss,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "recall_macro": float(
                    recall_score(
                        true_loss,
                        preds_loss,
                        average="macro",
                        zero_division=0,
                    )
                ),
            },
            "severity": {
                "accuracy": float(accuracy_score(true_sev, preds_sev)),
                "f1_macro": float(
                    f1_score(true_sev, preds_sev, average="macro", zero_division=0)
                ),
                "precision_macro": float(
                    precision_score(
                        true_sev,
                        preds_sev,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "recall_macro": float(
                    recall_score(
                        true_sev,
                        preds_sev,
                        average="macro",
                        zero_division=0,
                    )
                ),
            },
        }
        return metrics

    # ---------- Serialization ----------

    def to_json(self, normalized: NormalizedClaim) -> str:
        """
        Serialize a NormalizedClaim to a pretty JSON string.
        """
        return json.dumps(normalized.to_dict(), indent=2)


# End of file
