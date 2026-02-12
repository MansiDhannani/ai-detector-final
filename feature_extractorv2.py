import re
import math
from collections import Counter
import tokenize
from io import BytesIO
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound


class FeatureExtractorV2:
    """
    High-signal feature extractor for AI vs Human code detection
    """
    def __init__(self):
        pass

    def detect_language(self, code: str) -> str:
        """Detect programming language using pygments library"""
        try:
            lexer = guess_lexer(code)
            return lexer.name.lower()  # e.g., "python", "java"
        except ClassNotFound:
            return "unknown"

    def extract(self, code: str) -> dict:
        features = {}

        lines = code.splitlines()
        total_lines = len(lines) if lines else 1

        # LANGUAGE DETECTION
        lang = self.detect_language(code)
        lang_mapping = {"python": 0, "java": 1, "cpp": 2, "unknown": 3}
        features["language_id"] = lang_mapping.get(lang, 3)

        # --- BASIC LINE TYPES ---
        comment_lines = [l for l in lines if l.strip().startswith("#")]
        blank_lines = [l for l in lines if not l.strip()]
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]

        # 1️⃣ Comment ratio
        features["comment_ratio"] = len(comment_lines) / total_lines

        # 2️⃣ Structured docstring flag
        features["has_structured_docstring"] = int(
            bool(re.search(r"Args:|Arguments:|Parameters:|Returns:", code))
        )

        # 3️⃣ Line length standard deviation
        line_lengths = [len(l) for l in code_lines if l.strip()]
        features["std_line_length"] = (
            self._std(line_lengths) if len(line_lengths) > 1 else 0.0
        )

        # 4️⃣ Indentation variance
        indent_levels = [
            len(l) - len(l.lstrip(" "))
            for l in code_lines
            if l.startswith(" ")
        ]
        features["indentation_variance"] = (
            self._std(indent_levels) if len(indent_levels) > 1 else 0.0
        )

        # 5️⃣ Over-explanation index
        features["over_explanation_count"] = self._over_explanation_score(comment_lines)

        # 6️⃣ Naming consistency score
        features["naming_consistency"] = self._naming_consistency(code)

        # 7️⃣ Character entropy
        features["char_entropy"] = self._char_entropy(code)

        # 8️⃣ Token repetition ratio
        features["token_repetition_ratio"] = self._token_repetition(code)

        return features

    # ---------------- HELPERS ---------------- #

    def _std(self, values):
        mean = sum(values) / len(values)
        return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))

    def _over_explanation_score(self, comments):
        """
        Counts comments explaining obvious actions
        """
        patterns = [
            r"initialize",
            r"loop through",
            r"iterate",
            r"check if",
            r"base case",
            r"return result",
        ]
        score = 0
        for c in comments:
            for p in patterns:
                if re.search(p, c.lower()):
                    score += 1
                    break
        return score

    def _naming_consistency(self, code):
        """
        Measures how consistent variable naming is
        AI → very consistent
        Human → mixed styles
        """
        names = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", code)
        if not names:
            return 0.0

        snake = sum(1 for n in names if "_" in n and n.islower())
        camel = sum(1 for n in names if re.match(r"[a-z]+[A-Z]", n))

        total = len(names)
        dominant = max(snake, camel)

        return dominant / total

    def _char_entropy(self, text):
        if not text:
            return 0.0

        counts = Counter(text)
        total = len(text)
        entropy = 0.0

        for c in counts.values():
            p = c / total
            entropy -= p * math.log2(p)

        return entropy

    def _token_repetition(self, code):
        try:
            tokens = []
            g = tokenize.tokenize(BytesIO(code.encode("utf-8")).readline)
            for tok in g:
                if tok.type == tokenize.NAME:
                    tokens.append(tok.string)

            if not tokens:
                return 0.0

            unique = len(set(tokens))
            return 1 - (unique / len(tokens))
        except Exception:
            return 0.0
