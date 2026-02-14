#!/usr/bin/env python3
"""
HYBRID AI CODE DETECTOR - Phase 3
Combines feature-based ML, CodeBERT, and metadata analysis
Target Accuracy: 91-94%
"""

import torch
import torch.nn as nn
from xgboost import XGBClassifier
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import ast
import os
import joblib
import re
import gc
from pathlib import Path
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# COMPONENT 1: FEATURE-BASED DETECTOR
# ============================================================================

class FeatureExtractor:
    """Extracts 20+ features from code"""
    
    def __init__(self):
        pass

    def detect_language(self, code: str) -> str:
        """Detect programming language using pygments library"""
        try:
            lexer = guess_lexer(code)
            return lexer.name.lower()  # e.g., "python", "java"
        except ClassNotFound:
            return "unknown"

    def extract_all_features(self, code: str) -> Dict:
        """Extract comprehensive feature set"""
        features = {}
        
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]

        # LANGUAGE DETECTION
        lang = self.detect_language(code)
        lang_mapping = {"python": 0, "java": 1, "cpp": 2, "unknown": 3}
        features['language_id'] = lang_mapping.get(lang, 3)
        
        # STRUCTURAL FEATURES (5)
        features['total_lines'] = len(lines)
        features['non_empty_lines'] = len(non_empty_lines)
        features['avg_line_length'] = np.mean([len(l) for l in lines]) if lines else 0
        features['max_line_length'] = max([len(l) for l in lines]) if lines else 0
        features['empty_line_ratio'] = (len(lines) - len(non_empty_lines)) / len(lines) if lines else 0
        
        # LANGUAGE SPECIFIC FEATURES
        features['semicolon_count'] = code.count(';') if lang in ['java', 'cpp'] else 0
        features['indent_line_count'] = sum(1 for l in lines if l.startswith('    ') or l.startswith('\t'))
        
        # COMMENT ANALYSIS (5)
        comment_lines = [l for l in lines if l.strip().startswith('#')]
        features['comment_count'] = len(comment_lines)
        features['comment_density'] = len(comment_lines) / len(lines) if lines else 0
        features['avg_comment_length'] = np.mean([len(c) for c in comment_lines]) if comment_lines else 0
        features['inline_comments'] = sum(1 for l in lines if '#' in l and not l.strip().startswith('#'))
        features['docstring_count'] = code.count('"""') + code.count("'''")
        
        # AI-SPECIFIC PATTERNS (6)
        ai_phrases = {
            'this_function': code.lower().count('this function'),
            'helper_function': code.lower().count('helper'),
            'utility': code.lower().count('utility'),
            'initialize': code.lower().count('initialize'),
            'implement': code.lower().count('implement'),
            'calculate': code.lower().count('calculate'),
        }
        features['ai_phrase_total'] = sum(ai_phrases.values())
        for phrase, count in ai_phrases.items():
            features[f'phrase_{phrase}'] = count
        
        # VARIABLE NAMING (4)
        generic_vars = ['data', 'result', 'temp', 'value', 'item', 'obj', 'arr', 'lst', 'dict']
        features['generic_var_count'] = sum(code.count(f' {var}') for var in generic_vars)
        features['single_char_vars'] = sum(code.count(f' {c} ') for c in 'ijkxyz')
        features['long_var_names'] = len(re.findall(r'\b\w{15,}\b', code))
        features['camel_case_count'] = len(re.findall(r'\b[a-z]+[A-Z][a-zA-Z]*\b', code))
        
        # COMPLEXITY METRICS (5)
        features['if_statements'] = code.count('if ')
        features['for_loops'] = code.count('for ')
        features['while_loops'] = code.count('while ')
        features['try_blocks'] = code.count('try:')
        features['function_defs'] = code.count('def ')
        
        # ERROR HANDLING (3)
        features['except_blocks'] = code.count('except')
        features['exception_types'] = len(re.findall(r'except\s+\w+', code))
        features['error_handling_ratio'] = features['try_blocks'] / (features['total_lines'] / 20) if features['total_lines'] > 0 else 0
        
        # DOCUMENTATION QUALITY (3)
        features['has_module_docstring'] = 1 if code.strip().startswith('"""') or code.strip().startswith("'''") else 0
        features['docstring_density'] = features['docstring_count'] / (features['function_defs'] + 1)
        features['args_documented'] = code.lower().count('args:')
        
        # STYLE PATTERNS (4)
        features['import_statements'] = code.count('import ')
        features['type_hints'] = code.count('->') + code.count(': int') + code.count(': str')
        features['f_strings'] = code.count('f"') + code.count("f'")
        features['list_comprehensions'] = len(re.findall(r'\[.+for .+ in .+\]', code))
        
        return features
    
    def extract_features_batch(self, codes: List[str]) -> pd.DataFrame:
        """Extract features for multiple code samples"""
        features_list = [self.extract_all_features(code) for code in codes]
        return pd.DataFrame(features_list)

class FeatureBasedDetector:
    """Random Forest classifier on extracted features"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.calibrated_model = None
        self.feature_names: List[str] = []
        self.X_train = None
        self.y_train = None
        self.is_trained = False
    
    def train(self, human_codes: List[str], ai_codes: List[str]):
        """Train the feature-based model"""
        print("🔧 Training feature-based detector...")
        
        # Extract features
        X_human = self.feature_extractor.extract_features_batch(human_codes)
        X_ai = self.feature_extractor.extract_features_batch(ai_codes)
        
        X = pd.concat([X_human, X_ai], ignore_index=True)
        y = np.array([0] * len(human_codes) + [1] * len(ai_codes))
        self.X_train = X
        self.y_train = y
        
        self.feature_names = X.columns.tolist()
        
        # 1. Hyperparameter Tuning with GridSearchCV
        print("🔎 Tuning XGBoost parameters...")
        xgb_base = XGBClassifier(eval_metric='logloss', random_state=42)
        param_grid = {
            "n_estimators": [200, 400],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8],
            "colsample_bytree": [0.8]
        }
        
        grid_search = GridSearchCV(
            estimator=xgb_base,
            param_grid=param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=-1
        )
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        print(f"✅ Best Params: {grid_search.best_params_}")
        
        # RF Backup for comparison
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf_backup = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_backup.fit(X_train, y_train)
        rf_acc = accuracy_score(y_test, rf_backup.predict(X_test))
        print(f"📊 RF Backup Accuracy: {rf_acc:.4f}")

        # 2. Calibrate XGBoost probabilities
        print("⚖️ Calibrating XGBoost probabilities...")
        self.calibrated_model = CalibratedClassifierCV(self.model, method='isotonic', cv=5)
        self.calibrated_model.fit(X, y)

        # Get feature importance
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"✅ Trained on {len(X)} samples")
        print(f"\nTop 5 features:")
        print(importance.head())
        self.is_trained = True
        
        return self
    
    def predict(self, code: str) -> Dict:
        """Predict single code sample"""
        if not self.is_trained:
            raise RuntimeError(
                "FeatureBasedDetector used before training."
            )

        features = self.feature_extractor.extract_all_features(code)
        features_df = pd.DataFrame([features])
        
        # Ensure same features as training
        for col in self.feature_names:
            if col not in features_df.columns:
                features_df[col] = 0
        features_df = features_df[self.feature_names]
        
        proba = self.calibrated_model.predict_proba(features_df)[0]
        
        return {
            'human_prob': float(proba[0]),
            'ai_prob': float(proba[1]),
            'features': features
        }

# ============================================================================
# COMPONENT 2: CODEBERT DETECTOR
# ============================================================================

class CodeBERTWrapper:
    """Wrapper for CodeBERT feature extraction and lightweight classification"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        print("📥 Loading CodeBERT for feature extraction...")
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        self.model = AutoModel.from_pretrained('microsoft/codebert-base').to(self.device)
        self.clf = LogisticRegression(max_iter=1000)
        self.train_embeddings = None
        self.y_train = None
        self.is_trained = False
        print("✅ CodeBERT loaded")
        
    def get_embeddings(self, codes: List[str]) -> torch.Tensor:
        """Extract [CLS] token embeddings"""
        self.model.to(self.device)
        self.model.eval()
        all_embeddings = []
        batch_size = 8
        with torch.inference_mode():
            for i in range(0, len(codes), batch_size):
                batch = codes[i:i+batch_size]
                encoded = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=512, return_tensors='pt'
                ).to(self.device)
                outputs = self.model(**encoded)
                embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)

    def train(self, human_codes: List[str], ai_codes: List[str]):
        """Train the lightweight classifier on CodeBERT embeddings"""
        print("🧠 Extracting CodeBERT embeddings for training...")
        codes = human_codes + ai_codes
        y = np.array([0] * len(human_codes) + [1] * len(ai_codes))
        self.train_embeddings = self.get_embeddings(codes)
        self.y_train = y
        print("📉 Training LogisticRegression on embeddings...")
        self.clf.fit(self.train_embeddings.numpy(), y)
        self.is_trained = True
        print("✅ CodeBERT classifier training complete")
        return self
    
    def predict(self, code: str) -> Dict:
        """Predict single code sample"""
        if not self.is_trained:
            return {'human_prob': 0.5, 'ai_prob': 0.5}
        embedding = self.get_embeddings([code])
        probs = self.clf.predict_proba(embedding.numpy())[0]
        return {
            'human_prob': float(probs[0]),
            'ai_prob': float(probs[1])
        }

# ============================================================================
# COMPONENT 3: METADATA ANALYZER
# ============================================================================

class MetadataAnalyzer:
    """Analyzes git history and file metadata"""
    
    def analyze(self, metadata: Dict) -> Dict:
        """Analyze metadata signals"""
        score = 0.0
        flags = []
        
        # Check commit patterns
        if metadata.get('total_commits', 0) <= 2:
            score += 0.25
            flags.append("Very few commits (suspicious)")
        
        if metadata.get('large_initial_commit', False):
            score += 0.20
            flags.append("Large initial commit")
        
        # Check development speed
        completion_speed = metadata.get('completion_speed', 0.5)
        if completion_speed > 0.8:  # Too fast
            score += 0.20
            flags.append("Completed too quickly")
        
        # Check file organization
        if metadata.get('perfect_structure', False):
            score += 0.15
            flags.append("Suspiciously perfect organization")
        
        # Check documentation
        if metadata.get('full_documentation', False):
            score += 0.10
            flags.append("100% documentation (unusual)")
        
        # Check test coverage
        test_coverage = metadata.get('test_coverage', 0)
        if test_coverage > 0.95:
            score += 0.10
            flags.append("Unusually high test coverage")
        
        return {
            'score': min(score, 1.0),
            'flags': flags
        }

# ============================================================================
# COMPONENT 4: PATTERN MATCHER
# ============================================================================

class PatternMatcher:
    """Matches against known AI code patterns"""
    
    def __init__(self):
        self.patterns = {
            'chatgpt': [
                r'#\s*This function (?:does|performs|implements)',
                r'"""[\s\S]*?Args:[\s\S]*?Returns:[\s\S]*?"""',
                r'#\s*Helper function to',
                r'#\s*Initialize ',
                r'#\s*Note:',
            ],
            'copilot': [
                r'//\s*TODO:',
                r'//\s*Implementation',
                r'#\s*Type:',
            ],
            'claude': [
                r'"""[\s\S]*?Note:[\s\S]*?"""',
                r'#\s*Implementation note:',
            ]
        }
    
    def check(self, code: str) -> Dict:
        """Check code against known patterns"""
        matches = {'chatgpt': 0, 'copilot': 0, 'claude': 0}
        total_patterns = sum(len(p) for p in self.patterns.values())
        
        for tool, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    matches[tool] += 1
        
        total_matches = sum(matches.values())
        overall_score = total_matches / total_patterns
        
        # Determine most likely source
        likely_source = max(matches, key=matches.get) if total_matches > 0 else None
        
        return {
            'score': overall_score,
            'matches': matches,
            'likely_source': likely_source,
            'total_matches': total_matches
        }

# ============================================================================
# HYBRID ENSEMBLE SYSTEM
# ============================================================================

class HybridAIDetector:
    """Main hybrid detector combining all components"""
    
    def __init__(self, use_gpu=False):
        print("🚀 Initializing Hybrid AI Detector...")
        
        self.feature_detector = FeatureBasedDetector()
        
        device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f"   Using device: {device}")
        self.codebert_detector = CodeBERTWrapper(device=device)
        
        self.metadata_analyzer = MetadataAnalyzer()
        self.pattern_matcher = PatternMatcher()
        
        # Ensemble weights (can be learned)
        self.weights = {
            'features': 0.80,
            'codebert': 0.20,
            'patterns': 0.00  # Patterns now handled as heuristic boost if needed
        }
        
        print("✅ Hybrid detector initialized\n")
    
    def train(self, human_codes: List[str], ai_codes: List[str], epochs=3):
        """Train all components"""
        print("=" * 70)
        print("TRAINING HYBRID AI DETECTOR")
        print("=" * 70)
        print(f"Human samples: {len(human_codes)}")
        print(f"AI samples: {len(ai_codes)}")
        print()
        
        # Train feature-based
        self.feature_detector.train(human_codes, ai_codes)
        print()
        
        # Train CodeBERT
        self.codebert_detector.train(human_codes, ai_codes)
        print()
        
        print("=" * 70)
        print("✅ TRAINING COMPLETE")
        print("=" * 70)
        
        return self
    
    def load_pretrained(self, path="saved_models"):
        """Load saved model components"""
        model_file = f"{path}/hybrid_ai_detector_ensemble.pkl"
        feature_file = f"{path}/feature_names_v2.pkl"
        training_file = f"{path}/training_data.pkl"
        
        # Fallback for the other filename you mentioned
        if not os.path.exists(model_file):
            if os.path.exists(f"{path}/hybrid_ai_detector.pkl"):
                model_file = f"{path}/hybrid_ai_detector.pkl"
        
        # Check for Git LFS pointer files (common cause of KeyError: 118)
        # We only strictly require the model and feature names for detection
        required_files = [model_file, feature_file]
        optional_files = [training_file]
        
        for f_path in required_files + [f for f in optional_files if os.path.exists(f)]:
            if os.path.exists(f_path):
                file_size = os.path.getsize(f_path)
                with open(f_path, 'rb') as f:
                    header = f.read(100)
                    if b"version https://git-lfs" in header:
                        raise RuntimeError(
                            f"File {f_path} is a Git LFS pointer (Size: {file_size} bytes), not the actual model data. "
                            "The binary weights are missing from the deployment. Run 'git lfs push origin main --all' locally."
                        )

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found at {model_file}. Ensure models are pushed to the repository.")
        
        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"Critical Error: Feature names file not found at {feature_file}.")

        ensemble_data = joblib.load(model_file)
        
        self.feature_detector.model = ensemble_data.get("xgb_base_model")
        self.feature_detector.calibrated_model = ensemble_data["xgb_model"]
        self.feature_detector.X_train = ensemble_data.get("X_train")
        self.feature_detector.y_train = ensemble_data.get("y_train")

        # Load training data from separate file if not found in ensemble
        if (self.feature_detector.X_train is None or self.feature_detector.y_train is None) and os.path.exists(training_file):
            train_data = joblib.load(training_file)
            self.feature_detector.X_train = train_data.get("X_train")
            self.feature_detector.y_train = train_data.get("y_train")
            print(f"📈 Loaded training data from {training_file}")

        self.codebert_detector.clf = ensemble_data.get("codebert_clf")
        self.codebert_detector.train_embeddings = ensemble_data.get("codebert_embeddings")
        self.codebert_detector.y_train = ensemble_data.get("codebert_y_train")
        self.codebert_detector.is_trained = self.codebert_detector.clf is not None
        
        self.weights['features'] = ensemble_data["weight_xgb"]
        self.weights['codebert'] = ensemble_data["weight_codebert"]
        
        self.feature_detector.feature_names = joblib.load(feature_file)
        self.feature_detector.is_trained = True
        print(f"✅ Loaded ensemble from {model_file}")
        # Force garbage collection to free up memory after loading heavy models
        gc.collect() 

    def save_model(self, path="saved_models"):
        os.makedirs(path, exist_ok=True)

        # Save training data separately for easier retraining/feedback
        joblib.dump({
            "X_train": self.feature_detector.X_train,
            "y_train": self.feature_detector.y_train
        }, f"{path}/training_data.pkl")

        # Save ensemble dictionary
        joblib.dump({
            "xgb_base_model": self.feature_detector.model,
            "xgb_model": self.feature_detector.calibrated_model,
            "X_train": self.feature_detector.X_train,
            "y_train": self.feature_detector.y_train,
            "codebert_clf": self.codebert_detector.clf,
            "codebert_embeddings": self.codebert_detector.train_embeddings,
            "codebert_y_train": self.codebert_detector.y_train,
            "weight_xgb": self.weights['features'],
            "weight_codebert": self.weights['codebert']
        }, f"{path}/hybrid_ai_detector_ensemble.pkl")
        
        joblib.dump(self.feature_detector.feature_names, f"{path}/feature_names_v2.pkl")

        print(f"💾 Models saved to `{path}/`")

    def analyze(self, code: str, metadata: Dict = None) -> Dict:
        """Complete analysis of code sample"""
        
        if not self.feature_detector.is_trained:
            raise RuntimeError(
                "HybridAIDetector not trained. Call train() first."
            )
        lang = self.feature_detector.feature_extractor.detect_language(code)

        # Get predictions from all components
        feature_result = self.feature_detector.predict(code)
        codebert_result = self.codebert_detector.predict(code)
        pattern_result = self.pattern_matcher.check(code)
        
        metadata_result = {'score': 0.5, 'flags': []}
        if metadata:
            metadata_result = self.metadata_analyzer.analyze(metadata)

        # --- WEIGHTED ENSEMBLE (0.8 XGB / 0.2 CodeBERT) ---
        feature_score = feature_result['ai_prob']
        codebert_score = codebert_result['ai_prob']

        final_score = (feature_score * self.weights['features']) + (codebert_score * self.weights['codebert'])
        
        # --- DECISION LOGIC (0.5 Threshold) ---
        prediction = "AI-generated" if final_score >= 0.5 else "Human-written"
        risk_level = self._assess_risk(final_score)

        # Generate explanation
        explanation = self._generate_explanation(
            feature_result, codebert_result, metadata_result, pattern_result
        )
        
        # --- CONFIDENCE CALCULATION ---
        confidence = abs(final_score - 0.5) * 2
        confidence = round(min(confidence, 1.0), 2)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "risk_level": risk_level,
            "language": lang,
            "final_score": final_score,
            "components": {
                "feature_based": feature_result,
                "codebert": codebert_result,
                "patterns": pattern_result,
            }
        }

    def detect(self, code: str) -> Dict:
        """
        High-level detection method for API usage.
        Returns a simplified result dictionary.
        """
        analysis = self.analyze(code)
        return {
            "prediction": analysis["prediction"],
            "confidence": analysis["confidence"],
            "language": analysis["language"]
        }

    def apply_feedback(self, feedback_samples: List[Dict]):
        """Update models with new labeled samples"""
        print("🔄 Applying feedback to models...")
        
        # 1. Update Feature-based model
        X_fb = self.feature_detector.feature_extractor.extract_features_batch([f["code"] for f in feedback_samples])
        y_fb = np.array([1 if f["true_label"] == "AI-generated" else 0 for f in feedback_samples])
        
        if not isinstance(self.feature_detector.X_train, pd.DataFrame):
            self.feature_detector.X_train = pd.DataFrame(self.feature_detector.X_train, columns=self.feature_detector.feature_names)

        self.feature_detector.X_train = pd.concat([self.feature_detector.X_train, X_fb], ignore_index=True)
        self.feature_detector.y_train = np.concatenate([self.feature_detector.y_train, y_fb])
        
        # Refit XGBoost and recalibrate
        self.feature_detector.model.fit(self.feature_detector.X_train, self.feature_detector.y_train)
        self.feature_detector.calibrated_model = CalibratedClassifierCV(self.feature_detector.model, method='isotonic', cv=5)
        self.feature_detector.calibrated_model.fit(self.feature_detector.X_train, self.feature_detector.y_train)
        
        # 2. Update CodeBERT classifier
        embeddings_fb = self.codebert_detector.get_embeddings([f["code"] for f in feedback_samples])
        self.codebert_detector.train_embeddings = torch.cat([self.codebert_detector.train_embeddings, embeddings_fb.cpu()], dim=0)
        self.codebert_detector.y_train = np.concatenate([self.codebert_detector.y_train, y_fb])
        
        self.codebert_detector.clf.fit(self.codebert_detector.train_embeddings.numpy(), self.codebert_detector.y_train)
        
        # Persist the updated model and training state
        self.save_model()
        
        print("✅ Feedback applied and models updated successfully!")
    
    def _generate_explanation(self, feature_result, codebert_result, metadata_result, pattern_result):
        """Generate human-readable explanation"""
        reasons = []
        
        if codebert_result['ai_prob'] > 0.8:
            reasons.append(f"CodeBERT semantic analysis shows {codebert_result['ai_prob']*100:.1f}% similarity to AI code")
        
        if feature_result['ai_prob'] > 0.7:
            reasons.append("Structural features match AI patterns")
        
        if metadata_result['flags']:
            reasons.append(f"Metadata flags: {', '.join(metadata_result['flags'])}")
        
        if pattern_result['likely_source']:
            reasons.append(f"Detected {pattern_result['likely_source']} patterns ({pattern_result['matches'][pattern_result['likely_source']]} matches)")
        
        return reasons if reasons else ["Inconclusive - mixed signals"]
    
    def _assess_risk(self, score):
        """Assess risk level"""
        if score > 0.9:
            return "VERY HIGH"
        elif score > 0.75:
            return "HIGH"
        elif score > 0.6:
            return "MEDIUM"
        elif score > 0.4:
            return "LOW"
        else:
            return "VERY LOW"
    
    def _check_agreement(self, feature_result, codebert_result):
        """Check if components agree"""
        feature_verdict = feature_result['ai_prob'] > 0.5
        codebert_verdict = codebert_result['ai_prob'] > 0.5
        return feature_verdict == codebert_verdict
    
    def evaluate(self, test_human: List[str], test_ai: List[str]):
        """Evaluate on test set"""
        print("\n" + "=" * 70)
        print("EVALUATION")
        print("=" * 70)
        
        predictions = []
        true_labels = []
        
        # Test human code
        for code in test_human:
            result = self.analyze(code)
            predictions.append(1 if result['prediction'] == 'AI-generated' else 0)
            true_labels.append(0)
        
        # Test AI code
        for code in test_ai:
            result = self.analyze(code)
            predictions.append(1 if result['prediction'] == 'AI-generated' else 0)
            true_labels.append(1)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        
        print(f"\n📊 Results:")
        print(f"   Accuracy:  {accuracy*100:.2f}%")
        print(f"   Precision: {precision*100:.2f}%")
        print(f"   Recall:    {recall*100:.2f}%")
        print(f"   F1 Score:  {f1:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        print(f"\n   Confusion Matrix:")
        print(f"                Predicted Human  Predicted AI")
        print(f"   Actual Human      {cm[0][0]:6d}          {cm[0][1]:6d}")
        print(f"   Actual AI         {cm[1][0]:6d}          {cm[1][1]:6d}")
        
        print("=" * 70)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist()
        }

# ============================================================================
# DEMO & USAGE
# ============================================================================

def demo():
    """Quick demo of the hybrid detector"""
    
    # Sample data (you'll replace with real data)
    sample_human_code = """
def fibonacci(n):
    # Quick fib calc
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n-1):
        a, b = b, a+b
    return b
    """
    
    sample_ai_code = """
def fibonacci(n):
    \"\"\"
    Calculate the nth Fibonacci number.
    
    This function implements the Fibonacci sequence calculation
    using an iterative approach for efficiency.
    
    Args:
        n (int): The position in the Fibonacci sequence
        
    Returns:
        int: The nth Fibonacci number
        
    Note:
        This implementation uses O(n) time complexity
    \"\"\"
    # Initialize base cases
    if n <= 1:
        return n
    
    # Helper variables to store previous values
    previous, current = 0, 1
    
    # Calculate Fibonacci iteratively
    for _ in range(n - 1):
        previous, current = current, previous + current
    
    return current
    """
    
    print("\n🔍 DEMO: Analyzing Sample Code\n")
    
    # Create detector
    detector = HybridAIDetector(use_gpu=False)
    
    # Train on minimal data (for demo)
    print("Training on sample data...\n")
    detector.train([sample_human_code] * 10, [sample_ai_code] * 10, epochs=1)
    
    # Analyze
    print("\n📝 Analyzing AI-generated code:")
    result = detector.analyze(sample_ai_code)
    print(json.dumps(result, indent=2))
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    demo()

_detector_instance = None

def detect(text: str):
    """
    Top-level function for easy API access.
    Handles singleton initialization of the HybridAIDetector.
    """
    global _detector_instance
    if _detector_instance is None:
        try:
            # Railway environment variable check
            use_gpu = os.environ.get("USE_GPU", "false").lower() == "true"
            print(f"Initializing detector (GPU={use_gpu})...")
            _detector_instance = HybridAIDetector(use_gpu=use_gpu)
            
            # Try multiple paths for Railway environment
            base_dir = Path(__file__).parent
            paths_to_try = [base_dir / "saved_models", base_dir]
            
            for p in paths_to_try:
                if (p / "hybrid_ai_detector_ensemble.pkl").exists() or (p / "hybrid_ai_detector.pkl").exists():
                    _detector_instance.load_pretrained(path=str(p))
                    break
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize detector: {str(e)}")
    return _detector_instance.detect(text)
