#!/usr/bin/env python3
"""
Dataset Validator for AI Code Detector
Checks quality and balance of training data
"""

from pathlib import Path
import ast
import json

class DatasetValidator:
    """Validate training dataset quality"""
    
    def __init__(self, data_dir='training_data'):
        self.data_dir = Path(data_dir)
        self.human_dir = self.data_dir / 'human'
        self.ai_dir = self.data_dir / 'ai'
    
    def validate_file(self, filepath):
        """Validate a single Python file"""
        issues = []
        
        try:
            code = filepath.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            return {'valid': False, 'issues': [f"Cannot read file: {e}"]}
        
        # Check 1: Minimum length
        if len(code) < 50:
            issues.append("Too short (< 50 characters)")
        
        # Check 2: Valid Python syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
        
        # Check 3: Not empty
        if not code.strip():
            issues.append("Empty file")
        
        # Check 4: Has actual code (not just comments)
        lines = [l.strip() for l in code.split('\n')]
        code_lines = [l for l in lines if l and not l.startswith('#')]
        if len(code_lines) < 5:
            issues.append("Too few code lines (< 5)")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'lines': len(code.split('\n')),
            'characters': len(code)
        }
    
    def analyze_dataset(self):
        """Analyze complete dataset"""
        
        print("=" * 70)
        print("DATASET VALIDATION")
        print("=" * 70)
        print()
        
        # Check human samples
        print("📊 HUMAN CODE SAMPLES")
        print("-" * 70)
        human_files = list(self.human_dir.glob('*.py'))
        human_stats = self._analyze_category(human_files, "Human")
        
        print()
        
        # Check AI samples  
        print("📊 AI CODE SAMPLES")
        print("-" * 70)
        ai_files = list(self.ai_dir.glob('*.py'))
        ai_stats = self._analyze_category(ai_files, "AI")
        
        print()
        
        # Overall assessment
        self._print_overall_assessment(human_stats, ai_stats)
        
        return {
            'human': human_stats,
            'ai': ai_stats
        }
    
    def _analyze_category(self, files, category):
        """Analyze files in a category"""
        
        total = len(files)
        valid = 0
        invalid = 0
        total_lines = 0
        total_chars = 0
        issues_count = {}
        
        print(f"Total files: {total}")
        
        if total == 0:
            print("⚠️  No files found!")
            return {'total': 0, 'valid': 0}
        
        for filepath in files:
            result = self.validate_file(filepath)
            
            if result['valid']:
                valid += 1
                total_lines += result['lines']
                total_chars += result['characters']
            else:
                invalid += 1
                for issue in result['issues']:
                    issues_count[issue] = issues_count.get(issue, 0) + 1
        
        print(f"Valid files: {valid} ({valid/total*100:.1f}%)")
        print(f"Invalid files: {invalid}")
        
        if valid > 0:
            print(f"Avg lines/file: {total_lines/valid:.1f}")
            print(f"Avg chars/file: {total_chars/valid:.1f}")
        
        if issues_count:
            print(f"\nCommon issues:")
            for issue, count in sorted(issues_count.items(), key=lambda x: -x[1]):
                print(f"  - {issue}: {count} files")
        
        return {
            'total': total,
            'valid': valid,
            'invalid': invalid,
            'avg_lines': total_lines/valid if valid > 0 else 0,
            'avg_chars': total_chars/valid if valid > 0 else 0,
            'issues': issues_count
        }
    
    def _print_overall_assessment(self, human_stats, ai_stats):
        """Print overall dataset assessment"""
        
        print("=" * 70)
        print("OVERALL ASSESSMENT")
        print("=" * 70)
        print()
        
        total_samples = human_stats['valid'] + ai_stats['valid']
        balance = min(human_stats['valid'], ai_stats['valid']) / max(human_stats['valid'], ai_stats['valid'], 1)
        
        print(f"✅ Total valid samples: {total_samples}")
        print(f"   - Human: {human_stats['valid']}")
        print(f"   - AI: {ai_stats['valid']}")
        print()
        
        print(f"⚖️  Balance ratio: {balance:.2f}")
        if balance < 0.7:
            print("   ⚠️  WARNING: Imbalanced dataset!")
            print("   Recommendation: Add more samples to smaller category")
        elif balance >= 0.9:
            print("   ✅ Well balanced!")
        else:
            print("   ✅ Acceptable balance")
        print()
        
        # Recommendations
        print("📋 RECOMMENDATIONS:")
        print()
        
        if total_samples < 100:
            print("❌ INSUFFICIENT DATA")
            print(f"   Current: {total_samples} samples")
            print(f"   Minimum: 100 samples (50 human + 50 AI)")
            print(f"   Recommended: 200 samples (100 human + 100 AI)")
            print(f"   Optimal: 400+ samples (200 human + 200 AI)")
            print()
            print(f"   ACTION NEEDED: Collect {100 - total_samples} more samples")
        
        elif total_samples < 200:
            print("⚠️  MINIMUM DATASET")
            print(f"   Current: {total_samples} samples")
            print(f"   This will work but accuracy may be limited")
            print(f"   Expected accuracy: 75-85%")
            print()
            print(f"   RECOMMENDED: Add {200 - total_samples} more samples for better results")
        
        elif total_samples < 400:
            print("✅ GOOD DATASET")
            print(f"   Current: {total_samples} samples")
            print(f"   Expected accuracy: 85-92%")
            print()
            print(f"   OPTIONAL: Add more samples for production deployment")
        
        else:
            print("🌟 EXCELLENT DATASET")
            print(f"   Current: {total_samples} samples")
            print(f"   Expected accuracy: 91-94%")
            print(f"   Ready for production training!")
        
        print()
        
        # Next steps
        if total_samples >= 100:
            print("✅ READY TO TRAIN!")
            print()
            print("Next steps:")
            print("  1. Run: python hybrid_ai_detector.py")
            print("  2. Wait for training to complete (10-15 hours)")
            print("  3. Evaluate results")
            print("  4. Deploy!")
        
        print()
        print("=" * 70)
    
    def export_report(self, filename='dataset_report.json'):
        """Export validation report"""
        
        human_files = list(self.human_dir.glob('*.py'))
        ai_files = list(self.ai_dir.glob('*.py'))
        
        report = {
            'timestamp': str(Path.cwd()),
            'human': {
                'total': len(human_files),
                'files': [str(f.name) for f in human_files]
            },
            'ai': {
                'total': len(ai_files),
                'files': [str(f.name) for f in ai_files]
            }
        }
        
        report_path = self.data_dir / filename
        report_path.write_text(json.dumps(report, indent=2))
        
        print(f"\n📄 Report exported to: {report_path}")

def main():
    """Main execution"""
    
    validator = DatasetValidator()
    
    # Check if directories exist
    if not validator.human_dir.exists():
        print(f"❌ Human code directory not found: {validator.human_dir}")
        print("   Create it first: mkdir -p training_data/human")
        return
    
    if not validator.ai_dir.exists():
        print(f"❌ AI code directory not found: {validator.ai_dir}")
        print("   Create it first: mkdir -p training_data/ai")
        return
    
    # Validate
    results = validator.analyze_dataset()
    
    # Export report
    validator.export_report()

if __name__ == "__main__":
    main()
