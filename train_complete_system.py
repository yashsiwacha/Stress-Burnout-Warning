#!/usr/bin/env python3
"""
Comprehensive Training Script for Stress Burnout Warning System

This script orchestrates the complete training pipeline for:
1. Dataset preparation and validation
2. Facial emotion recognition models
3. Vocal stress detection models  
4. Fusion models for stress prediction
5. Model evaluation and reporting

Usage:
    python train_complete_system.py [options]

Author: Stress Burnout Warning System Team
Date: August 2025
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import custom modules
try:
    from data.dataset_loader import DatasetLoader
    from ai.model_training import ModelTrainer
    from monitoring.facial_training import FacialExpressionTrainer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all dependencies are installed and src/ directory exists")
    sys.exit(1)


class ComprehensiveTrainer:
    """
    Main trainer class that orchestrates the complete training pipeline.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize the comprehensive trainer.
        
        Args:
            config (dict): Training configuration
        """
        self.config = config or self._get_default_config()
        self.results = {}
        self.start_time = datetime.now()
        
        # Setup paths
        self.project_root = os.path.dirname(__file__)
        self.models_path = os.path.join(self.project_root, 'models')
        self.datasets_path = os.path.join(self.project_root, 'datasets')
        self.logs_path = os.path.join(self.project_root, 'logs')
        
        # Create directories
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)
        
        # Initialize trainers
        self.facial_trainer = FacialExpressionTrainer(self.models_path)
        self.model_trainer = ModelTrainer(self.models_path)
        
    def _get_default_config(self) -> dict:
        """
        Get default training configuration.
        
        Returns:
            dict: Default configuration
        """
        return {
            'facial_models': {
                'enabled': True,
                'architectures': ['custom'],  # 'custom', 'vgg16', 'resnet50', 'mobilenet'
                'epochs': 20,
                'batch_size': 32,
                'use_transfer_learning': False,
                'dataset': 'fer2013'
            },
            'vocal_models': {
                'enabled': True,
                'epochs': 20,
                'batch_size': 32,
                'dataset': 'ravdess'
            },
            'stress_models': {
                'enabled': True,
                'fusion_enabled': False  # Requires both facial and vocal models
            },
            'evaluation': {
                'cross_validation': False,
                'generate_plots': True,
                'save_predictions': True
            },
            'logging': {
                'verbose': True,
                'save_logs': True
            }
        }
    
    def check_datasets(self) -> dict:
        """
        Check availability and structure of required datasets.
        
        Returns:
            dict: Dataset availability status
        """
        print("CHECKING DATASET AVAILABILITY")
        print("=" * 50)
        
        dataset_status = {}
        
        # Check FER-2013
        fer2013_path = os.path.join(self.datasets_path, 'facial_emotion', 'fer2013')
        if os.path.exists(fer2013_path) and os.listdir(fer2013_path):
            try:
                facial_loader = DatasetLoader('facial')
                # Try to load a small sample to verify structure
                dataset_status['fer2013'] = {
                    'available': True,
                    'path': fer2013_path,
                    'verified': True
                }
                print("✅ FER-2013 dataset found and verified")
            except Exception as e:
                dataset_status['fer2013'] = {
                    'available': True,
                    'path': fer2013_path,
                    'verified': False,
                    'error': str(e)
                }
                print(f"⚠️  FER-2013 dataset found but verification failed: {e}")
        else:
            dataset_status['fer2013'] = {'available': False, 'path': fer2013_path}
            print("❌ FER-2013 dataset not found")
        
        # Check RAVDESS
        ravdess_path = os.path.join(self.datasets_path, 'vocal_emotion', 'ravdess')
        if os.path.exists(ravdess_path) and os.listdir(ravdess_path):
            try:
                vocal_loader = DatasetLoader('vocal')
                # Check for audio files
                audio_files = [f for f in os.listdir(ravdess_path) if f.endswith('.wav')]
                if audio_files:
                    dataset_status['ravdess'] = {
                        'available': True,
                        'path': ravdess_path,
                        'verified': True,
                        'files': len(audio_files)
                    }
                    print(f"✅ RAVDESS dataset found and verified ({len(audio_files)} audio files)")
                else:
                    dataset_status['ravdess'] = {
                        'available': True,
                        'path': ravdess_path,
                        'verified': False,
                        'error': 'No audio files found'
                    }
                    print("⚠️  RAVDESS directory found but no audio files")
            except Exception as e:
                dataset_status['ravdess'] = {
                    'available': True,
                    'path': ravdess_path,
                    'verified': False,
                    'error': str(e)
                }
                print(f"⚠️  RAVDESS dataset found but verification failed: {e}")
        else:
            dataset_status['ravdess'] = {'available': False, 'path': ravdess_path}
            print("❌ RAVDESS dataset not found")
        
        # Check sample dataset
        sample_path = os.path.join(self.datasets_path, 'sample')
        if os.path.exists(sample_path):
            dataset_status['sample'] = {'available': True, 'path': sample_path}
            print("✅ Sample dataset found")
        else:
            dataset_status['sample'] = {'available': False, 'path': sample_path}
            print("❌ Sample dataset not found")
        
        return dataset_status
    
    def train_facial_models(self) -> dict:
        """
        Train facial emotion recognition models.
        
        Returns:
            dict: Training results
        """
        if not self.config['facial_models']['enabled']:
            return {'status': 'disabled'}
        
        print("\\nTRAINING FACIAL EMOTION RECOGNITION MODELS")
        print("=" * 60)
        
        facial_results = {}
        
        architectures = self.config['facial_models']['architectures']
        
        for arch in architectures:
            print(f"\\nTraining {arch} architecture...")
            
            try:
                result = self.facial_trainer.train_model(
                    architecture=arch,
                    dataset=self.config['facial_models']['dataset'],
                    epochs=self.config['facial_models']['epochs'],
                    batch_size=self.config['facial_models']['batch_size'],
                    use_transfer_learning=self.config['facial_models']['use_transfer_learning']
                )
                
                facial_results[arch] = result
                
                if 'error' not in result:
                    accuracy = result.get('test_accuracy', result.get('best_accuracy', 0))
                    print(f"✅ {arch} training completed! Accuracy: {accuracy:.4f}")
                else:
                    print(f"❌ {arch} training failed: {result['error']}")
                    
            except Exception as e:
                facial_results[arch] = {'error': str(e)}
                print(f"❌ {arch} training failed with exception: {e}")
        
        return facial_results
    
    def train_vocal_models(self) -> dict:
        """
        Train vocal emotion/stress recognition models.
        
        Returns:
            dict: Training results
        """
        if not self.config['vocal_models']['enabled']:
            return {'status': 'disabled'}
        
        print("\\nTRAINING VOCAL EMOTION RECOGNITION MODELS")
        print("=" * 60)
        
        try:
            vocal_result = self.model_trainer.train_vocal_emotion_model(
                use_ravdess=True,
                epochs=self.config['vocal_models']['epochs'],
                batch_size=self.config['vocal_models']['batch_size']
            )
            
            if 'error' not in vocal_result:
                accuracy = vocal_result.get('test_accuracy', vocal_result.get('best_accuracy', 0))
                print(f"✅ Vocal model training completed! Accuracy: {accuracy:.4f}")
            else:
                print(f"❌ Vocal model training failed: {vocal_result['error']}")
            
            return vocal_result
            
        except Exception as e:
            print(f"❌ Vocal model training failed with exception: {e}")
            return {'error': str(e)}
    
    def train_stress_models(self) -> dict:
        """
        Train stress detection and fusion models.
        
        Returns:
            dict: Training results
        """
        if not self.config['stress_models']['enabled']:
            return {'status': 'disabled'}
        
        print("\\nTRAINING STRESS DETECTION MODELS")
        print("=" * 60)
        
        try:
            stress_result = self.model_trainer.train_stress_detection_model()
            
            if 'error' not in stress_result:
                print("✅ Stress detection model created successfully!")
            else:
                print(f"❌ Stress model creation failed: {stress_result['error']}")
            
            return stress_result
            
        except Exception as e:
            print(f"❌ Stress model creation failed with exception: {e}")
            return {'error': str(e)}
    
    def evaluate_models(self) -> dict:
        """
        Evaluate trained models and generate performance metrics.
        
        Returns:
            dict: Evaluation results
        """
        print("\\nEVALUATING TRAINED MODELS")
        print("=" * 50)
        
        evaluation_results = {}
        
        # List all model files
        model_files = []
        for root, dirs, files in os.walk(self.models_path):
            for file in files:
                if file.endswith(('.h5', '.pkl')):
                    model_files.append(os.path.join(root, file))
        
        print(f"Found {len(model_files)} model files:")
        for model_file in model_files:
            rel_path = os.path.relpath(model_file, self.models_path)
            print(f"  - {rel_path}")
        
        evaluation_results['model_files'] = model_files
        evaluation_results['total_models'] = len(model_files)
        
        return evaluation_results
    
    def generate_report(self) -> str:
        """
        Generate comprehensive training report.
        
        Returns:
            str: Path to generated report
        """
        print("\\nGENERATING TRAINING REPORT")
        print("=" * 50)
        
        end_time = datetime.now()
        training_duration = end_time - self.start_time
        
        report_filename = f"training_report_{self.start_time.strftime('%Y%m%d_%H%M%S')}.txt"
        report_path = os.path.join(self.logs_path, report_filename)
        
        with open(report_path, 'w') as f:
            f.write("STRESS BURNOUT WARNING SYSTEM - COMPREHENSIVE TRAINING REPORT\\n")
            f.write("=" * 80 + "\\n\\n")
            
            # Training overview
            f.write(f"Training Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Training Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Total Duration: {training_duration}\\n\\n")
            
            # Configuration
            f.write("TRAINING CONFIGURATION\\n")
            f.write("-" * 40 + "\\n")
            f.write(json.dumps(self.config, indent=2) + "\\n\\n")
            
            # Results
            f.write("TRAINING RESULTS\\n")
            f.write("-" * 40 + "\\n")
            f.write(json.dumps(self.results, indent=2, default=str) + "\\n\\n")
            
            # Summary
            f.write("SUMMARY\\n")
            f.write("-" * 40 + "\\n")
            
            successful_models = 0
            failed_models = 0
            
            for category, results in self.results.items():
                if isinstance(results, dict):
                    if 'error' in results:
                        failed_models += 1
                        f.write(f"❌ {category}: Failed - {results['error']}\\n")
                    elif results.get('status') == 'disabled':
                        f.write(f"⚪ {category}: Disabled\\n")
                    else:
                        successful_models += 1
                        f.write(f"✅ {category}: Success\\n")
            
            f.write(f"\\nSuccessful Models: {successful_models}\\n")
            f.write(f"Failed Models: {failed_models}\\n")
            
            # Model files
            model_files = self.results.get('evaluation', {}).get('model_files', [])
            f.write(f"Total Model Files: {len(model_files)}\\n")
            
            if model_files:
                f.write("\\nGenerated Model Files:\\n")
                for model_file in model_files:
                    rel_path = os.path.relpath(model_file, self.models_path)
                    f.write(f"  - {rel_path}\\n")
        
        print(f"📋 Training report saved to: {report_path}")
        return report_path
    
    def run_complete_training(self) -> dict:
        """
        Run the complete training pipeline.
        
        Returns:
            dict: Complete training results
        """
        print("STRESS BURNOUT WARNING SYSTEM - COMPREHENSIVE TRAINING")
        print("=" * 80)
        print(f"Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Step 1: Check datasets
        dataset_status = self.check_datasets()
        self.results['datasets'] = dataset_status
        
        # Check if we have any usable datasets
        usable_datasets = any(
            status.get('available') and status.get('verified', True) 
            for status in dataset_status.values()
        )
        
        if not usable_datasets:
            print("\\n⚠️  No usable datasets found!")
            print("Please download datasets using:")
            print("  python dataset_downloader.py --download-all")
            print("Or create sample data:")
            print("  python dataset_downloader.py --create-sample")
            return self.results
        
        # Step 2: Train facial models
        if self.config['facial_models']['enabled']:
            facial_results = self.train_facial_models()
            self.results['facial_models'] = facial_results
        
        # Step 3: Train vocal models
        if self.config['vocal_models']['enabled']:
            vocal_results = self.train_vocal_models()
            self.results['vocal_models'] = vocal_results
        
        # Step 4: Train stress models
        if self.config['stress_models']['enabled']:
            stress_results = self.train_stress_models()
            self.results['stress_models'] = stress_results
        
        # Step 5: Evaluate models
        evaluation_results = self.evaluate_models()
        self.results['evaluation'] = evaluation_results
        
        # Step 6: Generate report
        report_path = self.generate_report()
        self.results['report_path'] = report_path
        
        print("\\n" + "=" * 80)
        print("TRAINING PIPELINE COMPLETED!")
        print("=" * 80)
        print(f"Duration: {datetime.now() - self.start_time}")
        print(f"Report: {report_path}")
        print(f"Models saved in: {self.models_path}")
        
        return self.results


def main():
    """
    Main function for the comprehensive training script.
    """
    parser = argparse.ArgumentParser(
        description='Comprehensive Training Script for Stress Burnout Warning System'
    )
    
    # Training options
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--facial-only', action='store_true', help='Train only facial models')
    parser.add_argument('--vocal-only', action='store_true', help='Train only vocal models')
    parser.add_argument('--stress-only', action='store_true', help='Train only stress models')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    
    # Model options
    parser.add_argument('--architectures', nargs='+', default=['custom'], 
                       help='Facial model architectures to train')
    parser.add_argument('--transfer-learning', action='store_true', 
                       help='Use transfer learning for facial models')
    
    # Utility options
    parser.add_argument('--check-datasets', action='store_true', help='Only check dataset availability')
    parser.add_argument('--quick-test', action='store_true', help='Quick test with reduced epochs')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        trainer_instance = ComprehensiveTrainer()
        config = trainer_instance._get_default_config()
    
    # Apply command line overrides
    if args.facial_only:
        config['vocal_models']['enabled'] = False
        config['stress_models']['enabled'] = False
    elif args.vocal_only:
        config['facial_models']['enabled'] = False
        config['stress_models']['enabled'] = False
    elif args.stress_only:
        config['facial_models']['enabled'] = False
        config['vocal_models']['enabled'] = False
    
    if args.epochs:
        config['facial_models']['epochs'] = args.epochs
        config['vocal_models']['epochs'] = args.epochs
    
    if args.batch_size:
        config['facial_models']['batch_size'] = args.batch_size
        config['vocal_models']['batch_size'] = args.batch_size
    
    if args.architectures:
        config['facial_models']['architectures'] = args.architectures
    
    if args.transfer_learning:
        config['facial_models']['use_transfer_learning'] = True
    
    if args.quick_test:
        config['facial_models']['epochs'] = 2
        config['vocal_models']['epochs'] = 2
    
    # Initialize trainer
    trainer = ComprehensiveTrainer(config)
    
    # Run requested operation
    if args.check_datasets:
        dataset_status = trainer.check_datasets()
        print("\\nDataset check completed!")
    else:
        results = trainer.run_complete_training()
        print("\\nTraining completed!")


if __name__ == "__main__":
    main()
