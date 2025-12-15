"""
Main entry point for sign language detection system.
"""
import argparse

def main():
    parser = argparse.ArgumentParser(description='Sign Language Detector')
    parser.add_argument('--mode', type=str, default='detect',
                       choices=['collect', 'train', 'detect'],
                       help='Mode: collect data, train model, or real-time detection')
    parser.add_argument('--label', type=str, help='Label for data collection')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of samples to collect')
    
    args = parser.parse_args()
    
    if args.mode == 'collect':
        from src.collect_data import collect_landmarks
        collect_landmarks(label=args.label, num_samples=args.samples)
    elif args.mode == 'train':
        from src.train_model import train_and_save_model
        train_and_save_model()
    elif args.mode == 'detect':
        from src.realtime_detect import realtime_detection
        realtime_detection()

if __name__ == "__main__":
    main()
