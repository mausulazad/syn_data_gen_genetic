import argparse

from inference import build_synthetic_dataset, generate_qars

def main():
    parser = argparse.ArgumentParser(description="A simple argparse example")

    parser.add_argument('--dataset', type=str, required=True, help='Seed Dataset')
    parser.add_argument('--generator_models', type=str, nargs=3, required=True, help='Model(s) for generating synthetic qars')
    parser.add_argument('--judge_model', type=str, required=True, help="Model for judging qar quality")
    parser.add_argument('--br_model', type=str, required=True, help="Model for backward reasoning")

    args = parser.parse_args()

    dataset = args.dataset 
    generator_models = args.generator_models
    judge_model = args.judge_model
    br_model = args.br_model

    generate_qars(dataset, generator_models, judge_model, br_model)
    #build_synthetic_dataset(dataset, generator_models, judge_model, br_model)

if __name__ == "__main__":
    main()