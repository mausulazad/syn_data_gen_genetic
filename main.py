import argparse
from multiprocess import set_start_method

from inference import build_synthetic_dataset, build_dataset

def main():
    parser = argparse.ArgumentParser(description="A simple argparse example")

    parser.add_argument('--dataset', type=str, required=True, help='Seed Dataset')
    parser.add_argument('--generator_models', type=str, nargs=3, required=True, help='Model(s) for generating synthetic qars')
    parser.add_argument('--jury_models', type=str, nargs=2, required=True, help='Model(s) for evaluating and generating evolution methods for generated qars')
    #parser.add_argument('--judge_model', type=str, required=True, help="Model for judging qar quality")
    #parser.add_argument('--br_model', type=str, required=True, help="Model for backward reasoning")

    args = parser.parse_args()

    set_start_method("spawn", force=True)
    dataset = args.dataset 
    generator_models = args.generator_models
    jury_models = args.jury_models
    #judge_model = args.judge_model
    #br_model = args.br_model

    #generate_qars(dataset, generator_models, judge_model, br_model)
    #build_synthetic_dataset(dataset, generator_models, judge_model, br_model)
    build_dataset(dataset, generator_models, jury_models)

# TODO: debug prometheus vision inference
#python3 main.py --dataset aokvqa --generator_models llama_32 llava_next molmo --jury_models llava_critic qwen2_vl prometheus_vision

if __name__ == "__main__":
    main()