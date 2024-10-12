import os
import argparse
import pandas as pd
import requests
import torch

from model_utils import *
from methods import *
from specgem import *
from gpt4 import *

from huggingface_hub import login
access_token_read = ""
login(token = access_token_read)

def evaluate(
    spec,
    model_url,
    api_token,
    max_length,
    min_length,
    top_p,
    file_input_path,
    knowledge_base_path,
    file_output_name,
    method,
    classifier_features_path,
    icralm_type
):

    # Read specifications in CSV file
    knowledge_base_df = pd.read_csv(knowledge_base_path)

    # Use an input data ex. ELG
    story_data = pd.read_csv(file_input_path)
    prompt_list = story_data["prompt"].tolist() # prompt trxts
    level_list = story_data["level"].tolist() # CEFR or CCS level

    if method == "ic-ralm":
        model_generations = ic_ralm_generate(
            spec,
            model_url,
            api_token,
            max_length,
            top_p,
            prompt_list,
            level_list,
            knowledge_base_df,
            icralm_type
        )

        model_generations_df = pd.DataFrame({"level": level_list, "generated_story": model_generations})
        model_generations_df.to_csv(file_output_name, index=False)

    elif method == "simple-prompt":
        model_generations = simple_prompt(
            spec,
            model_url,
            api_token,
            max_length,
            top_p,
            prompt_list,
            level_list
        )

        model_generations_df = pd.DataFrame({"level": level_list, "generated_story": model_generations})
        model_generations_df.to_csv(file_output_name, index=False)

    elif method == "ic-ralm-hf":
        model_generations = ic_ralm_generate_hf(
            spec,
            model_url,
            api_token,
            max_length,
            min_length,
            top_p,
            prompt_list,
            level_list,
            knowledge_base_df,
            icralm_type
        )

        model_generations_df = pd.DataFrame({"level": level_list, "generated_story": model_generations})
        model_generations_df.to_csv(file_output_name, index=False)

    elif method == "simple-prompt-hf":
        model_generations = simple_prompt_generate_hf(
            spec,
            model_url,
            api_token,
            max_length,
            min_length,
            top_p,
            prompt_list,
            level_list,
            knowledge_base_df
        )

        model_generations_df = pd.DataFrame({"level": level_list, "generated_story": model_generations})
        model_generations_df.to_csv(file_output_name, index=False)

    elif method == "specgem-hf":

        # Read gold standard features for classifier, need for SpecGEM linguistic means
        # should come with CEFR/CCS levels
        classifier_features_df = pd.read_csv(classifier_features_path)
        
        model_generations = specgem_hf(
            spec,
            model_url,
            api_token,
            max_length,
            min_length,
            top_p,
            prompt_list,
            level_list,
            knowledge_base_df,
            classifier_features_df,
            icralm_type
        )

        model_generations_df = pd.DataFrame({"level": level_list, "generated_story": model_generations})
        model_generations_df.to_csv(file_output_name, index=False)

    elif method == "simple-prompt-openai":
        model_generations = simple_prompt_generate_openai(
            spec,
            model_url,
            api_token,
            max_length,
            min_length,
            top_p,
            prompt_list,
            level_list,
            knowledge_base_df
        )

        model_generations_df = pd.DataFrame({"level": level_list, "generated_story": model_generations})
        model_generations_df.to_csv(file_output_name, index=False)        

    elif method == "ic-ralm-openai":
        model_generations = ic_ralm_generate_openai(
            spec,
            model_url,
            api_token,
            max_length,
            min_length,
            top_p,
            prompt_list,
            level_list,
            knowledge_base_df,
            icralm_type
        )

        model_generations_df = pd.DataFrame({"level": level_list, "generated_story": model_generations})
        model_generations_df.to_csv(file_output_name, index=False)   

    elif method == "specgem-openai":

        # Read gold standard features for classifier, need for SpecGEM linguistic means
        # should come with CEFR/CCS levels
        classifier_features_df = pd.read_csv(classifier_features_path)
        
        model_generations = specgem_openai(
            spec,
            model_url,
            api_token,
            max_length,
            min_length,
            top_p,
            prompt_list,
            level_list,
            knowledge_base_df,
            classifier_features_df,
            icralm_type
        )

        model_generations_df = pd.DataFrame({"level": level_list, "generated_story": model_generations})
        model_generations_df.to_csv(file_output_name, index=False)   

def main(args):
    print_args(args) #dump args
    
    spec = args.spec
    model_url = args.model_api_url #API_URL of model for Inference API
    api_token = args.auth_token
    file_input_path = args.dataset_path
    max_length = args.max_length
    min_length = args.max_length
    top_p = args.top_p
    knowledge_base_path = args.knowledge_base_path
    temp_output_name = args.file_output_name
    file_output_name = temp_output_name
    method = args.method
    classifier_features_path = args.classifier_features_path
    icralm_type = args.icralm_type

    print("Finished loading the models...")
    print(file_output_name)
    print(type(file_output_name))

    evaluate(
        spec,
        model_url,
        api_token,
        max_length,
        min_length,
        top_p,
        file_input_path,
        knowledge_base_path,
        file_output_name,
        method,
        classifier_features_path,
        icralm_type
    )

if __name__ == '__main__':

    # Configure CUDA memory allocator
    torch.cuda.empty_cache()
    
    parser = argparse.ArgumentParser()

    #Setup params
    parser.add_argument("--spec", type=str, default="cefr") # cefr or ccs

    # File params
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--file_output_name", type=str, required=True)
    parser.add_argument("--knowledge_base_path", type=str, required=True)
    parser.add_argument("--classifier_features_path", type=str)

    # Model params
    parser.add_argument("--model_api_url", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=300)
    parser.add_argument("--min_length", type=int, default=30)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--auth_token", type=str, default="")

    # Method param
    parser.add_argument("--method", type=str, default="simple-prompt")
    parser.add_argument("--icralm_type", type=str, default="standard")

    args = parser.parse_args()
    main(args)
