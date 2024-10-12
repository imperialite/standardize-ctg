import pandas as pd
import numpy as np
import re, time
import tiktoken
import os
from methods import *
from specgem import *

from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def call_gpt(transformed_prompt, added_tokens, gpt_model):

    while True:
        try:
            response = client.chat.completions.create(
                model = gpt_model,
                messages = [
                    {"role": "system", "content": "You are a good storyteller that can generate stories aligned with specifications based on text complexities provided. You will always output the generated story directly. "},
                    {"role": "user", "content": transformed_prompt}
                ],
                max_tokens = 300 + added_tokens
            )

            gpt_output = response.choices[0].message.content
            break

        except Exception as e:
            print("Sleeping due to timeout.")
            time.sleep(600)

    return gpt_output

def simple_prompt_generate_openai(spec, model_url, api_token, max_length, min_length, top_p, prompt_list, level_list, knowledge_base_df):
    
    gpt_model = "gpt-4-0125-preview"
    
    # Empty list to contain generations from model
    model_generations = []

    counter = 0
    for prompt, target_level in zip(prompt_list, level_list):

        transformed_prompt = ""

        if spec == 'cefr':
            transformed_prompt = simple_prompt_transform(prompt, spec, target_level)
            transformed_prompt = prompt_format(transformed_prompt, model_url)

        elif spec == 'ccs':
            transformed_prompt = simple_prompt_transform(prompt, spec, target_level)
            transformed_prompt = prompt_format(transformed_prompt, model_url)

        added_tokens = num_tokens_from_string(transformed_prompt,gpt_model)

        gpt_output = call_gpt(transformed_prompt, added_tokens, gpt_model)
        
        model_generations.append(gpt_output)
        counter += 1
        print(counter)
    
    return model_generations


def ic_ralm_generate_openai(spec, model_url, api_token, max_length, min_length, top_p, prompt_list, level_list, knowledge_base_df, icralm_type):

    gpt_model = "gpt-4-0125-preview"
    
    # Empty list to contain generations from model
    model_generations = []

    counter = 0
    for prompt, target_level in zip(prompt_list, level_list):

        transformed_prompt = prompt #temporary

        if spec == 'cefr':
            print("HERE AT CEFR SPEC")
            if icralm_type == 'standard':
                spec_dict = query_cefr_standard(target_level, knowledge_base_df)
                transformed_prompt = cefr_prompt_transform_standard(prompt, spec, spec_dict, target_level)
            
            elif icralm_type == 'exemplar':
                spec_dict = query_cefr_exemplar(target_level, knowledge_base_df)
                transformed_prompt = cefr_prompt_transform_exemplar(prompt, spec, spec_dict, target_level)

        elif spec == 'ccs':
            print("HERE AT CCS SPEC")
            if icralm_type == 'standard':
                spec_dict = query_ccs_standard(target_level, knowledge_base_df)
                transformed_prompt = ccs_prompt_transform_standard(prompt, spec, spec_dict, target_level)

            elif icralm_type == 'exemplar':
                spec_dict = query_ccs_exemplar(target_level, knowledge_base_df)
                transformed_prompt = ccs_prompt_transform_exemplar(prompt, spec, spec_dict, target_level)
              
        transformed_prompt = prompt_format(transformed_prompt, model_url)
        
        added_tokens = num_tokens_from_string(transformed_prompt,gpt_model)
        gpt_output = call_gpt(transformed_prompt, added_tokens, gpt_model)

        model_generations.append(gpt_output)
        
        counter += 1
        print(counter)
    
    return model_generations


def specgem_openai(spec, model_url, api_token, max_length, min_length, top_p, prompt_list, level_list, knowledge_base_df, classifier_features_df, icralm_type):

    gpt_model = "gpt-4-0125-preview"

    # Empty list to contain generations from model
    model_generations = []

    max_generation_iter = 2 # (n+1) rounds of rewrite
    counter = 0

    for prompt, target_level in zip(prompt_list, level_list):

        current_generated_text = "" # temp value
        tracker = 1 # temp value
        
        if spec == 'cefr':

            if icralm_type == 'standard':
                spec_dict = query_cefr_standard(target_level, knowledge_base_df)
                transformed_prompt = cefr_prompt_transform_standard(prompt, spec, spec_dict, target_level)
                transformed_prompt = prompt_format(transformed_prompt, model_url)
            
            elif icralm_type == 'exemplar':
                spec_dict = query_cefr_exemplar(target_level, knowledge_base_df)
                transformed_prompt = cefr_prompt_transform_exemplar(prompt, spec, spec_dict, target_level)
                transformed_prompt = prompt_format(transformed_prompt, model_url)

            elif icralm_type == 'all':
                spec_dict_aspect = query_cefr_standard(target_level, knowledge_base_df)
                spec_dict_exemplar = query_cefr_exemplar(target_level, knowledge_base_df)
                transformed_prompt = cefr_prompt_transform_all(prompt, spec, spec_dict_aspect, spec_dict_exemplar, target_level)
                transformed_prompt = prompt_format(transformed_prompt, model_url)

        elif spec == 'ccs':
            
            if icralm_type == 'standard':
                spec_dict = query_ccs_standard(target_level, knowledge_base_df)
                transformed_prompt = ccs_prompt_transform_standard(prompt, spec, spec_dict, target_level)
                transformed_prompt = prompt_format(transformed_prompt, model_url)

            elif icralm_type == 'exemplar':
                spec_dict = query_cefr_exemplar(target_level, knowledge_base_df)
                transformed_prompt = ccs_prompt_transform_standard(prompt, spec, spec_dict, target_level)
                transformed_prompt = prompt_format(transformed_prompt, model_url)

            elif icralm_type == 'all':
                spec_dict_aspect = query_ccs_standard(target_level, knowledge_base_df)
                spec_dict_exemplar = query_ccs_exemplar(target_level, knowledge_base_df)
                transformed_prompt = ccs_prompt_transform_all(prompt, spec, spec_dict_aspect, spec_dict_exemplar, target_level)
                transformed_prompt = prompt_format(transformed_prompt, model_url)
            
        ## GENERATION PART

        for generation_iter in range(max_generation_iter):

            current_generated_text = re.sub("\n+"," ",current_generated_text) # a bit of cleaning

            if generation_iter == 0:

                added_tokens = num_tokens_from_string(current_generated_text,gpt_model)
                current_generated_text = call_gpt(transformed_prompt, added_tokens, gpt_model)

            # we cap the number of rewrites based on max_generation_iter
            elif generation_iter > 0:
                print("Here at rewrite round.")

                # transformed_prompt = remove_prompt_format(current_generated_text, model_url)
                transformed_prompt, tracker = specgem_prompt(transformed_prompt, spec, classifier_features_df, target_level, "rewrite")
                transformed_prompt = prompt_format(transformed_prompt, model_url)
                
                # however, if there are no more changes left from linguistic prompts, stop iterating
                if tracker == 0:
                    break

                added_tokens = num_tokens_from_string(transformed_prompt,gpt_model)
                current_generated_text = call_gpt(transformed_prompt, added_tokens, gpt_model)
                 
        #print(current_generated_text)
        model_generations.append(current_generated_text)
        
        counter += 1
        print(counter)

    return model_generations