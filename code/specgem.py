import torch
import requests
import time
import pandas as pd
import numpy as np
import spacy
import lftk
import re, statistics, string

from methods import prompt_format

# don't forget to download python -m spacy download en_core_web_sm

from model_utils import *
from methods import *

def extract_linguistic_flags(text, spec):
  nlp = spacy.load("en_core_web_sm")
  doc = nlp(text)
  LFTK = lftk.Extractor(docs = doc)
  LFTK.customize(stop_words=False, punctuations=False, round_decimal=3)
  
  if spec == "cefr":
    syntax_structure = ['t_word', 't_sent', 'a_word_ps', 'n_ucconj', 'n_sconj'] # total word count, total sentence count, average sentence length
    grammatical = ['auto', 'corr_ttr'] # ARI readability, corrected TTR
    meaning_purpose = ['a_n_ent_ps', 'a_kup_ps', 'a_subtlex_us_zipf_ps'] # ave entities per sentence, average AoA per sentence, average USubtlex per sentence

    feature_list = syntax_structure + grammatical + meaning_purpose
    extracted_features = LFTK.extract(features = feature_list)
    #print(extracted_features)
    #print(type(extracted_features))

    linguistic_flag_dict = {
        'Automated Readability Index score' : extracted_features['auto'],
        'type token ratio' : extracted_features['corr_ttr'],
        'total number of words' : extracted_features['t_word'],
        'total number of sentences' : extracted_features['t_sent'],
        'average sentence length' : extracted_features['a_word_ps'],
        'total number of conjunctions' : extracted_features['n_ucconj'] + extracted_features['n_sconj'],
        'average word familiarity score' : extracted_features['a_kup_ps'],
        'average number of entities' : extracted_features['a_n_ent_ps'],
    }

  elif spec == "ccs":
    quanti_length = ['t_word', 't_sent', 'a_word_ps'] # total word count, total sentence count, average sentence length
    quali_struct = ['corr_ttr', 'n_ucconj', 'n_sconj'] # coordinating and subordinating, corrected TTR
    quali_meaning = ['a_n_ent_ps', 'n_upropn'] # ave entities per sentence, total number of unique proper nouns

    feature_list = quanti_length + quali_struct + quali_meaning
    extracted_features = LFTK.extract(features = feature_list)
    #print(extracted_features)
    #print(type(extracted_features))

    linguistic_flag_dict = {
        'type token ratio' : extracted_features['corr_ttr'],
        'total number of words' : extracted_features['t_word'],
        'total number of sentences' : extracted_features['t_sent'],
        'average sentence length' : extracted_features['a_word_ps'],
        'total number of conjunctions' : extracted_features['n_ucconj'] + extracted_features['n_sconj'],
        'total number of unique proper nouns' : extracted_features['n_upropn'],
        'average number of entities' : extracted_features['a_n_ent_ps']
    }

  return linguistic_flag_dict

"""
SPECGEM Prompt Formatter
"""

def specgem_prompt(prompt, spec, unselected_classifier_features_df, target_level, iteration_flag):

    # Select dataframe of features based on target level
    classifier_features_df = unselected_classifier_features_df[unselected_classifier_features_df['level']==target_level]
    #print(len(classifier_features_df['level']))

    transformed_prompt = """Given this story: {narrative}.

    
    Continue the story and make sure it is readable for {target_level} learners in the {spec} scale. 
    Use the following specifications to reach the target level of the story:
    """

    counter = 1
    tracker = 0 # this counts how many items for rewriting is needed
    target_level_str = "" # for string version of CCS levels

    if spec == 'cefr':
        transformed_prompt = transformed_prompt.replace("{spec}", "CEFR")
        transformed_prompt = transformed_prompt.replace("{narrative}", prompt)
        transformed_prompt = transformed_prompt.replace("{target_level}", target_level)
    
    elif spec == 'ccs':
        if target_level == 1:
            target_level_str = "6 to 8"
        else:
            target_level_str = "9 to 12"
        transformed_prompt = transformed_prompt.replace("{spec}", "Common Core Standards")
        transformed_prompt = transformed_prompt.replace("{narrative}", prompt)
        transformed_prompt = transformed_prompt.replace("{target_level}", target_level_str)
    
    # Extract linguistic flags from prompt
    prompt_linguistic_flag_dict = extract_linguistic_flags(prompt, spec)
    
    if spec == "cefr":
    
        # Extract gold standard MEAN values of linguistic flags from existing features to train classified 
        goldstandard_linguistic_flag_mean_dict = {
        #'Automated Readability Index score' : classifier_features_df['auto'].mean(),
        #'type token ratio' : classifier_features_df['corr_ttr'].mean(),
        'total number of words' : classifier_features_df['t_word'].mean(),
        'total number of sentences' : classifier_features_df['t_sent'].mean(),
        'average sentence length' : classifier_features_df['a_word_ps'].mean(),
        'total number of conjunctions' : classifier_features_df['n_ucconj'].mean() + classifier_features_df['n_sconj'].mean()
        #'average word familiarity score' : classifier_features_df['a_kup_ps'].mean(),
        #'average number of entities' : classifier_features_df['a_n_ent_ps'].mean()
        }

        # Extract gold standard STANDARD DEVIATION values of linguistic flags from existing features to train classified 
        goldstandard_linguistic_flag_std_dict = {
        #'Automated Readability Index score' : classifier_features_df['auto'].std(),
        #'type token ratio' : classifier_features_df['corr_ttr'].std(),
        'total number of words' : classifier_features_df['t_word'].std(),
        'total number of sentences' : classifier_features_df['t_sent'].std(),
        'average sentence length' : classifier_features_df['a_word_ps'].std(),
        'total number of conjunctions' : classifier_features_df['n_ucconj'].std() + classifier_features_df['n_sconj'].std()
        #'average word familiarity score' : classifier_features_df['a_kup_ps'].std(),
        #'average number of entities' : classifier_features_df['a_n_ent_ps'].std()
        }
    
    elif spec == 'ccs':

        # Extract gold standard MEAN values of linguistic flags from existing features to train classified 
        goldstandard_linguistic_flag_mean_dict = {
        'type token ratio' : classifier_features_df['corr_ttr'].mean(),
        'total number of words' : classifier_features_df['t_word'].mean(),
        'total number of sentences' : classifier_features_df['t_sent'].mean(),
        'average sentence length' : classifier_features_df['a_word_ps'].mean(),
        'total number of conjunctions' : classifier_features_df['n_ucconj'].mean() + classifier_features_df['n_sconj'].mean(),
        'total number of unique proper nouns' : classifier_features_df['n_upropn'].mean(),
        'average number of entities' : classifier_features_df['a_n_ent_ps'].mean()
        }

        # Extract gold standard STANDARD DEVIATION values of linguistic flags from existing features to train classified 
        goldstandard_linguistic_flag_std_dict = {
        'type token ratio' : classifier_features_df['corr_ttr'].std(),
        'total number of words' : classifier_features_df['t_word'].std(),
        'total number of sentences' : classifier_features_df['t_sent'].std(),
        'average sentence length' : classifier_features_df['a_word_ps'].std(),
        'total number of conjunctions' : classifier_features_df['n_ucconj'].std() + classifier_features_df['n_sconj'].std(),
        'total number of unique proper nouns' : classifier_features_df['n_upropn'].std(),
        'average number of entities' : classifier_features_df['a_n_ent_ps'].std()
        }

        
    for (prompt_aspect, prompt_aspect_value), (gold_aspect_mean, gold_aspect_mean_value),  (gold_aspect_std, gold_aspect_std_value) in zip(prompt_linguistic_flag_dict.items(), goldstandard_linguistic_flag_mean_dict.items(), goldstandard_linguistic_flag_std_dict.items()):
        
        # DISTANCE EVALUATOR: Calculate the z score (deviation) of the value of linguistic flag of prompt from the gold standard mean
        prompt_aspect_deviation = abs((prompt_aspect_value - gold_aspect_mean_value) / gold_aspect_std_value)
        print("Prompt aspect deviation:",prompt_aspect_deviation)

        # VERBALIZER: If z score is greater than 3, rewrite prompt for the next round
        if prompt_aspect_deviation > 1:
            tracker += 1

            prompt_aspect_value = round(prompt_aspect_value, 3)
            gold_aspect_mean_value = round(gold_aspect_mean_value, 3)

            transformed_prompt = transformed_prompt + '\n'
            transformed_prompt = transformed_prompt + str(counter) + ". The {gold_aspect_mean} of the story is {prompt_aspect_value} while the target score should be close to {gold_aspect_mean_value}."
            transformed_prompt = transformed_prompt.replace("{gold_aspect_mean}", prompt_aspect)
            transformed_prompt = transformed_prompt.replace("{prompt_aspect_value}", str(prompt_aspect_value))
            transformed_prompt = transformed_prompt.replace("{gold_aspect_mean_value}", str(gold_aspect_mean_value))

            if prompt_aspect_value < gold_aspect_mean_value:
                transformed_prompt = transformed_prompt + " Increase the complexity by aiming for higher " + prompt_aspect + ". "
            elif prompt_aspect_value > gold_aspect_mean_value:
                transformed_prompt = transformed_prompt + " Decrease the complexity by aiming for lower " + prompt_aspect + ". "

            counter += 1

    if iteration_flag == "rewrite":
        transformed_prompt = transformed_prompt.replace("Continue", "Rewrite")
        transformed_prompt = transformed_prompt.replace("prompt", "story")

    transformed_prompt = transformed_prompt + "\n\nOutput the generated story directly."

    print(transformed_prompt)
    print("Linguistic flags:",tracker)

    return transformed_prompt, tracker


"""
SPECGEM Function Call
"""

def specgem_hf(spec, model_url, api_token, max_length, min_length, top_p, prompt_list, level_list, knowledge_base_df, classifier_features_df, icralm_type):

    model, tokenizer, device = load_model_and_tokenizer(
            model_name = model_url,
            access_token = api_token
        )

    # Empty list to contain generations from model
    model_generations = []

    max_generation_iter = 2 # (n+1) rounds of rewrite
    counter = 0

    for prompt, target_level in zip(prompt_list, level_list):

        current_generated_text = "" # temp value
        tracker = 1 # temp value

        if spec == 'cefr':
            spec_dict = {}
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
            spec_dict = {}
            
            if icralm_type == 'standard':
                spec_dict = query_ccs_standard(target_level, knowledge_base_df)
                transformed_prompt = ccs_prompt_transform_standard(prompt, spec, spec_dict, target_level)
                transformed_prompt = prompt_format(transformed_prompt, model_url)

            elif icralm_type == 'exemplar':
                spec_dict = query_ccs_exemplar(target_level, knowledge_base_df)
                transformed_prompt = ccs_prompt_transform_exemplar(prompt, spec, spec_dict, target_level)
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
                input_ids = tokenizer.encode(transformed_prompt, return_tensors='pt').to(device)

                model.eval()
                with torch.no_grad():
                    output_undecoded = model.generate(
                        input_ids,
                        min_new_tokens = min_length,
                        max_new_tokens = max_length,
                        top_p = top_p,
                        do_sample=True
                    )
                prompt_length = len(tokenizer.encode(transformed_prompt))
                current_generated_text = tokenizer.decode(output_undecoded[0][prompt_length:], skip_special_tokens=True)

            # we cap the number of rewrites based on max_generation_iter
            elif generation_iter > 0:
                print("Here at rewrite round.")

                transformed_prompt = remove_prompt_format(current_generated_text, model_url)
                transformed_prompt, tracker = specgem_prompt(transformed_prompt, spec, classifier_features_df, target_level, "rewrite")
                transformed_prompt = prompt_format(transformed_prompt, model_url)
                
                # however, if there are no more changes left from linguistic prompts, stop iterating
                if tracker == 0:
                    break
                
                input_ids = tokenizer.encode(transformed_prompt, return_tensors='pt').to(device)

                model.eval()
                with torch.no_grad():
                    output_undecoded = model.generate(
                        input_ids,
                        min_new_tokens = min_length,
                        max_new_tokens = max_length,
                        top_p = top_p,
                        do_sample=True
                    )
                prompt_length = len(tokenizer.encode(transformed_prompt))
                current_generated_text = tokenizer.decode(output_undecoded[0][prompt_length:], skip_special_tokens=True)
            
        #print(current_generated_text)
        model_generations.append(current_generated_text)
        
        counter += 1
        print(counter)

    return model_generations