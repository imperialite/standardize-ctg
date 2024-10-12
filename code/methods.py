import torch
import requests
import time
import re

from model_utils import *

def clean(text):
  text = str(text)
  text = re.sub('\n',' ',text)
  text = re.sub(' +',' ',text)
  return text

def remove_prompt_format(prompt, filename):
    #prompt = clean(prompt)

    # if the specgem method is not used, don't do anything
    if "specgem" in filename:
        return prompt

    if "llama" in filename:
        if "[/INST]" in prompt:
            splitted_prompt = prompt.split("[/INST]")
            prompt = splitted_prompt[-1]

    elif "longform" in filename:
        splitted_prompt = prompt.split("Output the generated story directly.")
        prompt = splitted_prompt[1]

    elif "openchat" in filename:
        if "Output the generated story directly." in prompt:
            splitted_prompt = prompt.split("Output the generated story directly.")
            prompt = splitted_prompt[1]
        
    return prompt

def query_ccs_standard(target_level, knowledge_base_df):

    quali_meaning = knowledge_base_df.loc[knowledge_base_df['level'] == target_level, 'quali_meaning'].values[0]
    quali_struct = knowledge_base_df.loc[knowledge_base_df['level'] == target_level, 'quali_struct'].values[0]
    quanti_lengths = knowledge_base_df.loc[knowledge_base_df['level'] == target_level, 'quanti_length'].values[0]

    # Place in dictionary
    ccs_kb_rules = {
        'quali_meaning': quali_meaning,
        'quali_struct': quali_struct,
        'quanti_length' : quanti_lengths
    }

    return ccs_kb_rules

def query_ccs_exemplar(target_level, knowledge_base_df):

    exemplar1 = knowledge_base_df.loc[knowledge_base_df['level'] == target_level, 'exemplar1'].values[0]
    exemplar2 = knowledge_base_df.loc[knowledge_base_df['level'] == target_level, 'exemplar2'].values[0]
    exemplar3 = knowledge_base_df.loc[knowledge_base_df['level'] == target_level, 'exemplar3'].values[0]

    # Place in dictionary
    ccs_kb_exemplars = {
        'exemplar1': exemplar1,
        'exemplar2': exemplar2,
        'exemplar3' : exemplar3
    }

    return ccs_kb_exemplars

def query_cefr_exemplar(target_level, knowledge_base_df):

    exemplar1 = knowledge_base_df.loc[knowledge_base_df['level'] == target_level, 'exemplar1'].values[0]
    exemplar2 = knowledge_base_df.loc[knowledge_base_df['level'] == target_level, 'exemplar2'].values[0]
    exemplar3 = knowledge_base_df.loc[knowledge_base_df['level'] == target_level, 'exemplar3'].values[0]

    # Place in dictionary
    cefr_kb_exemplars = {
        'exemplar1': exemplar1,
        'exemplar2': exemplar2,
        'exemplar3' : exemplar3
    }

    return cefr_kb_exemplars

def query_cefr_standard(target_level, knowledge_base_df):
    
    # Qualitative specifications from CEFR per aspect, reference using target level given by user
    meaning_aspect_spec = knowledge_base_df.loc[knowledge_base_df['level'] == target_level, 'meaning_purpose'].values[0]
    structure_aspect_spec = knowledge_base_df.loc[knowledge_base_df['level'] == target_level, 'structure'].values[0]
    grammatical_aspect_spec = knowledge_base_df.loc[knowledge_base_df['level'] == target_level, 'grammatical_complexity'].values[0]
    knowledge_aspect_spec = knowledge_base_df.loc[knowledge_base_df['level'] == target_level, 'knowledge_demands'].values[0]

    # Place in dictionary
    cefr_kb_rules = {
        'meaning_purpose': meaning_aspect_spec,
        'structure': structure_aspect_spec,
        'grammatical_complexity' : grammatical_aspect_spec,
        'knowledge_demands': knowledge_aspect_spec
    }

    return cefr_kb_rules


"""
PROMPT FORMATTER FUNCTIONS
"""

def cefr_prompt_transform_exemplar(prompt, spec, cefr_kb_exemplars, target_level):

    transformed_prompt = """Given this prompt: {narrative}
    
    Continue the story and make sure it is readable for {target_level} learners in the {spec} scale. 
    For your guidance, example books in this level of complexity include {exemplar1}, {exemplar2}, and {exemplar3}.
    
    Output the generated story directly.
    """

    # Replace {narrative} token with actual base prompt
    transformed_prompt = transformed_prompt.replace("{narrative}", prompt)
    transformed_prompt = transformed_prompt.replace("{target_level}", target_level)
    transformed_prompt = transformed_prompt.replace("{spec}", "CEFR")

    for key,value in cefr_kb_exemplars.items():
        exemplar_replace = "{" + key + "}"
        transformed_prompt = transformed_prompt.replace(exemplar_replace, value)

    return transformed_prompt

def ccs_prompt_transform_exemplar(prompt, spec, ccs_kb_exemplars, target_level_num):

    target_level = ""
    if target_level_num == 1:
        target_level = "6 to 8"
    else:
        target_level = "9 to 12"

    transformed_prompt = """Given this prompt: {narrative}
    
    Generate a short story based on the theme of the topic word and make sure it is readable for grades {target_level} learners in the {spec} scale.
    For your guidance, example books in this level of complexity include {exemplar1}, {exemplar2}, and {exemplar3}.
    
    Output the generated story directly.
    """

    # Replace {narrative} token with actual base prompt
    transformed_prompt = transformed_prompt.replace("{narrative}", prompt)
    transformed_prompt = transformed_prompt.replace("{target_level}", target_level)
    transformed_prompt = transformed_prompt.replace("{spec}", "Common Core Standards")

    for key,value in ccs_kb_exemplars.items():
        exemplar_replace = "{" + key + "}"
        transformed_prompt = transformed_prompt.replace(exemplar_replace, value)

    print(transformed_prompt)
    return transformed_prompt

def ccs_prompt_transform_standard(prompt, spec, ccs_kb_target_genre_rules, target_level_num):

    target_level = ""
    if target_level_num == 1:
        target_level = "6 to 8"
    else:
        target_level = "9 to 12"

    transformed_prompt = """Given this topic word: {narrative}
    
    Generate a short story based on the theme of the topic word and make sure it is readable for grades {target_level} learners in the {spec} scale and observes the following specifications:
    1. Meaning or purpose: {quali_meaning}
    2. Structure: {quali_struct}
    3. Lengths: {quanti_length}

    Output the generated story directly.
    """

    # Replace {narrative} token with actual base prompt
    transformed_prompt = transformed_prompt.replace("{narrative}", prompt)
    transformed_prompt = transformed_prompt.replace("{target_level}", target_level)
    transformed_prompt = transformed_prompt.replace("{spec}", "Common Core Standards")

    for key,value in ccs_kb_target_genre_rules.items():
        aspect_replace = "{" + key + "}"
        transformed_prompt = transformed_prompt.replace(aspect_replace, value)

    return transformed_prompt

def ccs_prompt_transform_all(prompt, spec, ccs_kb_exemplars, ccs_kb_target_genre_rules, target_level_num):

    target_level = ""
    if target_level_num == 1:
        target_level = "6 to 8"
    else:
        target_level = "9 to 12"

    transformed_prompt = """Given this topic word: {narrative}
    
    Generate a short story based on the theme of the topic word and make sure it is readable for grades {target_level} learners in the {spec} scale and observes the following specifications:
    1. Meaning or purpose: {quali_meaning}
    2. Structure: {quali_struct}
    3. Lengths: {quanti_length}

    As additional reference, example books in this level of complexity include {exemplar1}, {exemplar2}, and {exemplar3}.

    Output the generated story directly.
    """

    # Replace {narrative} token with actual base prompt
    transformed_prompt = transformed_prompt.replace("{narrative}", prompt)
    transformed_prompt = transformed_prompt.replace("{target_level}", target_level)
    transformed_prompt = transformed_prompt.replace("{spec}", "Common Core Standards")

    for key,value in ccs_kb_target_genre_rules.items():
        aspect_replace = "{" + key + "}"
        transformed_prompt = transformed_prompt.replace(aspect_replace, value)

    for key,value in ccs_kb_exemplars.items():
        exemplar_replace = "{" + key + "}"
        transformed_prompt = transformed_prompt.replace(exemplar_replace, value)

    return transformed_prompt

def cefr_prompt_transform_standard(prompt, spec, cefr_kb_target_level_rules, target_level):

    transformed_prompt = """Given this prompt: {narrative}
    
    Continue the story and make sure it is readable for {target_level} learners in the {spec} scale and observes the following specifications:
    1. Meaning or purpose: {meaning_purpose}
    2. Structure: {structure}
    3. Grammatical complexity: {grammatical_complexity}
    4. Assumed level of knowledge from readers: {knowledge_demands}

    Output the generated story directly.
    """

    # Replace {narrative} token with actual base prompt
    transformed_prompt = transformed_prompt.replace("{narrative}", prompt)
    transformed_prompt = transformed_prompt.replace("{target_level}", target_level)

    transformed_prompt = transformed_prompt.replace("{spec}", "CEFR")

    for key,value in cefr_kb_target_level_rules.items():
        aspect_replace = "{" + key + "}"
        transformed_prompt = transformed_prompt.replace(aspect_replace,value)
        
    return transformed_prompt

def cefr_prompt_transform_all(prompt, spec, cefr_kb_exemplars, cefr_kb_target_level_rules, target_level):

    transformed_prompt = """Given this prompt: {narrative}
    
    Continue the story and make sure it is readable for {target_level} learners in the {spec} scale and observes the following specifications:
    1. Meaning or purpose: {meaning_purpose}
    2. Structure: {structure}
    3. Grammatical complexity: {grammatical_complexity}
    4. Assumed level of knowledge from readers: {knowledge_demands}

    As additional reference, example books in this level of complexity include {exemplar1}, {exemplar2}, and {exemplar3}.

    Output the generated story directly.
    """

    # Replace {narrative} token with actual base prompt
    transformed_prompt = transformed_prompt.replace("{narrative}", prompt)
    transformed_prompt = transformed_prompt.replace("{target_level}", target_level)
    transformed_prompt = transformed_prompt.replace("{spec}", "CEFR")

    for key,value in cefr_kb_target_level_rules.items():
        aspect_replace = "{" + key + "}"
        transformed_prompt = transformed_prompt.replace(aspect_replace,value)

    for key,value in cefr_kb_exemplars.items():
        exemplar_replace = "{" + key + "}"
        transformed_prompt = transformed_prompt.replace(exemplar_replace, value)
        
    return transformed_prompt

def simple_prompt_transform(prompt, spec, target_level):

    if spec == 'ccs':
        if target_level == 1:
            target_level= "6 to 8"
        else:
            target_level= "9 to 12"

        transformed_prompt = """Given this topic word: {narrative}
        
        Generate a short story based on the theme of the topic word and make sure it is readable for grades {target_level} learners in the {spec}. Output the generated story directly. 
        """
    
    elif spec == 'cefr':
        transformed_prompt = """Given this prompt: {narrative}
        
        Continue the story and make sure it is readable for {target_level} learners in the {spec} scale. Output the generated story directly. 
        """

    # Replace {narrative} token with actual base prompt
    transformed_prompt = prompt.replace("{narrative}", prompt)
    transformed_prompt = transformed_prompt.replace("{target_level}", target_level)

    if spec == 'cefr':
        transformed_prompt = transformed_prompt.replace("{spec}", "CEFR")
    elif spec == 'ccs':
        transformed_prompt = transformed_prompt.replace("{spec}", "Common Core Standards")

    return transformed_prompt

def prompt_format(prompt, model_name):
    model_name = model_name.lower()

    if "llama" in model_name:
        prompt = "<s>[INST] <<SYS>> You are a good story teller that can generate stories aligned with specifications based on text complexities provided. You will always output the generated text directly. <</SYS>> " + prompt + " [/INST]"

    elif "longform" in model_name:
        prompt = prompt + " [EOI]"

    elif "openchat" in model_name:
        prompt = prompt + " <|end_of_turn|>"

    elif "mistral" in model_name:
        prompt = "<s>[INST] " + prompt + "[/INST]"

    return prompt

"""
API REQUEST FUNCTIONS FOR GENERATION
"""

def ic_ralm_generate(spec, model_url, api_token, max_length, top_p, prompt_list, level_list, knowledge_base_df):
    
    # Inference call
    api_token = "Bearer " + api_token
    headers = {"Authorization": api_token}
    def query(payload):
        response = requests.post(model_url, headers=headers, json=payload)
        return response.json()
    
    # Empty list to contain generations from model
    model_generations = []

    counter = 0
    for prompt, target_level in zip(prompt_list, level_list):

        if spec == 'cefr':
            spec_dict = query_cefr_standard(target_level, knowledge_base_df)
            transformed_prompt = cefr_prompt_transform_standard(prompt, spec, spec_dict, target_level)
            transformed_prompt = prompt_format(transformed_prompt, model_url)

        elif spec == 'ccs':
            spec_dict = query_ccs_standard(knowledge_base_df)
            transformed_prompt = ccs_prompt_transform_standard(prompt, spec, spec_dict, target_level)
            transformed_prompt = prompt_format(transformed_prompt, model_url)

        #Handle request timed out
        while True:
            try:
                output = query(
                    {
                    "inputs": transformed_prompt,
                    "top_p": top_p,
                    "max_new_tokens": max_length
                    }
                )
                
                output_decoded = output[0]['generated_text']
                transformed_prompt = transformed_prompt + output_decoded
                
                # another round of continuation from the first generation
                output = query(
                    {
                    "inputs": transformed_prompt,
                    "top_p": top_p,
                    "max_new_tokens": max_length
                    }
                )
                
                output_decoded = output[0]['generated_text']
                model_generations.append(output_decoded)

                counter += 1
                print(counter)
                break

            except KeyError as e:
                print("Sleeping due to timeout.")
                time.sleep(900)
    
    return model_generations

def simple_prompt(spec, model_url, api_token, max_length, top_p, prompt_list, level_list):
    
    # Inference call
    api_token = "Bearer " + api_token
    print(api_token)
    headers = {"Authorization": api_token}
    def query(payload):
        response = requests.post(model_url, headers=headers, json=payload)
        return response.json()
    
    # Empty list to contain generations from model
    model_generations = []

    counter = 0
    for prompt, target_level in zip(prompt_list, level_list):

        if spec == 'cefr':
            transformed_prompt = simple_prompt_transform(prompt, spec, target_level)
            transformed_prompt = prompt_format(transformed_prompt, model_url)

        elif spec == 'ccs':
            transformed_prompt = simple_prompt_transform(prompt, spec, target_level)
            transformed_prompt = prompt_format(transformed_prompt, model_url)

         #Handle request timed out
        while True:
            try:
                output = query(
                    {
                    "inputs": transformed_prompt,
                    "top_p": top_p,
                    "max_new_tokens": max_length
                    }
                )
                
                output_decoded = output[0]['generated_text']
                transformed_prompt = transformed_prompt + output_decoded
                
                # another round of continuation from the first generation
                output = query(
                    {
                    "inputs": transformed_prompt,
                    "top_p": top_p,
                    "max_new_tokens": max_length
                    }
                )
                
                output_decoded = output[0]['generated_text']
                model_generations.append(output_decoded)

                counter += 1
                print(counter)
                break

            except KeyError as e:
                print("Sleeping due to timeout.")
                time.sleep(900)
    
    return model_generations

"""
HUGGINGFACE REQUEST FUNCTIONS FOR GENERATION
"""
def ic_ralm_generate_hf(spec, model_url, api_token, max_length, min_length, top_p, prompt_list, level_list, knowledge_base_df, icralm_type):

    model, tokenizer, device = load_model_and_tokenizer(
            model_name = model_url,
            access_token = api_token
        )
    
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

            elif icralm_type == 'all':
                spec_dict_exemplar = query_cefr_exemplar(target_level, knowledge_base_df)
                spec_dict_aspect = query_cefr_standard(target_level, knowledge_base_df)
                transformed_prompt = cefr_prompt_transform_all(prompt, spec, spec_dict_exemplar, spec_dict_aspect, target_level)

        elif spec == 'ccs':
            print("HERE AT CCS SPEC")
            if icralm_type == 'standard':
                spec_dict = query_ccs_standard(target_level, knowledge_base_df)
                transformed_prompt = ccs_prompt_transform_standard(prompt, spec, spec_dict, target_level)

            elif icralm_type == 'exemplar':
                spec_dict = query_ccs_exemplar(target_level, knowledge_base_df)
                transformed_prompt = ccs_prompt_transform_exemplar(prompt, spec, spec_dict, target_level)

            elif icralm_type == 'all':
                spec_dict_aspect = query_ccs_standard(target_level, knowledge_base_df)
                spec_dict_exemplar = query_ccs_exemplar(target_level, knowledge_base_df)
                transformed_prompt = ccs_prompt_transform_all(prompt, spec, spec_dict_aspect, spec_dict_exemplar, target_level)
              
        transformed_prompt = prompt_format(transformed_prompt, model_url)
        print(transformed_prompt)
        input_ids = tokenizer.encode(transformed_prompt, return_tensors='pt').to(device)

        model.eval()
        with torch.no_grad():
            output_undecoded = model.module.generate(
                input_ids,
                min_new_tokens = min_length,
                max_new_tokens = max_length,
                top_p = top_p,
                do_sample=True
            )
        
        output_decoded = tokenizer.decode(output_undecoded[0], skip_special_tokens=True)
        model_generations.append(output_decoded)
        
        counter += 1
        print(counter)
    
    return model_generations

def simple_prompt_generate_hf(spec, model_url, api_token, max_length, min_length, top_p, prompt_list, level_list, knowledge_base_df):

    model, tokenizer, device = load_model_and_tokenizer(
            model_name = model_url,
            access_token = api_token
        )
    
    # Empty list to contain generations from model
    model_generations = []

    counter = 0
    for prompt, target_level in zip(prompt_list, level_list):

        if spec == 'cefr':
            transformed_prompt = simple_prompt_transform(prompt, spec, target_level)
            transformed_prompt = prompt_format(transformed_prompt, model_url)

        elif spec == 'ccs':
            transformed_prompt = simple_prompt_transform(prompt, spec, target_level)
            transformed_prompt = prompt_format(transformed_prompt, model_url)

        input_ids = tokenizer.encode(transformed_prompt, return_tensors='pt').to(device)

        model.eval()
        with torch.no_grad():
            output_undecoded = model.module.generate(
                input_ids,
                min_new_tokens = min_length,
                max_new_tokens = max_length,
                top_p = top_p,
                do_sample = True
            )
        
        output_decoded = tokenizer.decode(output_undecoded[0], skip_special_tokens=True)
        model_generations.append(output_decoded)
        
        counter += 1
        print(counter)
    
    return model_generations


