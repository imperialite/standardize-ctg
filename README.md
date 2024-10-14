# Standardize: Aligning Language Models with Expert-Defined Standards for Content Generation

This repository is being updated.

This repository contains the code and data used for the paper [Standardize: Aligning Language Models with Expert-Defined Standards for Content Generation](https://arxiv.org/abs/2402.12593) by Joseph Imperial, Gail Forey, and Harish Tayyar Madabushi accepted to **EMNLP 2024 (Main).**

## Depedencies


### Data
The paper makes use of the following existing datasets which can be found in the data folder. Please cite the associated papers when using the preprocessed data in this work.

#### Common European Framework of Reference for Languages (CEFR)

 - European Language Grid (ELG) - `elg_data.csv` contains CEFR-labelled narratives used as prompts for Task 1 in Section 6. Data can be downloaded [here](https://live.european-language-grid.eu/catalogue/corpus/9477). Citation found below.

	> Breuker, M. (2022). CEFR Labelling and Assessment Services. In European Language Grid: A Language Technology Platform for Multilingual Europe (pp. 277-282). Cham: Springer International Publishing.

- Cambridge Exams -  CEFR-labelled exam narratives used as gold-standard reference where linguistic features are extracted for the `Standardize` framework, particularly with through `Standardize-L`. The dataset can be downloaded [here](https://ilexir.co.uk/datasets/index.html). To get the linguistic features, use [LFTK](https://github.com/brucewlee/lftk). Unfortunately, we cannot redistribute any derived copy of the dataset due to license restrictions but extracting the features through LFTK is easy enough. Citation found below.

	> Menglin Xia, Ekaterina Kochmar, and Ted Briscoe. 2016. [Text Readability Assessment for Second Language Learners](https://aclanthology.org/W16-0502). In _Proceedings of the 11th Workshop on Innovative Use of NLP for Building Educational Applications_, pages 12–22, San Diego, CA. Association for Computational Linguistics.

#### Common Core Standards (CCS)

- Corpus of Contemporary American English (COCA) - `coca_data.csv` contains a sample of the large COCA data used for Task 2 in Section 6. These are merely keywords used for prompting the LLM to generate narratives based on the keyword as the main topic. Citation found below.

	> Davies, Mark. "The 385+ million word Corpus of Contemporary American English (1990–2008+): Design, architecture, and linguistic insights." International journal of corpus linguistics 14.2 (2009): 159-190.

- CCS Exemplars - CCS-labelled stories used as gold-standard reference where linguistic features are extracted for the `Standardize` framework, particularly with through `Standardize-L`. To get the data, please contact the authors from the citation below. Unfortunately, we cannot redistribute any derived copy of the dataset due to license restrictions. Same with Cambridge Exams, extracting the features through LFTK is easy. 

	> Michael Flor, Beata Beigman Klebanov, and Kathleen M. Sheehan. 2013. [Lexical Tightness and Text Complexity](https://aclanthology.org/W13-1504). In _Proceedings of the Workshop on Natural Language Processing for Improving Textual Accessibility_, pages 29–38, Atlanta, Georgia. Association for Computational Linguistics.

### Model
The paper makes use of four models listed below. Please make sure to cite the 

**Open -Weight Models**

 - Llama2-Chat (https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
 - OpenChat (https://huggingface.co/openchat/openchat-3.5-0106)
 - LongForm (https://huggingface.co/akoksal/LongForm-OPT-2.7B)

**Closed Models**

- GPT4 - The study makes use of the paid [OpenAI API](https://openai.com/api/pricing/) to access GPT-4. However, GPT-4o is now the current default model. We adivse to use this new version instead in the interest of higher capabilities and cheaper cost. 

## Generating using Knowledge Artifacts

For all Hugginface models (Llama2-chat, Longform, OpenChat), you need to create a Hugginface account first and request access to these models and use your own user access token for the code.

Some parts of the code may mention the word `Specgem`, this is the old name of the framework before we changed to `Standardize`.

#### Config
The code makes use of the following arguments and their definitions:

 - `spec` choose either 'cefr' or 'ccs'
 - `dataset_path` provide path of prompt data as input
 - `file_output_name` provide any desired filename of output in csv
 - `knowledge_base_path` provide specifications from either CEFR or CCS, can be found in their respective files
 - `classifier_features_path` provide linguistic features from gold-standard CEFR or CCS data
 - `model_api_url` provide Hugginface-style link of model (ex. "meta-llama/Llama-2-7b-chat-hf")
 - `max_length` or `min_length` provide target min and max length of generated content, defaults to 300 and 30
 - `top_p` provide nucleus sampling value, defaults to 0.95
 - `auth_token` provide your Huggingface read and write access key
 - `method` see below
 - `icralm_type` see below

#### Hugginface or OpenAI Setup Selection (`method`)

The `method` argument is dependent on the the type of model you want to use whether it comes from Hugginface of OpenAI. It can have the following values:

 - `simple-prompt-hf` use this for the teacher-style method of prompting using HF models
 - `simple-prompt-openai` same as above but using OpenAI models
 - `ic-ralm-hf` use this for Standardize-A and E using HF models
 - `ic-ralm-openai` same as above but using Open AI models
 - `specgem-hf` use this for Standardize-L and ★ (all artifacts) using HF models
 - `specgem-openai` same as above but OpenAU models

#### Knowledge Artifacts Setup Selection (`icralm_type`)

The `icralm_type` identifies what **knowledge artifact** you want to use with respect to the Standardize framework. See Section 5 of the paper.

 - `standard` used for Standardize-A or aspect-based knowledge artifact
 - `exemplar` used for Standardize-E or exemplar-based knowledge artifact
 - `all` combine `standard` and `exemplar` artifacts, use this argument if you want to use Standardize-L or ★

#### Examples

1. Generate using teacher-style prompting, CCS data, and Llama2 model

	    python script.py --dataset_path "CCS/coca_data.csv" --model_api_url "meta-llama/Llama-2-7b-chat-hf" --method "simple-prompt-hf" --file_output_name "coca_llama_7b_simpleprompt.csv" --auth_token AUTH_TOKEN --max_length 300 --knowledge_base_path "CCS/ccs_specs_finegrained.csv" --spec "ccs"

2. Generate using Standardize-A (aspect knowledge artifact), CEFR data, and Llama2 model

		python script.py --dataset_path "cefr/elg_data.csv" --model_api_url "meta-llama/Llama-2-7b-chat-hf" --method "ic-ralm-hf" --file_output_name "elg_llama2_aspect.csv" --auth_token AUTH_TOKEN --max_length 300 --knowledge_base_path "cefr/cefr_specs.csv" --spec "cefr" --classifier_features_path "cefr/cambridge_all_features.csv" --icralm_type "standard"

3. Generate using Standardize-E (aspect knowledge artifact), CEFR data, and Llama2 model

		python script.py --dataset_path "cefr/elg_data.csv" --model_api_url "meta-llama/Llama-2-7b-chat-hf" --method "ic-ralm-hf" --file_output_name "elg_llama2_exemplar.csv" --auth_token AUTH_TOKEN --max_length 300 --knowledge_base_path "cefr/cefr_specs.csv" --spec "cefr" --classifier_features_path "cefr/cambridge_all_features.csv" --icralm_type "exemplar"

4. Generate using Standardize-★ (all knowledge artifacts), CEFR data, and Llama2 model

	    python script.py --dataset_path "cefr/elg_data.csv" --model_api_url "meta-llama/Llama-2-7b-chat-hf" --method "specgem-hf" --file_output_name "elg_llama2_standardize.csv" --auth_token AUTH_TOKEN --max_length 300 --knowledge_base_path "cefr/cefr_specs.csv" --spec "cefr" --classifier_features_path "cefr/cambridge_all_features.csv" --icralm_type "all"

5. Generate using Standardize-★ (all knowledge artifacts), CEFR data, and GPT-4

	    python script.py --dataset_path "CEFR/elg_data.csv" --model_api_url "gpt" --method "specgem-openai" --file_output_name "elg_gpt4_standardize.csv" --auth_token AUTH_TOKEN --max_length 300 --knowledge_base_path "CEFR/cefr_specs_finegrained.csv" --spec "cefr" --classifier_features_path "CEFR/cambridge_all_features.csv" --icralm_type "all"

You can easily switch from using CEFR to CCS data and choose whichever model from HF you want to use. Most of the models are captured by the AutoTokenizer and AutoModelForCausalLM in `model_utils.py`. If not, just add the specific Auto model reader.


## Evaluation

The eval folder contains `eval_script.ipynb` which is a Python notebook that contains both automatic model-based and fluency/diversity evaluation as described in Section 6.3.

#### Model-Based Evaluation (Precise and Adjacent Accuracies)

For model-based evaluation, you need the `cambridge_features.csv` for CEFR and `commoncore_10_features_bin_with_sbert.csv` for CCS. These files contain the extracted linguistics features to train model-based classifiers Random Forest XGBoost for CEFR and CCS respectively as described in Section 6.3. You will also need to provide `elg_data.csv` or `coca_data.csv` which are both present in repo as well as a csv file for `generation_file_name` containing the model generations you want to evaluate. 

For precise accuracy, the script will output a `classification_report` as a result based on model-classifier's prediction which you can get the values.

For adjacent accuracy, the code after the classification report performs this. Note that adjacent accuracy should only be used for CEFR and not CCS as this requires ordinal data.

#### Fluency and Diversity

For evaluating fluency and diversity, `frugalscore` and `distinct-n` is used. The formula for `distinct-n` is already in the script and make sure that the `frugalscore.py`  is in the same folder as the `eval_script`.


## Paper Citation
If you use any resource from this repository, please cite the `Standardize` paper as referenced below:

```
@inproceedings{imperial-etal-2024-standardize,
	title = "Standardize: {A}ligning {L}anguage {M}odels with {E}xpert-{D}efined {S}tandards for {C}ontent {G}eneration",
	author = "Imperial, Joseph Marvin and Forey, Gail and Tayyar Madabushi, Harish",
	editor = "Al-Onaizan, Yaser and Bansal, Mohit and Chen, Yun-Nung",
	booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
	month = nov,
	year = "2024",
	address = "Miami, Florida",
	publisher = "Association for Computational Linguistics"
}
```
