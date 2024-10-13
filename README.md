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

Note: Some parts of the code may mention the word `Specgem`, this is the old name of the framework before we changed to `Standardize`.

## Evaluation


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
