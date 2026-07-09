"""
Chat-style LCB: English instruction + SFT language tag → target-language response.

Prompt format (same as SFT training template):
  ### Instruction:
  <|tgt_lang:xx|> {english_instruction}

  ### Response:

Baseline model doesn't know the tag → low lc_rate (meaningful reference point).
Fine-tuned models should follow the tag; cross-language interference shows as lc_rate
drop when the tag specifies a language other than the one trained on.

Detection: GlotLID (cis-lmu/glotlid).
"""

import argparse
import json
import os
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GLOTLID_REPO = "cis-lmu/glotlid"
GLOTLID_CACHE = "/root/project/hf_cache"
GLOTLID_LABELS = {"yo": "yor_Latn", "so": "som_Latn", "ha": "hau_Latn"}
ENGLISH_LABEL = "eng_Latn"

MAX_NEW_TOKENS = 200
BATCH_SIZE = 4  # shorter prompts, can use larger batch

# 50 diverse English instructions (topics: society, nature, culture, economy, daily life)
ENGLISH_INSTRUCTIONS = [
    "Describe the importance of education for young people in rural communities.",
    "What are the main challenges that farmers face in sub-Saharan Africa today?",
    "Explain how climate change is affecting water availability in your region.",
    "Describe a traditional wedding ceremony and its significance to the community.",
    "What is the role of elders in passing down knowledge to younger generations?",
    "Explain the importance of learning multiple languages for economic opportunities.",
    "Describe what a typical morning looks like for a family in your village.",
    "What are the main causes of food insecurity and how can communities address them?",
    "Describe the importance of clean drinking water for public health.",
    "Explain what good governance means for the development of a nation.",
    "Describe the role of women in the economic development of communities.",
    "What are the benefits of preserving indigenous languages and cultures?",
    "Explain how mobile phones have changed communication in rural areas.",
    "Describe the relationship between humans and the natural environment.",
    "What steps can individuals take to protect their local environment?",
    "Describe the importance of healthcare access in rural areas.",
    "Explain how trade and commerce contribute to community development.",
    "Describe the significance of music and dance in cultural celebrations.",
    "What are the challenges and opportunities of urbanization for young people?",
    "Explain the role of religion in the daily lives of people in your community.",
    "Describe how traditional medicine is used alongside modern healthcare.",
    "What is the importance of road infrastructure for rural development?",
    "Explain why children should be encouraged to read books from an early age.",
    "Describe the impact of drought on agricultural communities.",
    "What are the benefits of community cooperation during difficult times?",
    "Explain the importance of respecting cultural differences between ethnic groups.",
    "Describe the role of the family in raising children with good values.",
    "What opportunities does the digital economy offer to young Africans?",
    "Explain how deforestation affects local communities and wildlife.",
    "Describe the significance of oral storytelling in preserving history.",
    "What are the main barriers to girls' education and how can they be overcome?",
    "Explain the importance of saving money and financial planning for families.",
    "Describe the challenges of finding employment for young graduates.",
    "What role does sport play in bringing communities together?",
    "Explain how the price of food in markets affects household budgets.",
    "Describe the importance of vaccinations for children's health.",
    "What are the advantages and disadvantages of living in a big city?",
    "Explain the relationship between poverty and access to education.",
    "Describe traditional farming techniques and their value today.",
    "What is the importance of learning mathematics and science in school?",
    "Explain how social media has changed how people get news and information.",
    "Describe the challenges facing fishermen who depend on rivers and lakes.",
    "What are the benefits of renewable energy sources like solar power?",
    "Explain the importance of community markets for local economies.",
    "Describe the impact of migration on families left behind in villages.",
    "What lessons can young people learn from their grandparents' generation?",
    "Explain the importance of peace and conflict resolution in communities.",
    "Describe how traditional crafts and artisanship contribute to local income.",
    "What are the responsibilities of citizens toward their local government?",
    "Explain how access to electricity transforms daily life in rural communities.",
]

LANG_NAMES = {"yo": "Yoruba", "so": "Somali", "ha": "Hausa"}
PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "<|tgt_lang:en|> Please respond to the following in {lang_name}: {instruction}\n\n"
    "### Response:\n"
)


def detect_lang(ft_model, text):
    clean = text.replace("\n", " ").strip()
    if not clean:
        return "unknown"
    pred, _ = ft_model.predict(clean)
    return pred[0].replace("__label__", "")


def eval_one_lang(model, tokenizer, lang, ft_model):
    target_label = GLOTLID_LABELS[lang]
    lang_name = LANG_NAMES[lang]
    prompts = [PROMPT_TEMPLATE.format(lang_name=lang_name, instruction=instr)
               for instr in ENGLISH_INSTRUCTIONS]

    correct = en_count = total = 0
    tokenizer.padding_side = "left"

    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=256,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_len = enc["input_ids"].shape[1]
        for j in range(len(batch)):
            gen = tokenizer.decode(
                out[j][input_len:], skip_special_tokens=True
            ).strip()
            if len(gen.split()) < 5:
                continue
            detected = detect_lang(ft_model, gen)
            if detected == target_label:
                correct += 1
            if detected == ENGLISH_LABEL:
                en_count += 1
            total += 1
        print(f"  [{lang}] {min(i + BATCH_SIZE, len(prompts))}/{len(prompts)}", flush=True)

    return {
        "lc_rate": round(correct / total, 4) if total else 0.0,
        "en_leak": round(en_count / total, 4) if total else 0.0,
        "n": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--langs", default="yo,so,ha")
    args = parser.parse_args()

    langs = args.langs.split(",")

    import fasttext
    from huggingface_hub import hf_hub_download
    glotlid_path = hf_hub_download(
        repo_id=GLOTLID_REPO, filename="model.bin", cache_dir=GLOTLID_CACHE
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ft = fasttext.load_model(glotlid_path)

    print(f"Loading model: {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    scores = {}
    for lang in langs:
        print(f"\n=== LCB-chat: {lang} ({len(ENGLISH_INSTRUCTIONS)} instructions) ===",
              flush=True)
        scores[lang] = eval_one_lang(model, tokenizer, lang, ft)
        print(f"  {scores[lang]}", flush=True)

    result = {"model_path": args.model_path, "scores": scores}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
