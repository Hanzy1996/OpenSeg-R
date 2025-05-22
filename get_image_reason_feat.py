import os
import argparse
import json
from glob import glob
from tqdm import tqdm
import shutil
import open_clip
# import clip
import imagenet_templates
import torch
from einops import rearrange
import numpy as np

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

@torch.no_grad()
def get_embedding(sentence, num_per_class, tokenizer, clip_model, templates):
    if tokenizer is not None:
        sentence = tokenizer(sentence).cuda()
    else: 
        sentence = clip.tokenize(sentence).cuda()
    class_embeddings = clip_model.encode_text(sentence).float()
    if class_embeddings.shape[0] > (num_per_class*len(templates)):
        class_embeddings = rearrange(class_embeddings, '(c k n) d -> c k n d', c=len(templates), n=(num_per_class)).mean(dim=1)
    else:
        class_embeddings = rearrange(class_embeddings, '(c n) d -> c n d', c=len(templates))
    return class_embeddings


def process_cot_text(classname, class_des, templates, num_des=3):
    class_des = class_des['reason3']
    des_sentence = [template.format(classname+' that has ' + i_des) for template in templates for i_des in class_des[:num_des]]
    if len(class_des) < num_des:
        raise ValueError(f"num_des should be less than the length of class_des, which is {len(class_des)}")
    # des_sentence = [template.format(i_des+' of class ' + classname) for template in templates for i_des in class_des[:self.num_des]]
    class_name_sentence = [template.format(classname) for template in templates ]
    return des_sentence, num_des
    



def prepare_class_names_from_metadata_reason(class_name_list, vis_reason, prompt_templates):
    def split_labels(x):
        res = []
        for x_ in x:
            x_ = x_.replace(', ', ',')
            x_ = x_.split(',') # there can be multiple synonyms for single class
            res.append(x_)
        return res
    # get text classifier

    class_names = split_labels(class_name_list) # it includes both thing and stuff
    
    all_class_names = class_names
    validate_class_name = list(vis_reason.keys())


    def matched_class_indices(class_a, validate_class):
        matched_indices = None

        for idx, entry in enumerate(validate_class):
            parts = [item.strip() for item in entry.split(',')]
            if any(cls in parts for cls in class_a):
                return validate_class[idx]
        
        return matched_indices

    def fill_all_templates_ensemble(x_='', reasons=None):
        res = []
        for x in x_:
            sentence, num_perclass_reason = process_cot_text(x, reasons, prompt_templates)
            # import pdb; pdb.set_trace()
            res = res + sentence
        num_perclass_reason = num_perclass_reason
        return res, (len(res) // len(prompt_templates))
    num_templates = []
    templated_class_names = []
    validate_index = []
    for i, x in enumerate(class_names):
        # import pdb; pdb.set_trace()
        matched_validate = matched_class_indices(x, validate_class_name)
        # if not has_common_element(x, validate_class_name):
        if not matched_validate is None:
            # import pdb; pdb.set_trace()
            templated_classes, templated_classes_num = fill_all_templates_ensemble(x, vis_reason[matched_validate])
            templated_class_names += templated_classes
            num_templates.append(templated_classes_num) # how many templates for current classes
            validate_index.append(i)
        else:
            continue
    class_names = templated_class_names
    return num_templates, class_names, validate_index


def encode_text(text, clip_model, normalize: bool = False):
    cast_dtype = clip_model.transformer.get_cast_dtype()

    x = clip_model.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

    x = x + clip_model.positional_embedding.to(cast_dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x, attn_mask=clip_model.attn_mask)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection
    return F.normalize(x, dim=-1) if normalize else x
        
@torch.no_grad()
def get_text_classifier_image(clip_model, tokenizer, class_name, json_reason, prompt_templates):

        # all_text_classifier = []
        # for json_reason_file in json_reason_list:

    with open(json_reason, 'r') as f_in:
        vis_reason = json.load(f_in)['reasoning']['map_class_reason']

    test_num_templates, test_class_names, class_index = prepare_class_names_from_metadata_reason(class_name, vis_reason, prompt_templates)
    text_classifier = []
    # this is needed to avoid oom, which may happen when num of class is large
    bs = 8192
    for idx in range(0, len(test_class_names), bs):
        if tokenizer is not None:
            sentence = tokenizer(test_class_names[idx:idx+bs]).cuda()
        else: 
            sentence = clip.tokenize(test_class_names[idx:idx+bs]).cuda()


        class_embeddings = encode_text(sentence,clip_model, normalize=False).float()
        text_classifier.append(class_embeddings.detach().cpu())

    text_classifier = torch.cat(text_classifier, dim=0)

    # average across templates and normalization.
    text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
    text_classifier = text_classifier.reshape((text_classifier.shape[0]//len(prompt_templates)), len(prompt_templates), text_classifier.shape[-1]).mean(1)
    text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
    # all_text_classifier.append(text_classifier)
    return text_classifier, test_num_templates, class_index


def extract_category_names(file_path):
    category_list = []
    with open(file_path, 'r') as f:
        for line in f:
            # 去除前面的 index 和冒号，只保留类别名字部分
            parts = line.strip().split(':', 1)
            if len(parts) == 2:
                class_names = parts[1].strip()
                category_list.append(class_names)
    return category_list


def main():
    parser = argparse.ArgumentParser(description='Get image reason features')
    parser.add_argument('--categories', type=str, default=None)
    parser.add_argument('--categories_file', type=str, default=None)
    parser.add_argument('--model_name',  type=str, default="Qwen2.5-VL-72B-Instruct-AWQ") 
    parser.add_argument('--json_dir', type=str, default='./reason_text')
    parser.add_argument('--feat_dir', type=str, default='./reason_feat')
    parser.add_argument('--clip_pretrained', type=str, default='Convnext-B')
    parser.add_argument('--prompt_ensemble_type', type=str, default='maft')
    parser.add_argument('--datasets', type=str, default='ade150')
    parser.add_argument('--num_des', type=int, default=3, help='num_des')
    parser.add_argument('--splits', type=int, default=1)
    parser.add_argument('--split_id', type=int, default=0)
    args = parser.parse_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if args.prompt_ensemble_type == "imagenet_select":
        prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
    elif args.prompt_ensemble_type == "imagenet":
        prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
    elif args.prompt_ensemble_type == "single":
        prompt_templates = ['A photo of a {} in the scene',]
    elif args.prompt_ensemble_type == "maft":
        prompt_templates = imagenet_templates.MAFT_PROMPT
    else:
        raise ValueError("prompt_ensemble_type should be in ['imagenet_select', 'imagenet', 'single']")

    json_dir = os.path.join(args.json_dir, args.model_name, args.datasets)
    json_list = sorted(glob(os.path.join(json_dir, "*.json")))

    if args.split_id >= args.splits:
        raise ValueError("split_id should be less than splits")
    else:
        json_list = list(split(json_list, args.splits))[args.split_id]
    if len(json_list) == 0:
        raise ValueError(f"No json files found in the {json_dir}")

    save_dir = os.path.join(args.reason_feat, args.datasets, args.clip_pretrained)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tokenizer = None
    if args.clip_pretrained == "Convnext-L":
        name, pretrain = ('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup')
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            name,
            pretrained=pretrain,
            device=device,)

        tokenizer = open_clip.get_tokenizer(name)
    elif args.clip_pretrained == "Convnext-B":
        name, pretrain = ('convnext_base_w_320', 'laion_aesthetic_s13b_b82k_augreg')
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrain, device=device,)
        tokenizer = open_clip.get_tokenizer(name)
    elif args.clip_pretrained == "ViT-G" or args.clip_pretrained == "ViT-H":
        # for OpenCLIP models
        name, pretrain = ('ViT-H-14', 'laion2b_s32b_b79k') if args.clip_pretrained == 'ViT-H' else ('ViT-bigG-14', 'laion2b_s39b_b160k')
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            name, 
            pretrained=pretrain, 
            device=device, 
            force_image_size=336,)
    
        tokenizer = open_clip.get_tokenizer(name)
    else:
        # for OpenAI models
        clip_model, clip_preprocess = clip.load(args.clip_pretrained, device=device, jit=False, prompt_depth=prompt_depth, prompt_length=prompt_length)
    clip_model.eval()
    print(f"Loaded {args.clip_pretrained} model!")
    class_name_txt = f'./datasets_maft/{args.datasets}.txt'
    class_names = extract_category_names(class_name_txt)[1:]
    failed_json = []
    for json_file in tqdm(json_list):
        save_file = save_dir + '/' + json_file.split('/')[-1].split('.')[0]+'.json'
        if os.path.exists(save_file):
            # print(f"😁😁File {save_file} existed! Have a rest!😁😁")
            continue
        else:
            text_classifier, test_num_templates, class_index = get_text_classifier_image(clip_model, tokenizer, class_names, json_file, prompt_templates, args.reason_type)
            text_classifier = text_classifier.cpu().numpy()
            with open(save_file, 'w') as f:
                json.dump({'text_classifier': text_classifier.tolist(), 'test_num_templates':test_num_templates, 'class_index':class_index}, f)
            print(f"😁File saved to {save_file}!")
    
    import gc
    torch.cuda.empty_cache()
    gc.collect()
            # features = torch.stack(features, dim=1).cpu().numpy()
    print(f"😁😁File saved to {save_dir}!")
        
if __name__ == "__main__":
    main()
   