from .constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, \
TRAJ_TOKEN_INDEX, POINT_TOKEN_INDEX, EGO_TOKEN_INDEX
from . import conversation as conversation_lib
import transformers
import torch
from typing import Dict, Optional, Sequence, List
import copy


def tokenizer_image_token_qwenvl_25(
    prompt, 
    tokenizer, 
    image_token_index=IMAGE_TOKEN_INDEX, 
    return_tensors=None,
    image_length=3019,
    query_len=600,
    img_len=400,
    img_num=6,
    post_query: bool = False,
):# -> Tensor | list:
    replace_target = '<|vision_start|><|image_pad|><|vision_end|>' * (img_num + 1)
    prompt = prompt.replace("<image>",replace_target)
    
    
    # print(prompt) for debug
    prompt_chunks = tokenizer(prompt).input_ids
    
    # 找到<|image_pad|>的token id
    image_pad_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    
    # 找到所有<|image_pad|>的位置
    pad_positions = []
    for i, token_id in enumerate(prompt_chunks):
        if token_id == image_pad_token_id:
            pad_positions.append(i)
    
    # 对每个<|image_pad|>位置进行扩展
    new_prompt_chunks = []
    current_pos = 0
    
    # first_image_length = 619 # stand for query
    first_image_length = query_len # stand for query
    single_image_length = img_len
    # single_image_length = 1600
    
    for pad_pos in pad_positions:
        # 添加pad位置之前的tokens
        new_prompt_chunks.extend(prompt_chunks[current_pos:pad_pos])
        # 添加 image_length image_pad tokens
        # new_prompt_chunks.extend([image_pad_token_id] * image_length)
        if not post_query:
            new_prompt_chunks.extend([image_pad_token_id] * single_image_length if pad_pos != pad_positions[0] else [image_pad_token_id] * first_image_length)
        else:
            new_prompt_chunks.extend([image_pad_token_id] * single_image_length if pad_pos != pad_positions[-1] else [image_pad_token_id] * first_image_length)
        current_pos = pad_pos + 1
    
    # 添加最后一段
    new_prompt_chunks.extend(prompt_chunks[current_pos:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(new_prompt_chunks, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return new_prompt_chunks

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def tokenizer_image_traj_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, traj_token_index=TRAJ_TOKEN_INDEX, point_token_index=POINT_TOKEN_INDEX, ego_token_index=EGO_TOKEN_INDEX, return_tensors=None):
    chunks = []
    current_text = ""
    
    i = 0
    while i < len(prompt):
        if prompt[i:i+7] == '<image>':
            if current_text:
                chunks.append(('text', current_text))
                current_text = ""
            chunks.append(('image', None))
            i += 7
        elif prompt[i:i+6] == '<traj>':
            if current_text:
                chunks.append(('text', current_text))
                current_text = ""
            chunks.append(('traj', None))
            i += 6
        elif prompt[i:i+7] == '<point>':
            if current_text:
                chunks.append(('text', current_text))
                current_text = ""
            chunks.append(('point', None))
            i += 7
        elif prompt[i:i+5] == '<ego>':
            if current_text:
                chunks.append(('text', current_text))
                current_text = ""
            chunks.append(('ego', None))
            i += 5
        else:
            current_text += prompt[i]
            i += 1
    if current_text:
        chunks.append(('text', current_text))

    input_ids = []
    offset = 0
    
    # 处理BOS token
    if len(chunks) > 0 and chunks[0][0] == 'text':
        first_chunk_tokens = tokenizer(chunks[0][1]).input_ids
        if len(first_chunk_tokens) > 0 and first_chunk_tokens[0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(first_chunk_tokens[0])
            first_chunk_tokens = first_chunk_tokens[1:]
            input_ids.extend(first_chunk_tokens)
            chunks.pop(0)
    
    # 处理剩余chunks，对 image 和 traj 使用相同的插入逻辑
    for chunk_type, text in chunks:
        if chunk_type == 'image':
            input_ids.append(image_token_index)
        elif chunk_type == 'traj':
            input_ids.append(traj_token_index)
        elif chunk_type == 'point':
            input_ids.append(point_token_index)
        elif chunk_type == 'ego':
            input_ids.append(ego_token_index)
        elif chunk_type == 'text':
            chunk_tokens = tokenizer(text).input_ids
            input_ids.extend(chunk_tokens[offset:])
    
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids
    
def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len

def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )



def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    training_mode: bool =True,
    has_traj: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    # Apply prompt templates
    # import pdb; pdb.set_trace()
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        if training_mode:
            if has_traj:
                input_ids = torch.stack([tokenizer_image_traj_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
            else:
                input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)

        else:
            if has_traj:
                input_ids = [tokenizer_image_traj_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
            else:
                input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]

            return dict(
                input_ids=input_ids,
            )

    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    
    input_ids = input_ids[:, :tokenizer.model_max_length]
    
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": " # conv.sep 是对话轮次之间的分隔符 
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2) # 拆分多轮对话
        cur_len = 1 # 第一个token，通常是 BOS，不参与loss
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds): # 遍历每一轮对话
            if rou == "":
                break

            parts = rou.split(sep) # 拆分 instruction 和 response
            if len(parts) != 2:
                break
            parts[0] += sep # 把sep加回去，算作instruction的一部分

            if has_image:
                if has_traj:
                    round_len = len(tokenizer_image_traj_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_traj_token(parts[0], tokenizer)) - 2
                else:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                # hack remove
                # if len(rounds) != 1:
                #     print(
                #         f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                #         f" (ignored)"
                #     )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)

def preprocess_qwenvl_25(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    training_mode: bool = False,
    has_image: bool = False,
    has_traj: bool = False, # TODO: support has_traj
    query_len: int = 600,
    img_len: int = 400,
    img_num: int = 6,
    post_query: bool = False,
) -> Dict:
    conv = conversation_lib.conv_qwenvl_25.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # import pdb; pdb.set_trace()
    if has_image:
        if training_mode:
            input_ids = torch.stack(
                [tokenizer_image_token_qwenvl_25(prompt, tokenizer, return_tensors='pt', query_len=query_len, img_len=img_len, post_query=post_query, img_num=img_num) for prompt in conversations], dim=0)
        else:
            input_ids = [tokenizer_image_token_qwenvl_25(prompt, tokenizer, return_tensors='pt', query_len=query_len, img_len=img_len, post_query=post_query, img_num=img_num) for prompt in conversations]
            return dict(
                input_ids=input_ids,
            )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids[0].clone()
    
    # 首先将所有token mask掉
    targets[:] = IGNORE_INDEX

    # 然后只恢复assistant的回答部分
    for idx, conversation in enumerate(conversations):
        rounds = conversation.split("<|im_end|>")
        current_position = 0
        
        for round_text in rounds:
            if not round_text:  # 跳过空字符串
                continue
                
            # 找到assistant回答
            if "<|im_start|>assistant\n" in round_text:
                # 获取assistant回答的内容
                response_start = round_text.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
                response = round_text[response_start:] + "<|im_end|>"
                
                # 计算前缀长度
                prefix = round_text[:response_start]
                prefix_tokens = tokenizer_image_token_qwenvl_25(prefix, tokenizer)
                current_position += len(prefix_tokens)
                
                # 计算回答的长度
                response_tokens = tokenizer_image_token_qwenvl_25(response, tokenizer)
                response_length = len(response_tokens)
                
                # 恢复回答部分的标签
                targets[current_position:current_position + response_length] = input_ids[idx, current_position:current_position + response_length]
                
                current_position += response_length
            else:
                # 跳过非assistant回答的部分
                tokens = tokenizer_image_token_qwenvl_25(
                    round_text + "<|im_end|>", tokenizer, query_len=query_len, img_len=img_len, post_query=post_query, img_num=img_num)
                current_position += len(tokens)

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids.reshape(1, -1), 
        labels=targets.reshape(1, -1),
    )


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    training_mode: bool =True,
    has_traj: bool = False,
    use_qwen: bool = False,
    use_qwenvl_25: bool = False,
    query_len: int = 600,
    img_len: int = 400,
    img_num: int = 6,
    post_query: bool = False,
) -> Dict:
    if use_qwenvl_25:
        return preprocess_qwenvl_25(sources, tokenizer, training_mode=training_mode, 
                                    has_image=has_image, has_traj=has_traj,
                                    query_len=query_len, img_len=img_len, post_query=post_query, img_num=img_num
                                    )
    raise NotImplementedError("Please specify the conversation style for preprocessing.")
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image, training_mode=training_mode, has_traj=has_traj)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [min(len(tokenizer_image_token(prompt, tokenizer)), tokenizer.model_max_length) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt')[:tokenizer.model_max_length] for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


