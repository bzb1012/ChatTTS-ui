import json
import os
import datetime
import subprocess
import sys
import time

import ChatTTS
import torch
import torchaudio
from uilib import utils

# 全局初始化模型避免重复加载
chat = ChatTTS.Chat()
chat.load(compile=False)


def generate_audio_batch(texts_dict, temperature=0.2, output_dir="/content/ChatTTS-ui/DYLiveAudio/main/", voice_seed=1367, speech_speed=5):
    """
    批量生成带商品ID的音频文件

    :param texts_dict: 嵌套字典 {商品ID: [话术列表]}
    :param temperature: 生成随机性控制，默认0.2
    :param output_dir: 输出目录，默认"F:/DYLiveAudio"
    :param voice_seed: 音色随机种子，默认1367
    :param speech_speed: 语速级别（1-10），默认8
    :return: 生成的文件路径列表
    """
    torch.manual_seed(voice_seed)
    rand_spk = chat.sample_random_speaker()

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    # 遍历每个商品
    for product_id, sentences in texts_dict.items():
        # 处理文本预处理
        processed_text = utils.split_text(sentences)

        # 配置生成参数（与原始代码相同）
        params_refine_text = ChatTTS.Chat.RefineTextParams(
            prompt='[oral_2][laugh_0][break_6]',
            top_P=0.7,
            top_K=20,
            max_new_token=384,
            temperature=temperature
        )

        params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=rand_spk,
            prompt=f"[speed_{speech_speed}]",
            max_new_token=2048,
            temperature=temperature,
        )

        # 生成音频
        wavs = chat.infer(
            processed_text,
            skip_refine_text=True,
            do_text_normalization=False,
            do_homophone_replacement=True,
            params_refine_text=params_refine_text,
            params_infer_code=params_infer_code
        )
        filename_list = []

        # 保存带商品ID的文件
        for idx, wav in enumerate(wavs):
            filename = f"{product_id}_{idx}_{datetime.datetime.now().strftime('%H%M%S')}.wav"
            filename_list.append(filename)
            output_path = os.path.join(output_dir, filename)

            try:
                torchaudio.save(output_path, torch.from_numpy(wav).unsqueeze(0), 24000)
            except:
                torchaudio.save(output_path, torch.from_numpy(wav), 24000)

            generated_files.append(output_path)

        txt_tmp = "\n".join([f"file '{output_dir}/{it}'" for it in filename_list])
        txt_name = f'{time.time()}.txt'
        with open(f'{output_dir}/{txt_name}', 'w', encoding='utf-8') as f:
            f.write(txt_tmp)
        outname = f"{product_id}_{datetime.datetime.now().strftime('%H%M%S')}"+ '_merge.wav'
        merge_filepath = os.path.join(output_dir, 'merge')
        os.makedirs(merge_filepath, exist_ok=True)

        try:
            subprocess.run(["ffmpeg", "-hide_banner", "-ignore_unknown", "-y", "-f", "concat", "-safe", "0", "-i",
                            f'{output_dir}/{txt_name}', "-c:a", "copy", merge_filepath +'/'+ outname],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           encoding="utf-8",
                           check=True,
                           text=True,
                           creationflags=0 if sys.platform != 'win32' else subprocess.CREATE_NO_WINDOW)
        except Exception as e:
            return e

    return generated_files


def generate_audio_single(text, temperature=0.2, output_dir="/content/ChatTTS-ui/DYLiveAudio", voice_seed=1367, speech_speed=8):
    """
    生成单个音频文件

    :param text: 单个文本字符串
    :param temperature: 生成随机性控制，默认0.2
    :param output_dir: 输出目录，默认"F:/DYLiveAudio"
    :param voice_seed: 音色随机种子，默认1367
    :param speech_speed: 语速级别（1-10），默认8
    :return: 生成的音频文件绝对路径
    """
    torch.manual_seed(voice_seed)
    rand_spk = chat.sample_random_speaker()

    processed_text = utils.split_text([text])  # 包装为列表处理

    params_refine_text = ChatTTS.Chat.RefineTextParams(
        prompt='[oral_2][laugh_0][break_6]',
        top_P=0.7,
        top_K=20,
        max_new_token=384,
        temperature=temperature
    )

    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=rand_spk,
        prompt=f"[speed_{speech_speed}]",
        max_new_token=2048,
        temperature=temperature,
    )

    os.makedirs(output_dir, exist_ok=True)
    filename_list = []
    wavs = chat.infer(
        processed_text,
        skip_refine_text=True,
        do_text_normalization=False,
        do_homophone_replacement=True,
        params_refine_text=params_refine_text,
        params_infer_code=params_infer_code
    )

    if len(wavs) == 0:
        raise ValueError("音频生成失败，未产生有效输出")
    for idx,wav in enumerate(wavs):
        filename = f"single_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{idx}.wav"
        output_path = os.path.join(output_dir, filename)
        filename_list.append(filename)
        try:
            torchaudio.save(output_path, torch.from_numpy(wav).unsqueeze(0), 24000)
        except:
            torchaudio.save(output_path, torch.from_numpy(wav), 24000)

    txt_tmp = "\n".join([f"file '{output_dir}/{it}'" for it in filename_list])
    txt_name = f'{time.time()}.txt'
    with open(f'{output_dir}/{txt_name}', 'w', encoding='utf-8') as f:
        f.write(txt_tmp)
    outname = datetime.datetime.now().strftime('%H%M%S_') + "_merge.wav"
    try:
        subprocess.run(["ffmpeg", "-hide_banner", "-ignore_unknown", "-y", "-f", "concat", "-safe", "0", "-i",
                        f'{output_dir}/{txt_name}', "-c:a", "copy", output_dir + '/' + outname],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       encoding="utf-8",
                       check=True,
                       text=True,
                       creationflags=0 if sys.platform != 'win32' else subprocess.CREATE_NO_WINDOW)
    except Exception as e:
        return e

    return os.path.join(output_dir, outname)


if __name__ == "__main__":
    # 批量生成示例
    with open('say.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("批量生成目录:", generate_audio_batch(data))
    #
    # # 单个生成示例
    # single_text = "小黄车1号链接是宝可梦耿鬼鼠标垫，有需要的朋友可以了解一下。这款鼠标垫尺寸达到了八百*三百毫米，超大加厚，非常适合办公和电竞使用。它采用橡胶材质，高密度编织，防水防污，清洁起来很方便。运用高清印刷技术与定制图案，不仅实用，还能彰显个性。在使用场景方面，办公时它能缓解手腕与桌面的压力，让你长时间使用电脑更舒适；电竞游戏中，防滑设计能让鼠标使用更稳定，提升游戏体验。从用户反馈来看，大家都觉得它超级好看，性价比高。该商品享受7天无理由退货、晚发即赔服务，预计18小时内从广东省广州市发货，包邮。"
    # #
    # print("单个生成路径:", generate_audio_single(single_text))