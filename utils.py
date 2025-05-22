# 必要なライブラリを持ってくる
import pickle
import pandas as pd
import json
import os
import sys
import torch
from glob import glob
from tqdm import tqdm
from typing import Any
from collections import Counter
from datasets import Dataset
from spacy_alignments.tokenizations import get_alignments
from sklearn.model_selection import train_test_split

#モデル用
from torch.utils.data import DataLoader
from seqeval.metrics.sequence_labeling import get_entities

from torchcrf import CRF
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import set_seed
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers import BertForTokenClassification, PretrainedConfig
from transformers.modeling_outputs import TokenClassifierOutput

from typing import Any, List, Dict, Optional, Tuple, Union

# 簡単なデータセット件数確認する関数
def count_label_occurrences(dataset: list):
    type_ids = [entity['type_id'] for item in dataset for entity in item['entities'] if not pd.isna(entity['type_id'])]
    # 番号順ではなく，多い順に並び替え
    type_id_counts = dict(Counter(type_ids).most_common())

    # 結果をDataFrameに変換
    df = pd.DataFrame(list(type_id_counts.items()), columns=['ラベル名', 'count'])
    
    # 合計した行を追加
    total_count = df['count'].sum()
    df.loc[len(df)] = ['合計', total_count]

    return df


# ラベルとIDを紐付けるdictを作成する
# 抽出タグを変える場合はこちらのid_tagsを変更する
def create_label2id(dataset):
    """ラベルとIDを紐付けるdictを作成"""
    # type_idのtag名への変換
    id_tags = {
        1: 'd_',
        2: 'd_positive',
        3: 'd_suspicious',
        4: 'd_negative',
        5: 'd_general',
        25: 'm-key_executed',
        26: 'm-key_negated',
        27: 'm-key_scheduled',
        28: 'm-key_other'
        }
    # "O"のIDには0を割り当てる
    label2id = {"O": 0}
    # 固有表現タイプのsetを獲得して並び替える
    entity_types = set()
    for record in dataset:
        for entity in record['entities']:
            # NaN値のチェックと数値であれば変換
            type_id = entity['type_id']
            if not pd.isna(type_id) and isinstance(type_id, float):
                type_name = id_tags.get(int(type_id), "Unknown")
                entity_types.add(type_name)
            elif isinstance(type_id, str):
                entity_types.add(type_id)

    entity_types = sorted(entity_types)
    for i, entity_type in enumerate(entity_types):
        # "B-"のIDには奇数番号を割り当てる
        label2id[f"B-{entity_type}"] = i * 2 + 1
        # "I-"のIDには偶数番号を割り当てる
        label2id[f"I-{entity_type}"] = i * 2 + 2
    return label2id


# データの前処理関数を定義
# 0605: max_length の記述を追記，エラー出るようなら戻す
def preprocess_data(
    data: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    label2id: Dict[str, int],
) -> BatchEncoding:
    """データの前処理"""
    # テキストのトークナイゼーションを行う
    inputs = tokenizer(
        data["text"],
        return_tensors="pt",
        max_length = 512,
        truncation=True,
        return_special_tokens_mask=True,
    )
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}

    # 文字のlistとトークンのlistのアライメントをとる
    characters = list(data["text"])
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"])
    char_to_token_indices, _ = get_alignments(characters, tokens)

    # "O"のIDのlistを作成する
    labels = torch.zeros_like(inputs["input_ids"])
    for entity in data["entities"]: # 各固有表現を処理する
        start_token_indices = char_to_token_indices[entity["span"][0]]
        end_token_indices = char_to_token_indices[
            entity["span"][1] - 1
        ]
        # 文字に対応するトークンが存在しなければスキップする
        if (
            len(start_token_indices) == 0
            or len(end_token_indices) == 0
        ):
            continue
        start, end = start_token_indices[0], end_token_indices[0]
        entity_type = entity["type_id"]
        # 固有表現の開始トークンの位置に"B-"のIDを設定する
        labels[start] = label2id[f"B-{entity_type}"]
        # 固有表現の開始トークン以外の位置に"I-"のIDを設定する
        if start != end:
            labels[start + 1 : end + 1] = label2id[f"I-{entity_type}"]
    # 特殊トークンの位置のIDは-100とする
    labels[torch.where(inputs["special_tokens_mask"])] = -100
    inputs["labels"] = labels
    return inputs


def create_character_labels(
    text: str, entities: List[Dict[str, Union[List[int], str]]]
) -> List[str]:
    """文字ベースでラベルのlistを作成"""
    # "O"のラベルで初期化したラベルのlistを作成する
    labels = ["O"] * len(text)
    for entity in entities: # 各固有表現を処理する
        entity_span, entity_type = entity["span"], entity["type_id"]
        # 固有表現の開始文字の位置に"B-"のラベルを設定する
        labels[entity_span[0]] = f"B-{entity_type}"
        # 固有表現の開始文字以外の位置に"I-"のラベルを設定する
        for i in range(entity_span[0] + 1, entity_span[1]):
            labels[i] = f"I-{entity_type}"
    return labels

def convert_results_to_labels(
    results: List[Dict[str, Any]]
) -> Tuple[List[List[str]], List[List[str]]]:
    """正解データと予測データのラベルのlistを作成"""
    true_labels, pred_labels = [], []
    for result in results: # 各事例を処理する
        # 文字ベースでラベルのリストを作成してlistに加える
        true_labels.append(
            create_character_labels(result["text"], result["entities"])
        )
        pred_labels.append(
            create_character_labels(result["text"], result["pred_entities"])
        )
    return true_labels, pred_labels

# 正解データと予測データをwikipedia corpusなどのdataset形式にして出力
def extract_entities(
    predictions: List[Dict[str, Any]],
    dataset: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    id2label: Dict[int, str],
) -> List[Dict[str, Any]]:
    """固有表現を抽出"""
    results = []
    for prediction, data in zip(predictions, dataset):
        # 文字のlistを取得する
        characters = list(data["text"])

        # 特殊トークンを除いたトークンのlistと予測ラベルのlistを取得する
        tokens, pred_labels = [], []
        all_tokens = tokenizer.convert_ids_to_tokens(
            prediction["input_ids"]
        )
        for token, label_id in zip(
            all_tokens, prediction["pred_label_ids"]
        ):
            # 特殊トークン以外をlistに追加する
            if token not in tokenizer.all_special_tokens:
                tokens.append(token)
                pred_labels.append(id2label[label_id])

        # 文字のlistとトークンのlistのアライメントをとる
        _, token_to_char_indices = get_alignments(characters, tokens)

        # 予測ラベルのlistから固有表現タイプと、
        # トークン単位の開始位置と終了位置を取得して、
        # それらを正解データと同じ形式に変換する
        pred_entities = []
        for entity in get_entities(pred_labels):
            entity_type, token_start, token_end = entity
            # 文字単位の開始位置を取得する
            char_start = token_to_char_indices[token_start][0]
            # 文字単位の終了位置を取得する
            char_end = token_to_char_indices[token_end][-1] + 1
            pred_entity = {
                "name": "".join(characters[char_start:char_end]),
                "span": [char_start, char_end],
                "type_id": entity_type,
            }
            pred_entities.append(pred_entity)
        data["pred_entities"] = pred_entities
        results.append(data)
    return results

# 予測データのみをwikipedia corpusなどのdataset形式にして出力
# mednerの出力と合わせる+正解データなくても出力可能にするため
# ここに新しく関数定義



# 上記のextract_entitiesに，調剤日時など既存データを追加したもの
def extract_entities_v2(
    predictions: List[Dict[str, Any]],
    dataset: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    id2label: Dict[int, str],
) -> List[Dict[str, Any]]:
    """固有表現を抽出"""
    results = []
    for prediction, data in zip(predictions, dataset):
        # 文字のlistを取得する
        characters = list(data["text"])

        # 特殊トークンを除いたトークンのlistと予測ラベルのlistを取得する
        tokens, pred_labels = [], []
        all_tokens = tokenizer.convert_ids_to_tokens(
            prediction["input_ids"]
        )
        for token, label_id in zip(
            all_tokens, prediction["pred_label_ids"]
        ):
            # 特殊トークン以外をlistに追加する
            if token not in tokenizer.all_special_tokens:
                tokens.append(token)
                pred_labels.append(id2label[label_id])

        # 文字のlistとトークンのlistのアライメントをとる
        _, token_to_char_indices = get_alignments(characters, tokens)

        # 予測ラベルのlistから固有表現タイプと、
        # トークン単位の開始位置と終了位置を取得して、
        # それらを正解データと同じ形式に変換する
        pred_entities = []
        for entity in get_entities(pred_labels):
            entity_type, token_start, token_end = entity
            # 文字単位の開始位置を取得する
            char_start = token_to_char_indices[token_start][0]
            # 文字単位の終了位置を取得する
            char_end = token_to_char_indices[token_end][-1] + 1
            pred_entity = {
                "name": "".join(characters[char_start:char_end]),
                "span": [char_start, char_end],
                "type_id": entity_type,
            }
            pred_entities.append(pred_entity)
        data["pred_entities"] = pred_entities
        results.append(data)
    return results


# エラー分析
def find_error_results(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """エラー事例を発見"""
    error_results = []
    for idx, result in enumerate(results): # 各事例を処理する
        result["idx"] = idx
        # 正解データと予測データが異なるならばlistに加える
        if result["entities"] != result["pred_entities"]:
            error_results.append(result)
    return error_results

def output_text_with_label(result: Dict[str, Any], entity_column: str) -> str:
    """固有表現ラベル付きテキストを出力"""
    text_with_label = ""
    entity_count = 0
    for i, char in enumerate(result["text"]): # 各文字を処理する
        # 出力に加えていない固有表現の有無を判定する
        if entity_count < len(result[entity_column]):
            entity = result[entity_column][entity_count]
            # 固有表現の先頭の処理を行う
            if i == entity["span"][0]:
                entity_type = entity["type_id"]
                # プレフィックスとサフィックスを分割
                prefix, suffix = entity_type.split('_')
                # 変換ルールを適用
                if prefix == 'd':
                    tag = 'd'
                    attr = 'certainly'
                elif prefix == 'm-key':
                    tag = 'm-key'
                    attr = 'state'
                else:
                    continue # 未知のプレフィックスの場合は無視
                # 新しい文字列を生成してリストに追加
                text_with_label += f'<{tag} {attr}="{suffix}">'
            text_with_label += char
            # 固有表現の末尾の処理を行う
            if i == entity["span"][1] - 1:
                entity_type = entity["type_id"]
                prefix, _ = entity_type.split('_')
                text_with_label += f"</{prefix}>"
                entity_count += 1
        else:
            text_with_label += char
    return text_with_label


def create_transitions(
    label2id: Dict[str, int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """遷移スコアを定義"""
    # "B-"のラベルIDのlist
    b_ids = [v for k, v in label2id.items() if k[0] == "B"]
    # I-のラベルIDのlist
    i_ids = [v for k, v in label2id.items() if k[0] == "I"]
    o_id = label2id["O"]  # OのラベルID

    # 開始遷移スコアを定義する
    # すべてのスコアを-100で初期化する
    start_transitions = torch.full([len(label2id)], -100.0)
    # "B-"のラベルへ遷移可能として0を代入する
    start_transitions[b_ids] = 0
    # "O"のラベルへ遷移可能として0を代入する
    start_transitions[o_id] = 0

    # ラベル間の遷移スコアを定義する
    # すべてのスコアを-100で初期化する
    transitions = torch.full([len(label2id), len(label2id)], -100.0)
    # すべてのラベルから"B-"へ遷移可能として0を代入する
    transitions[:, b_ids] = 0
    # すべてのラベルから"O"へ遷移可能として0を代入する
    transitions[:, o_id] = 0
    # "B-"から同じタイプの"I-"へ遷移可能として0を代入する
    transitions[b_ids, i_ids] = 0
    # "I-"から同じタイプの"I-"へ遷移可能として0を代入する
    transitions[i_ids, i_ids] = 0

    # 終了遷移スコアを定義する
    # すべてのラベルから遷移可能としてすべてのスコアを0とする
    end_transitions = torch.zeros(len(label2id))
    return start_transitions, transitions, end_transitions


def decode_with_viterbi(
    emissions: torch.Tensor,  # ラベルの予測スコア
    mask: torch.Tensor,  # マスク
    start_transitions: torch.Tensor,  # 開始遷移スコア
    transitions: torch.Tensor,  # ラベル間の遷移スコア
    end_transitions: torch.Tensor,  # 終了遷移スコア
) -> torch.Tensor:
    """ビタビアルゴリズムを用いて最適なラベル列を探索"""
    # バッチサイズと系列長を取得する
    batch_size, seq_length = mask.shape
    # 予測スコアとマスクに関して、0次元目と1次元目を入れ替える
    emissions = emissions.transpose(1, 0)
    mask = mask.transpose(1, 0)

    histories = []  # 最適なラベル系列を保存するための履歴のlist
    # 開始遷移スコアと予測スコアを加算して、累積スコアの初期値とする
    score = start_transitions + emissions[0]
    for i in range(1, seq_length):
        # 累積スコアを3次元に変換する
        broadcast_score = score.unsqueeze(2)
        # 現在の予測スコアを3次元に変換する
        broadcast_emission = emissions[i].unsqueeze(1)
        # 累積スコアと遷移スコアと現在の予測スコアを加算して、
        # 現在の累積スコアを取得する
        next_score = (
            broadcast_score + transitions + broadcast_emission
        )
        # 現在の累積スコアの各ラベルの最大値とそのインデックスを取得する
        next_score, indices = next_score.max(dim=1)
        # マスクしない要素の場合、累積スコアを更新する
        score = torch.where(mask[i].unsqueeze(1), next_score, score)
        # スコアの高いインデックスを履歴のlistに追加する
        histories.append(indices)
    # 終了遷移スコアを加算して合計スコアとする
    score += end_transitions

    # 各事例で最適なラベル列を取得する
    best_labels_list = []
    for i in range(batch_size):
        # 合計スコアの中で最大のスコアとなるラベルを取得する
        _, best_last_label = score[i].max(dim=0)
        best_labels = [best_last_label.item()]
        # 最後のラベルの遷移を逆方向に探索し、最適なラベル列を取得する
        for history in reversed(histories):
            best_last_label = history[i][best_labels[-1]]
            best_labels.append(best_last_label.item())
        # 順序を反転する
        best_labels.reverse()
        best_labels_list.append(best_labels)
    return torch.LongTensor(best_labels_list)


# CRFによる予測
def run_prediction_crf(
    dataloader: DataLoader,
    model: PreTrainedModel,
) -> List[Dict[str, Any]]:
    """BERT-CRFモデルを用いてラベルを予測"""
    predictions = []
    for batch in tqdm(dataloader):  # 各ミニバッチを処理する
        inputs = {
            k: v.to(model.device)
            for k, v in batch.items()
            if k != "special_tokens_mask"
        }
        # [CLS]以外の予測スコアを取得する
        logits = model(**inputs).logits.cpu()[:, 1:, :]
        # [CLS]以外の特殊トークンのマスクを取得する
        mask = (batch["special_tokens_mask"] == 0).cpu()[:, 1:]
        # 訓練した遷移スコアを取得する
        start_transitions = model.crf.start_transitions.cpu()
        transitions = model.crf.transitions.cpu()
        end_transitions = model.crf.end_transitions.cpu()
        # ビタビアルゴリズムを用いて最適なIDの系列を探索する
        pred_label_ids = decode_with_viterbi(
            logits,
            mask,
            start_transitions,
            transitions,
            end_transitions,
        )
        # [CLS]のIDを0とする
        cls_pred_label_id = torch.zeros(pred_label_ids.shape[0], 1)
        # [CLS]のIDと探索したIDの系列を連結して予測ラベルとする
        batch["pred_label_ids"] = torch.concat(
            [cls_pred_label_id, pred_label_ids], dim=1
        )
        batch = {k: v.cpu().tolist() for k, v in batch.items()}
        # ミニバッチのデータを事例単位のlistに変換する
        predictions += convert_list_dict_to_dict_list(batch)
    return predictions


# evaluation_update_verの値と一致するか確認
def evaluate_model(entities_list, entities_predicted_list, type_id=None):
    """
    正解と予測を比較し、モデルの固有表現抽出の性能を評価する。
    type_idがNoneのときは、全ての固有表現のタイプに対して評価する。
    type_idが整数を指定すると、その固有表現のタイプのIDに対して評価を行う。
    """
    num_entities = 0 # 固有表現(正解)の個数
    num_predictions = 0 # BERTにより予測された固有表現の個数
    num_correct = 0 # BERTにより予測のうち正解であった固有表現の数

    if type_id =='disease':
        target_type_ids =  ['d_positive','d_suspicious','d_negative','d_general']
        for entities, entities_predicted in zip(entities_list, entities_predicted_list):

            if type_id:
                entities = [ e for e in entities if e['type_id'] in target_type_ids ]
                entities_predicted = [ 
                    e for e in entities_predicted if e['type_id'] in target_type_ids
                ]
                
            get_span_type = lambda e: (e['span'][0], e['span'][1], e['type_id'])
            set_entities = set( get_span_type(e) for e in entities )
            set_entities_predicted = set( get_span_type(e) for e in entities_predicted )

            num_entities += len(entities)
            num_predictions += len(entities_predicted)
            num_correct += len( set_entities & set_entities_predicted )

    else:
        for entities, entities_predicted in zip(entities_list, entities_predicted_list):

            if type_id:
                entities = [ e for e in entities if e['type_id'] == type_id ]
                entities_predicted = [ 
                    e for e in entities_predicted if e['type_id'] == type_id
                ]
                
            get_span_type = lambda e: (e['span'][0], e['span'][1], e['type_id'])
            set_entities = set( get_span_type(e) for e in entities )
            set_entities_predicted = set( get_span_type(e) for e in entities_predicted )

            num_entities += len(entities)
            num_predictions += len(entities_predicted)
            num_correct += len( set_entities & set_entities_predicted )

    # 指標を計算
    try:
        precision = num_correct/num_predictions # 適合率
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = num_correct/num_entities # 再現率
    except ZeroDivisionError:
        recall = 0.0    
    
    try:
        f_value = 2*precision*recall/(precision+recall) # F値
    except ZeroDivisionError:
        f_value = 0.0      

    result = {
        'num_entities': num_entities,
        'num_predictions': num_predictions,
        'num_correct': num_correct,
        'precision': precision,
        'recall': recall,
        'f_value': f_value
    }

    return result


# spanのみでの評価
def evaluate_model_only_span(entities_list, entities_predicted_list, type_id=None):
    """
    正解と予測を比較し、モデルの固有表現抽出の性能を評価する。
    type_idがNoneのときは、全ての固有表現のタイプに対して評価する。
    type_idが整数を指定すると、その固有表現のタイプのIDに対して評価を行う。
    """
    num_entities = 0 # 固有表現(正解)の個数
    num_predictions = 0 # BERTにより予測された固有表現の個数
    num_correct = 0 # BERTにより予測のうち正解であった固有表現の数

    if type_id =="disease":
        target_type_ids =  ['d_positive','d_suspicious','d_negative','d_general']
        for entities, entities_predicted in zip(entities_list, entities_predicted_list):

            if type_id:
                entities = [ e for e in entities if e['type_id'] in target_type_ids ]
                entities_predicted = [ 
                    e for e in entities_predicted if e['type_id'] in target_type_ids
                ]
                
            #get_span_type = lambda e: (e['span'][0], e['span'][1], e['type_id'])
            # タプルを作成する lambda 関数
            get_span_type = lambda e: (e['span'][0], e['span'][1])
            set_entities = set( get_span_type(e) for e in entities )
            set_entities_predicted = set( get_span_type(e) for e in entities_predicted )

            num_entities += len(entities)
            num_predictions += len(entities_predicted)
            num_correct += len( set_entities & set_entities_predicted )

    if type_id =="m-key":
        target_type_ids = [ 'm-key_executed', 'm-key_negated', 'm-key_scheduled', 'm-key_other']
        for entities, entities_predicted in zip(entities_list, entities_predicted_list):

            if type_id:
                entities = [ e for e in entities if e['type_id'] in target_type_ids ]
                entities_predicted = [ 
                    e for e in entities_predicted if e['type_id'] in target_type_ids
                ]
                
            #get_span_type = lambda e: (e['span'][0], e['span'][1], e['type_id'])
            # タプルを作成する lambda 関数
            get_span_type = lambda e: (e['span'][0], e['span'][1])
            set_entities = set( get_span_type(e) for e in entities )
            set_entities_predicted = set( get_span_type(e) for e in entities_predicted )

            num_entities += len(entities)
            num_predictions += len(entities_predicted)
            num_correct += len( set_entities & set_entities_predicted )

    else:
        for entities, entities_predicted in zip(entities_list, entities_predicted_list):

            if type_id:
                entities = [ e for e in entities if e['type_id'] == type_id ]
                entities_predicted = [ 
                    e for e in entities_predicted if e['type_id'] == type_id
                ]
                
            get_span_type = lambda e: (e['span'][0], e['span'][1])
            set_entities = set( get_span_type(e) for e in entities )
            set_entities_predicted = set( get_span_type(e) for e in entities_predicted )

            num_entities += len(entities)
            num_predictions += len(entities_predicted)
            num_correct += len( set_entities & set_entities_predicted )

    # 指標を計算
    try:
        precision = num_correct/num_predictions # 適合率
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = num_correct/num_entities # 再現率
    except ZeroDivisionError:
        recall = 0.0    
    
    try:
        f_value = 2*precision*recall/(precision+recall) # F値
    except ZeroDivisionError:
        f_value = 0.0      

    result = {
        'num_entities': num_entities,
        'num_predictions': num_predictions,
        'num_correct': num_correct,
        'precision': precision,
        'recall': recall,
        'f_value': f_value
    }

    return result

# アノテーションデータ抽出
def extract_data_for_multiple_folders(num_list:list):
    ann_files = []  # .annファイルのリスト
    txt_files = []  # .txtファイル（"all" 以外）のリスト
    txt_all = []    # .txtファイル（"all" のみ）のリスト

    path = r"C:\Users\DI\Documents\渡部哲\newNER_model_test"
    for num in num_list:
        folder_path = os.path.join(path, f"薬歴_店舗{num}_clinical")
        file_list = os.listdir(folder_path)
        ann_files.extend([os.path.join(folder_path, file) for file in file_list if file.endswith(".ann") and "all" not in file])
        txt_files.extend([os.path.join(folder_path, file) for file in file_list if file.endswith(".txt") and "all" not in file])
        txt_all.extend([os.path.join(folder_path, file) for file in file_list if file.endswith(".txt") and "all" in file])

    return ann_files, txt_files, txt_all

# アノテーションデータ加工
def preprocess_ann(ann_files, txt_files):
    entities_list = []
    result_data_list = []

    for h in range(len(ann_files)):
        df = pd.read_table(ann_files[h], sep=r'\s+', engine='python',names=["tag_num","type_id","span_b", "span_f", "name"])
        # 新しい列を格納するリストを作成
        new_column = []
        ae_class = []

        # 行ごとに処理
        # 5/15 集計方法変えました
        # 行ごとに処理
        for i in range(len(df) - 1):
            current_tag_num = df.loc[i, 'tag_num']
            new_df = df.query("type_id=='certainty'or type_id == 'state'").query("span_b==@current_tag_num")
            
            # new_valueが空の場合
            if new_df.empty:
                # '999'という値を追加
                new_column.append('Pending')
            else:
                # span_f列の値をae_class列に追加
                # n行目のtype_idとn+1行目のspan_fを結合して新しい列に追加
                new_value = f"{df.loc[i, 'type_id']} {new_df['span_f'].tolist()[0]}"
                new_column.append(new_value)

        # 最後の行が存在する場合のみ新しい列に追加
        if len(df) > 0:
            new_value = df.iloc[-1]['type_id']
            new_column.append(new_value)
        # 新しい列をデータフレームに追加
        df['new_column'] = new_column
        # display(new_column)
        # 4/18 集計方法変えました
        # 行ごとに処理
        for i in range(len(df) - 1):
            current_tag_num = df.loc[i, 'tag_num']
            new_value = df.query("span_b==@current_tag_num").query("type_id=='AnnotatorNotes'")
            
            # new_valueが空の場合
            if new_value.empty:
                # '999'という値を追加
                ae_class.append("999")
            else:
                # span_f列の値をae_class列に追加
                ae_class.extend(new_value['span_f'].tolist())

        # 最後の行が存在する場合のみ新しい列に追加
        if len(df) > 0:
            ae_class.append("999")
            #ae_class.append("999")

        # 新しい列をデータフレームに追加
        df['ae_class'] = ae_class
        # display(df)
        # type列を数値に変換するための辞書
        type_id_dict = {
            'Disease': 1,
            'Disease positive': 2,
            'Disease suspicious': 3,
            'Disease negative': 4,
            'Disease general': 5,
            'MedicineKey executed': 25,
            'MedicineKey negated': 26,
            'MedicineKey scheduled': 27,
            'MedicineKey other': 28,
            'Pending': 999
        }


        # 不要な行を削除
        df = df[~df['tag_num'].str.contains('A')]
        df = df[~df['tag_num'].str.contains('#')]

        df['type_id_num'] = df['new_column'].map(type_id_dict)

        df['span_b'] = df['span_b'].astype(int)
        df['span_f'] = df['span_f'].astype(int)
        
        entities = []
        for _, entity_row in df.iterrows():
            entities.append({
                "name": entity_row["name"],
                "span": [entity_row["span_b"], entity_row["span_f"]],
                "type_id": entity_row["type_id_num"],
                "ae_class": entity_row["ae_class"]
            })

        #display(entities)

        with open(txt_files[h], 'r', encoding='utf-8') as file:
            content = file.read()
            content = content.replace("\n", "")
        
        # パスからファイル名を取得
        filename_with_extension = os.path.basename(txt_files[h])

        # 拡張子を取り除く
        filename_without_extension = os.path.splitext(filename_with_extension)[0]

        result_data = {
                    "curid": filename_without_extension,
                    "text": content,
                    "entities": entities
                }
        

        entities_list.append(entities)
        result_data_list.append(result_data)

    return entities_list, result_data_list

def unknown_type_confirm(dataset):
    unknown_type_id_records = [
        item for item in dataset
        if any(entity['type_id'] == 'Unknown' for entity in item['entities'])
    ]
    unknown_type_id_records_list = []

    # これらのレコードの数といくつかの例を表示
    print(f"'Unknown' type_idを持つレコードの数: {len(unknown_type_id_records)}")
    print("'Unknown' type_idを持つレコードの例（最初の3件）:")
    for record in unknown_type_id_records:
        unknown_type_id_records_list.append(record["curid"])
    return unknown_type_id_records_list



# NoneまたはNaNのtype_idを持つエンティティを含むレコードを検索
def invalid_confirm(dataset):
    invalid_records = [item for item in dataset if any(pd.isna(entity['type_id']) for entity in item['entities'])]
    invalid_records_list = []

    # 不正なレコードの数といくつかの例を表示
    print(f"不正なレコードの数: {len(invalid_records)}")
    print("不正なレコード:")
    for record in invalid_records:
        print(record)
        invalid_records_list.append(record["curid"])
    return invalid_records_list


# あ
def run_prediction(
    dataloader: DataLoader, model: PreTrainedModel
) -> List[Dict[str, Any]]:
    """予測スコアに基づき固有表現ラベルを予測"""
    predictions = []
    for batch in tqdm(dataloader): # 各ミニバッチを処理する
        inputs = {
            k: v.to(model.device)
            for k, v in batch.items()
            if k != "special_tokens_mask"
        }
        # 予測スコアを取得する
        logits = model(**inputs).logits
        # 最もスコアの高いIDを取得する
        batch["pred_label_ids"] = logits.argmax(-1)
        batch = {k: v.cpu().tolist() for k, v in batch.items()}
        # ミニバッチのデータを事例単位のlistに変換する
        predictions += convert_list_dict_to_dict_list(batch)
    return predictions


def convert_list_dict_to_dict_list(
    list_dict: Dict[str, list]
) -> List[Dict[str, list]]:
    """ミニバッチのデータを事例単位のlistに変換"""
    dict_list = []
    # dictのキーのlistを作成する
    keys = list(list_dict.keys())
    for idx in range(len(list_dict[keys[0]])): # 各事例で処理する
        # dictの各キーからデータを取り出してlistに追加する
        dict_list.append({key: list_dict[key][idx] for key in keys})
    return dict_list


def compute_scores(
    true_labels: List[List[str]], pred_labels: List[List[str]], average: str
) -> Dict[str, float]:
    """適合率、再現率、F値を算出"""
    scores = {
        "precision": precision_score(true_labels, pred_labels, average=average),
        "recall": recall_score(true_labels, pred_labels, average=average),
        "f1-score": f1_score(true_labels, pred_labels, average=average),
    }
    return scores
    