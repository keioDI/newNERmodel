# %%
# 必要なライブラリを持ってくる
import pickle
import pandas as pd
import json
import os
import warnings
import sys
import torch
# 外部データ用の2つのライブラリ
import re
import unicodedata

from typing import Any
from collections import Counter
from transformers import AutoTokenizer
from datasets import Dataset
from spacy_alignments.tokenizations import get_alignments
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from sklearn.model_selection import train_test_split

# 必要なライブラリを持ってくる
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)

from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import set_seed

from tqdm import tqdm
from torch.utils.data import DataLoader
from seqeval.metrics.sequence_labeling import get_entities
from transformers import PreTrainedModel
from glob import glob

from torchcrf import CRF
from transformers import BertForTokenClassification, PretrainedConfig, AutoTokenizer
from transformers.modeling_outputs import TokenClassifierOutput

# 自作のutisから必要な関数を取ってくる
from utils import (
    count_label_occurrences,
    create_label2id,
    preprocess_data,
    create_character_labels,
    convert_results_to_labels,
    extract_entities,
    find_error_results,
    output_text_with_label,
    create_transitions,
    decode_with_viterbi,
    run_prediction_crf,
    evaluate_model,
    evaluate_model_only_span,
    extract_data_for_multiple_folders,
    preprocess_ann,
    unknown_type_confirm,
    invalid_confirm
)

#%%
from typing import Optional

class BertWithCrfForTokenClassification(BertForTokenClassification):
    """BertForTokenClassificationにCRF層を加えたクラス"""

    def __init__(self, config: PretrainedConfig):
        """クラスの初期化"""
        super().__init__(config)
        # CRF層を定義する
        self.crf = CRF(len(config.label2id), batch_first=True)

    def _init_weights(self, module: torch.nn.Module) -> None:
        """定義した遷移スコアでパラメータを初期化"""
        super()._init_weights(module)
        if isinstance(module, CRF):
            st, t, et = create_transitions(self.config.label2id)
            module.start_transitions.data = st
            module.transitions.data = t
            module.end_transitions.data = et

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> TokenClassifierOutput:

        """モデルの前向き計算を定義"""
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        if labels is not None:
            logits = output.logits
            mask = labels != -100
            labels *= mask
            # CRFによる損失を計算する
            output["loss"] = -self.crf(
                logits[:, 1:, :],
                labels[:, 1:],
                mask=mask[:, 1:],
                reduction="mean",
            )
        return output

# %%
if __name__ == '__main__':
    # モデル名定義
    # tokenizerの引用
    model_name = "cl-tohoku/bert-base-japanese-v3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # collate関数にDataCollatorForTokenClassificationを用いる
    data_collator = DataCollatorForTokenClassification(tokenizer)

    #エラー無視 
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # type_idのtag名への変換
    # id_tags定義
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
    
    # text読み込み（ここを変更 .txt前部分を入力）
    txt_name = r"data/text"

    with open(f"{txt_name}.txt", encoding="utf-8") as f:
            articles_raw = f.read()
    
    # unicode正規化
    article_norm = unicodedata.normalize('NFKC', articles_raw)

    sentences_raw = [s for s in re.split(r'\n', articles_raw) if s != '']
    sentences_norm = [s for s in re.split(r'\n', article_norm) if s != '']

    # ラベルとID紐づけ
    with open('label2id.pkl', 'rb') as file:
        label2id = pickle.load(file)
    id2label = {v: k for k, v in label2id.items()}

    # Dataset形式(pandasの拡張パッケージ？)に加工する
    dataset = [{'text': text, 'entities': []} for text in sentences_norm]
    test_dataset_df = Dataset.from_pandas(pd.DataFrame(dataset))

    # 外部データに対してtokenizerを含む前処理を行う
    test_dataset = test_dataset_df.map(
        preprocess_data,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label2id": label2id,
        },
        remove_columns=test_dataset_df.column_names
    )

    # データローダーへの加工
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=data_collator)
    
    # モデル読み込み (自動判定でcpuモードに)
    best_model = BertWithCrfForTokenClassification.from_pretrained('checkpoint-150')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = best_model.to(device)
    # 固有表現ラベルを予測する
    predictions = run_prediction_crf(test_dataloader, best_model)
    # 固有表現を抽出する
    results = extract_entities(
        predictions, test_dataset_df, tokenizer, id2label
    )

    # text_xmlに各項目を入れ，それをtexts_xmlに結合させる
    text_xml = []
    texts_xml = []
    for i in results:
        pred_text = output_text_with_label(i, "pred_entities")
        text_xml.append(pred_text)
    texts_xml.append("\n".join(text_xml))

    # xml形式で保存
    with open(txt_name + '_BERT_CRF.xml', mode='w', encoding='utf-8') as f:
                f.write(texts_xml[0])
                f.close()