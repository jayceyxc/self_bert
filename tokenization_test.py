# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import tokenization
import tensorflow as tf


class TokenizationTest(tf.test.TestCase):

    def test_full_tokenizer(self):
        vocab_tokens = [
            "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
            "##ing", ","
        ]
        with tempfile.NamedTemporaryFile(delete=False) as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

            vocab_file = vocab_writer.name

        tokenizer = tokenization.FullTokenizer(vocab_file)
        os.unlink(vocab_file)

        tokens = tokenizer.tokenize(u"UNwant\u00E9d,running")
        self.assertAllEqual(tokens, ["un", "##want", "##ed", ",", "runn", "##ing"])

        self.assertAllEqual(
            tokenizer.convert_tokens_to_ids(tokens), [7, 4, 5, 10, 8, 9])

    def test_chinese(self):
        tokenizer = tokenization.BasicTokenizer()

        tokens = tokenizer.tokenize(u'患者于4年前出现活动性心悸、胸痛，多在重体力活动时发作，胸痛位于剑突下和心前区，手掌大小，'
                                    u'呈闷压样疼痛不适，每次持续10分钟左右，休息数分钟可缓解，发作时伴明显心悸、呼吸困难，无咳嗽、'
                                    u'咳痰，无恶心、呕吐，无出汗，头晕、头痛。曾于2011年来我院就诊，诊断为“冠心病 不稳定心绞痛 '
                                    u'房颤 心功能3级”，后正规服用药物，症状仍间断发作。3月来上述症状明显加重，表现为明显不能耐受体力活动，'
                                    u'稍活动即有明显的胸痛发作，长舒气后症状有所缓解，伴四肢乏力，以双下肢为甚，'
                                    u'伴夜间阵发性呼吸困难及端坐呼吸，上述症状间断出现，进行性加重，后出现双下肢水肿，晨轻暮重，'
                                    u'今为进一步明确诊治，特来我院，门诊以“冠心病 心律失常 心功能不全”收入我科')
        for token in tokens:
            print(token)

        self.assertAllEqual(
            tokenizer.tokenize(u"ah\u535A\u63A8zz"),
            [u"ah", u"\u535A", u"\u63A8", u"zz"])

    def test_basic_tokenizer_lower(self):
        tokenizer = tokenization.BasicTokenizer(do_lower_case=True)

        self.assertAllEqual(
            tokenizer.tokenize(u" \tHeLLo!how  \n Are yoU?  "),
            ["hello", "!", "how", "are", "you", "?"])
        self.assertAllEqual(tokenizer.tokenize(u"H\u00E9llo"), ["hello"])

    def test_basic_tokenizer_no_lower(self):
        tokenizer = tokenization.BasicTokenizer(do_lower_case=False)

        self.assertAllEqual(
            tokenizer.tokenize(u" \tHeLLo!how  \n Are yoU?  "),
            ["HeLLo", "!", "how", "Are", "yoU", "?"])

    def test_wordpiece_tokenizer(self):
        vocab_tokens = [
            "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
            "##ing"
        ]

        vocab = {}
        for (i, token) in enumerate(vocab_tokens):
            vocab[token] = i
        tokenizer = tokenization.WordpieceTokenizer(vocab=vocab)

        self.assertAllEqual(tokenizer.tokenize(""), [])

        self.assertAllEqual(
            tokenizer.tokenize("unwanted running"),
            ["un", "##want", "##ed", "runn", "##ing"])

        self.assertAllEqual(
            tokenizer.tokenize("unwantedX running"), ["[UNK]", "runn", "##ing"])

    def test_convert_tokens_to_ids(self):
        vocab_tokens = [
            "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
            "##ing"
        ]

        vocab = {}
        for (i, token) in enumerate(vocab_tokens):
            vocab[token] = i

        self.assertAllEqual(
            tokenization.convert_tokens_to_ids(
                vocab, ["un", "##want", "##ed", "runn", "##ing"]), [7, 4, 5, 8, 9])

    def test_is_whitespace(self):
        self.assertTrue(tokenization._is_whitespace(u" "))
        self.assertTrue(tokenization._is_whitespace(u"\t"))
        self.assertTrue(tokenization._is_whitespace(u"\r"))
        self.assertTrue(tokenization._is_whitespace(u"\n"))
        self.assertTrue(tokenization._is_whitespace(u"\u00A0"))

        self.assertFalse(tokenization._is_whitespace(u"A"))
        self.assertFalse(tokenization._is_whitespace(u"-"))

    def test_is_control(self):
        self.assertTrue(tokenization._is_control(u"\u0005"))

        self.assertFalse(tokenization._is_control(u"A"))
        self.assertFalse(tokenization._is_control(u" "))
        self.assertFalse(tokenization._is_control(u"\t"))
        self.assertFalse(tokenization._is_control(u"\r"))

    def test_is_punctuation(self):
        self.assertTrue(tokenization._is_punctuation(u"-"))
        self.assertTrue(tokenization._is_punctuation(u"$"))
        self.assertTrue(tokenization._is_punctuation(u"`"))
        self.assertTrue(tokenization._is_punctuation(u"."))

        self.assertFalse(tokenization._is_punctuation(u"A"))
        self.assertFalse(tokenization._is_punctuation(u" "))


if __name__ == "__main__":
    tf.test.main()
