'''
Simple web service wrapping a Word2Vec as implemented in Gensim
Example call: curl http://127.0.0.1:5000/word2vec/n_similarity?s1=sushi&s1=shop&ws2=japanese&ws2=restaurant
@TODO: Add more methods
@TODO: Add command line parameter: path to the trained model
@TODO: Add command line parameters: host and port
'''
from __future__ import print_function

import numpy as np
from future import standard_library
from scipy import spatial

standard_library.install_aliases()
from flask import Flask
from flask_restful import Resource, Api, reqparse
import gensim.models.keyedvectors as word2vec

import argparse
import jieba
import re

parser = reqparse.RequestParser()


def filter_words(words, pickedModel):
    if words is None:
        return
    return [word for word in words if word in pickedModel.vocab]


class Similarity(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('w1', type=str, required=True, help="Word 1 cannot be blank!")
        parser.add_argument('w2', type=str, required=True, help="Word 2 cannot be blank!")
        args = parser.parse_args()
        return model.similarity(args['w1'], args['w2']).item()


class IsInVocabulary(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('w', type=str, required=True, help="Word 1 cannot be blank!")
        args = parser.parse_args()
        s = args['w'].replace('\'s', '')
        splitPattern = r'[,.?:\-_|/! ]'
        s = re.split(splitPattern, s)
        filterS = filter_words(s, model)
        res = 0 if len(filterS) == 0 else 1
        return {'length': len(''.join(filterS)), 'res': res}


class SentenceSimilarity(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('s1', type=str, required=True, help="Word set 1 cannot be blank!", action='append')
        parser.add_argument('s2', type=str, required=True, help="Word set 2 cannot be blank!", action='append')
        args = parser.parse_args()
        # 's不好解读
        s1 = args['s1'][0].replace('\'s', '')
        s2 = args['s2'][0].replace('\'s', '')
        # 拆解成词
        splitPattern = r'[,.?:\-_|/! ]'
        s1 = re.split(splitPattern, s1)
        s2 = re.split(splitPattern, s2)
        # 过滤掉词库里没有的
        filterS1 = filter_words(s1, model)
        filterS2 = filter_words(s2, model)
        if len(filterS1) == 0 or len(filterS2) == 0:
            return -2
        return model.n_similarity(filterS1, filterS2).item()

        # parser = reqparse.RequestParser()
        # parser.add_argument('s1', type=str, required=True, help="Sentence 1 cannot be blank!")
        # parser.add_argument('s2', type=str, required=True, help="Sentence 2 cannot be blank!")
        # args = parser.parse_args()
        #
        # def avg_feature_vector(sentence, model, num_features, index2word_set):
        #     words = sentence.split()
        #     feature_vec = np.zeros((num_features,), dtype='float32')
        #     n_words = 0
        #     for word in words:
        #         if word in index2word_set:
        #             n_words += 1
        #             feature_vec = np.add(feature_vec, model[word])
        #     if (n_words > 0):
        #         feature_vec = np.divide(feature_vec, n_words)
        #     return feature_vec
        #
        # s1_afv = avg_feature_vector(args['s1'], model=model, num_features=300, index2word_set=index2word_set)
        # s2_afv = avg_feature_vector(args['s2'], model=model, num_features=300, index2word_set=index2word_set)
        # sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
        #
        # return sim


class ChineseSenSimilarity(Resource):
    def get(self):
        # parser = reqparse.RequestParser()
        # parser.add_argument('s1', type=str, required=True, help="Word set 1 cannot be blank!", action='append')
        # parser.add_argument('s2', type=str, required=True, help="Word set 2 cannot be blank!", action='append')
        # args = parser.parse_args()
        # s1 = jieba.lcut(args['s1'][0])
        # s2 = jieba.lcut(args['s2'][0])
        # filtered1 = filter_words(s1, baike_model)
        # filtered2 = filter_words(s2, baike_model)
        # return baike_model.n_similarity(filtered1, filtered2).item()
        parser = reqparse.RequestParser()
        parser.add_argument('s1', type=str, required=True, help="Sentence 1 cannot be blank!", action='append')
        parser.add_argument('s2', type=str, required=True, help="Sentence 2 cannot be blank!", action='append')
        args = parser.parse_args()

        def avg_feature_vector(sentence, pickedModel, num_features, index2word_set):
            words = jieba.lcut(sentence[0])
            feature_vec = np.zeros((num_features,), dtype='float32')
            n_words = 0
            for word in words:
                if word in index2word_set:
                    n_words += 1
                    feature_vec = np.add(feature_vec, pickedModel[word])
                elif len(word) > 1:
                    singleChar = list(word)
                    for char in singleChar:
                        if char in index2word_set:
                            print('char', char)
                            n_words += 1
                            feature_vec = np.add(feature_vec, pickedModel[char])
            if n_words > 0:
                feature_vec = np.divide(feature_vec, n_words)
            return feature_vec

        s1_afv = avg_feature_vector(args['s1'], pickedModel=baike_model, num_features=64,
                                    index2word_set=baike_index2word_set)
        s2_afv = avg_feature_vector(args['s2'], pickedModel=baike_model, num_features=64,
                                    index2word_set=baike_index2word_set)
        print('s1_afv', s1_afv)
        print('s2_afv', s2_afv)
        sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
        print('sim', sim)


app = Flask(__name__)
api = Api(app)


@app.errorhandler(404)
def pageNotFound(error):
    return "page not found, you should use correct api"


@app.errorhandler(500)
def raiseError(error):
    return error


if __name__ == '__main__':
    global model
    global norm

    # ----------- Parsing Arguments ---------------
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="Path to the trained model")
    p.add_argument("--binary", help="Specifies the loaded model is binary")
    p.add_argument("--host", help="Host name (default: localhost)")
    p.add_argument("--port", help="Port (default: 5000)")
    p.add_argument("--path", help="Path (default: /word2vec)")
    p.add_argument("--norm",
                   help="How to normalize vectors. clobber: Replace loaded vectors with normalized versions. Saves a lot of memory if exact vectors aren't needed. both: Preserve the original vectors (double memory requirement). already: Treat model as already normalized. disable: Disable 'most_similar' queries and do not normalize vectors. (default: both)")
    args = p.parse_args()

    model_path = args.model if args.model else "./GoogleNews-vectors-negative300.bin.gz"
    baike_model_path = "./news_12g_baidubaike_20g_novel_90g_embedding_64.bin"
    binary = True if args.binary else False
    host = args.host if args.host else "localhost"
    path = args.path if args.path else "/word2vec"
    port = int(args.port) if args.port else 5000
    if not args.model:
        print("Usage: word2vec-apy.py --model path/to/the/model [--host host --port 1234]")

    print("Loading model...")
    model = word2vec.KeyedVectors.load_word2vec_format(model_path, binary=binary, limit=40000)
    index2word_set = set(model.index2word)
    baike_model = word2vec.KeyedVectors.load_word2vec_format(baike_model_path, binary=binary, limit=20000)
    baike_index2word_set = set(baike_model.index2word)

    norm = args.norm if args.norm else "both"
    norm = norm.lower()
    if (norm in ["clobber", "replace"]):
        norm = "clobber"
        print("Normalizing (clobber)...")
        model.init_sims(replace=True)
    elif (norm == "already"):
        model.wv.vectors_norm = model.wv.vectors  # prevent recalc of normed vectors (model.syn0norm = model.syn0)
    elif (norm in ["disable", "disabled"]):
        norm = "disable"
    else:
        norm = "both"
        print("Normalizing...")
        model.init_sims()
    if (norm == "both"):
        print("Model loaded.")
    else:
        print("Model loaded. (norm=", norm, ")")

    api.add_resource(Similarity, path + '/similarity')
    api.add_resource(SentenceSimilarity, path + '/sentence_similarity')
    api.add_resource(ChineseSenSimilarity, path + '/chinese_sen_similarity')
    api.add_resource(IsInVocabulary, path + '/in_vocabulary')
    app.run(host=host, port=port)
