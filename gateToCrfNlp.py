# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
import string
from nltk.tokenize.punkt import PunktSentenceTokenizer
import json
import re
from enum import Enum


class Term(object):
    def __init__(self):
        self.text = None
        self.begin = None
        self.end = None
        self.type = None
        self.pos_begin = None # in bytes
        self.pos_end = None # in bytes

class NamedEntity(Term):
    def __init__(self):
        self.eType = None
        self.id = None
        super().__init__()


class NEpart(Term):
    def __init__(self):
        self.nePartType = None
        super().__init__()


class NounPhrase(Term):
    def __init__(self):
        super().__init__()


class Sentiment(Term):
    def __init__(self):
        self.polarity = None
        self.id = None
        self.target = None
        self.isNegation = None
        super().__init__()

class Ner(Term):
    def __init__(self):
        self.semantic = None
        self.id = None
        self.morphPos = None
        self.morphCase = None
        self.chunkId = None
        self.chunkTag = None
        self.eType = None
        self.isUpper = None
        self.sentimentTag = None
        #Sentiment
        self.target = None
        self.isNegation = None
        self.polarity = None
        self.norm_form = None
        self.predictSentiment = None

        super().__init__()

def to_terms(line):
    words = line.split()
    #print(words)
    index = line.index
    offsets = []
    append = offsets.append
    running_offset = 0
    for word in words:
        #print(word)
        word_offset = index(word, running_offset)
        word_len = len(word)
        running_offset = word_offset + word_len
        append((word, word_offset, running_offset))
    return offsets


def to_np_chunks(text, npChunks):
    for c in npChunks:
        if c.type == 'nounPhrase':
            print(c.text)

def read_gazetteer_dicts_to_set(fileList):
    newSet = set()
    for fname in fileList:
        f = open('/home/artem/projects/resources/ru/gazetteer/' + fname, 'r')
        content = f.readlines()
        temp = [line.rstrip('\n').lower() for line in content]
        newSet.update(temp)
    return newSet



def write_gazetteers(nePartsDict):
    for n in nePartsDict:
        f = open('dictAddition/' + n, 'w')
        gazetteersSet = set()
        print(n)
        if n == 'positive':
            gazetteersSet = read_gazetteer_dicts_to_set(['sentiment_positive.lst'])
        elif n == 'negative':
            gazetteersSet = read_gazetteer_dicts_to_set(['sentiment_negative.lst'])
        elif n == 'loc_descr':
            gazetteersSet = read_gazetteer_dicts_to_set(['geoplace_keypost.lst', 'loc_key.lst', 'loc_prekey.lst'])
        elif n == 'name':
            gazetteersSet = read_gazetteer_dicts_to_set(['first_names_female.lst', 'first_names_male.lst'])
        elif n == 'org_descr':
            gazetteersSet = read_gazetteer_dicts_to_set(['org_key.lst'])


        tmpSet = set(nePartsDict[n])
        myset = tmpSet.difference(gazetteersSet)
        print('set(nePartsDict[n])', len(tmpSet), 'myset', len(myset) )
        for w in myset:
            f.write(w + '\n')


def make_gazetteer_list(termLists, nePartsDict):
    cnt = 0
    for tlnum, tl in enumerate(termLists):
        print(tlnum, 'of', len(termLists))
        for t in tl:
            # print(t.text, t.type, t.begin, t.end)
            try:
                if t.type == 'sentiment' or t.type == 'nePart':
                    wordslist = list()
                    words = t.text.split()
                    for w in words:
                        wordslist.append((w.lower(), w.istitle()))
                    newnormlist = mObj.getMorhList(wordslist)
                    norm = str()
                    for i, n in enumerate(newnormlist):
                        if i != 0:
                            norm += ' ' + n[2]
                        else:
                            norm = n[2]

                if t.type == 'sentiment':
                    #print(t.isNegation, t.polarity, t.id, t.target)
                    if t.polarity in nePartsDict:
                        nePartsDict[t.polarity].append(norm)
                    else:
                        nePartsDict[t.polarity] = [norm]
                elif t.type == 'nePart':
                    # print(t.nePartType)
                    if t.nePartType in nePartsDict:
                        nePartsDict[t.nePartType].append(norm)
                    else:
                        nePartsDict[t.nePartType] = [norm]

            except AttributeError:
                cnt = cnt + 1
                print(AttributeError)
                # print (cnt,'attributes missing')


def strip_terms_from_punct(npList):
    myPunctSet = set(string.punctuation)
    myPunctSet.update('»«')
    for t in npList:
        #if not (t.text[0].isalpha() and t.text[len(t.text)-1].isalpha()):
        if any(c.isalpha() for c in t.text):
            if t.text[0] in myPunctSet and len(t.text) > 1:
                finI = -1
                for i, c in enumerate(t.text):
                    if c in myPunctSet:
                        finI = i
                    else:
                        if finI > -1:
                            t.text = t.text[finI+1:]
                            t.begin = t.begin + finI +1
                            #print(t.text, t.mark, t.begin, t.end)
                        break
            if t.text[len(t.text)-1] in myPunctSet:
                finI = -1
                for i, c in enumerate(reversed(t.text)):
                    if c in myPunctSet:
                        finI = i
                    else:
                        if finI > -1:
                            t.text = t.text[:-(finI+1)]
                            t.end = t.end - finI -1
                        break
        #elif len(t.text) > 1 and any(c in myPunctSet for c in t.text):
        #    print(t.text, 'need to split those')


def find_all_punct(terms, allPunctlist):
    myPunctSet = set(string.punctuation)
    myPunctSet.update('»«')
    for t in terms:
        if t[0][0] in myPunctSet and len(t[0]) > 1:
            finI = -1
            for i, c in enumerate(t[0]):
                if c in myPunctSet:
                    finI = i
                else:
                    if finI > -1:
                        for x in range(0, finI+1):
                            newPT = Term()
                            newPT.begin = t[1] + x
                            newPT.end = newPT.begin + 1
                            newPT.text = t[0][x]
                            newPT.mark = 'O'
                            allPunctlist.append(newPT)
                    break
        if t[0][len(t[0])-1] in myPunctSet:
            finI = -1
            for i, c in enumerate(reversed(t[0])):
                if c in myPunctSet:
                    finI = i
                else:
                    if finI > -1:
                        for x in range(0, finI+1):
                            newPT = Term()
                            newPT.begin = t[2] - x - 1
                            newPT.end = newPT.begin + 1
                            newPT.text = t[0][len(t[0]) - 1 - x]
                            newPT.mark = 'O'
                            allPunctlist.append(newPT)
                    break

class CorpusType(Enum):
    NER_1 = 1,
    NER_2 = 2,
    NER_3 = 3,
    NER_2_testset = 4,
    NER_4 = 5,
    NER_ALL = 6,
    NER_ALL_EN = 7,

def parse_document(termList, tFile):

    tree = ET.parse(tFile)
    str1 = ''
    annotlist = list()
    for elem in tree.iter(tag='TextWithNodes'):
        str1 += ET.tostring(elem, encoding='unicode', method='text')
        #for child in elem:
            #str1 += ET.tostring(child, encoding='unicode', method='text')

    for elem in tree.iter(tag='AnnotationSet'):
        annotlist.extend(elem.findall('./Annotation'))

    #print(str1)

    for l in annotlist:
        if l.attrib['Type'] != 'paragraph':
            newTerm = Term()
            newTerm.text = str1[int(l.attrib['StartNode']):int(l.attrib['EndNode'])]
            newTerm.begin = int(l.attrib['StartNode'])
            newTerm.end = int(l.attrib['EndNode'])
            newTerm.pos_begin = len(str1[0:int(l.attrib['StartNode'])].encode('utf-8'))
            newTerm.pos_end = len(str1[0:int(l.attrib['EndNode'])].encode('utf-8'))
            newTerm.type = l.attrib['Type']

            #if (newTerm.type == 'nounPhrase'):
             #   print(newTerm.text, newTerm.begin, newTerm.end)

            if (newTerm.type == 'sentiment'):
                newTerm.isNegation = False

            for feat in l:
                paramName = ""
                for featchild in feat:
                    if(featchild.tag == 'Name'):
                        paramName = featchild.text
                        #if (featchild.tag == 'Value'):
                    elif(paramName == 'type') and (newTerm.type == 'namedEntity'):
                        newTerm.eType = featchild.text
                    elif(paramName == 'type') and (newTerm.type == 'nePart'):
                        newTerm.nePartType = featchild.text
                    elif(paramName == 'type') and (newTerm.type == 'sentiment'):
                        newTerm.polarity = featchild.text
                    elif(paramName == 'id'):
                        newTerm.id = featchild.text
                    elif(paramName == 'targetEntity-link'):
                        newTerm.target = featchild.text
                    elif(paramName == 'isNegation'):
                        newTerm.isNegation = True
            termList.append(newTerm)
    return str1
            # elif (l.attrib['Type'] == 'nePart'):
            #     print(l.attrib['Type'])
            #     print(str1[int(l.attrib['StartNode']):int(l.attrib['EndNode'])])


def find_npChunks(termList, npList):
    for t in termList:
        if t.type == 'nounPhrase':
            newterms = to_terms(t.text)
            lastpos = 0
            for index, nt in enumerate(newterms):
                newT = Term()
                if index == 0:
                    newT.mark = 'B'
                else:
                    newT.mark = 'I'
                newT.begin = t.begin + t.text.find(nt[0], lastpos)
                lastpos += len(nt[0])
                newT.end = newT.begin + len(nt[0])
                newT.text = nt[0]
                npList.append(newT)


def make_addition_list(addList, npList, terms):
    prev = Term()
    prev.begin = -2
    prev.end = -2
    prevI = 0
    for npterm in npList:
        #print('npTerm ', npterm.text, npterm.begin, npterm.end)
        #if npterm.begin - prev.end > 1:
        for i in range(prevI, len(terms) - 1):
            #print('t ', terms[i][0])
            if terms[i][1] >= npterm.end:
                prevI = i
                break
            elif terms[i][1] >= prev.end and terms[i][2] <= npterm.begin:
                addTerm = Term()
                addTerm.begin = terms[i][1]
                addTerm.end = terms[i][2]
                addTerm.text = terms[i][0]
                addTerm.mark = 'O'
                addList.append(addTerm)
                # print('npTerm ', npterm.text)

                #print('add ', addTerm.text, addTerm.begin, addTerm.end)
            prevI = i
        prev = npterm


def write_np_training_data(newlist, npList, sentenceList, tFile):
    f = open('npData/' + tFile, 'w')

    sentNum = -1
    prevSentNum = -2
    for i, w in enumerate(newlist):
        #print(w[2])
        for j, s in enumerate(sentenceList):
            if npList[i].begin >= s[0] and npList[i].begin <= s[1]:
                sentNum = j
                break

        if (prevSentNum != sentNum and prevSentNum != -2):
            print('', file=f)

        prevSentNum = sentNum
        if npList[i].text.istitle() and npList[i].begin!=sentenceList[j][0]:
            isUpper = 'U'
        elif len(npList[i].text)>1 and npList[i].text.isupper():
            isUpper = 'A'
        else:
            isUpper = 'L'

        print(npList[i].text, w[0], w[1], isUpper, npList[i].mark, file=f)
    print('', file=f)
    f.close()

def match_ne_type(ner_term_i, term, prefix):
    if hasattr(ner_term_i, 'eType'):
        if ner_term_i.eType == 'location':
            term.eType = prefix + '-LOC'
        elif ner_term_i.eType == 'organization':
            term.eType = prefix + '-ORG'
        elif ner_term_i.eType == 'person':
            term.eType = prefix + '-PER'
        else:
            term.eType = 'O'
    else:
        term.eType = 'O'

def match_chunk(term, first):
    if first:
        term.chunkTag = 'B'
    else:
        term.chunkTag = 'I'


def make_corpus_from_training_data(tFile, json_folder, pos_beg_to_ner_term, pos_beg_to_chunk, pos_beg_to_sentiment, corpus_name):
    corpus_type = "NOTSENTIMENT"
    print(tFile)
    termList = []
    json_name = json_folder + tFile + '.json_result.json'
    with open(json_name) as data_file:
        data = json.load(data_file)

    term_to_pos_beg = {}
    for i, term in enumerate(data['results'][0]['result']['fields'][0]['terms']):
        #print(term['term'], term['np_chunk_id'], term['morph_pos'], term['morph_case'], term['pos'])
        term_to_pos_beg[term['pos'][0]] = i

    entity_to_pos_beg = {}
    for i, entity in enumerate(data['results'][0]['result']['fields'][0]['entities']):
        entity = data['results'][0]['result']['fields'][0]['entities'][i]
        print(entity['original_form'], entity['semantic_info'], entity['semantic_info_str'], entity['pos'])
        # if corpus_type == "SENTIMENT":
        #     entity_to_pos_beg[entity['pos'][0]] = i
        # else:
        for j, entity_part in enumerate(entity['parts']):
            entity_to_pos_beg[entity_part['pos'][0]] = i

    sentiment_to_pos_beg = {}

    punct_to_pos_beg = {}
    for i, punct in enumerate(data['results'][0]['result']['fields'][0]['special_characters']):
        punct_to_pos_beg[punct['pos'][0]] = i

    gazeteer_filters = ['location', 'name', 'person', 'organization', 'company', 'geo_key', 'org_key', 'fam', 'patr', 'ignored']
    #gazeteer_filters = ['location', 'person_full', 'organization']

    for sentence in data['results'][0]['result']['fields'][0]['sentences']:
        print(sentence['text'], sentence['pos'])
        print(len(sentence['text'].encode('utf-8')))

        last_ner_term_i = None
        last_chunk_i = None
        last_sentim_i = None
        last_entity_i = None
        i = 0
        if (termList and termList[-1].text != '\n'): # add \n in the end of each sentence
            endlTerm = Term()
            endlTerm.text = '\n'
            termList.append(endlTerm)
        while i < len(sentence['text'].encode('utf-8')):
            try:
                #print("Curr_i:", sentence['text'].encode('utf-8')[i:].decode("utf-8"))
                if sentence['text'].encode('utf-8')[i:].decode("utf-8").find('<img data-src=') == 0:
                    img_end = sentence['text'].encode('utf-8')[i:].decode("utf-8").find('/>')
                    if img_end >= 0:
                        #print("img data-src", sentence['text'].encode('utf-8')[i:i+img_end+2])
                        #print("i before", i)
                        i += img_end + 2
                        #print("i after", i)
                        continue
                if sentence['text'].encode('utf-8')[i:].decode("utf-8").find('</p><p>') == 0:
                    print("</p><p>:", sentence['text'].encode('utf-8')[i:i+len('</p><p>')])
                    i += len('</p><p>')
                    continue
                if sentence['text'].encode('utf-8')[i:].decode("utf-8").find('</p>') == 0:
                    print("</p>:", sentence['text'].encode('utf-8')[i:i+len('</p>')])
                    i += len('</p>')
                    continue
                if sentence['text'].encode('utf-8')[i:].decode("utf-8").find('<p>') == 0:
                    print("<p>:", sentence['text'].encode('utf-8')[i:i+len('<p>')])
                    i += len('<p>')
                    continue
                if sentence['text'].encode('utf-8')[i:].decode("utf-8").find('http') == 0:
                    link_end = sentence['text'].encode('utf-8')[i:].decode("utf-8").find(' ')
                    if link_end >= 0:
                        #print("https:", sentence['text'][i:link_end + 4])
                        #print("i before", i)
                        i += link_end
                        #print("i after", i)
                        continue
            except UnicodeDecodeError:
                print("exception UnicodeDecodeError")
                i += 1
                continue

            term_i = term_to_pos_beg.get(i+sentence['pos'][0])
            if term_i is not None:
                newTerm = Ner()
                pos_end = data['results'][0]['result']['fields'][0]['terms'][term_i]['pos'][1] - sentence['pos'][0]

                newTerm.pos_begin = i + sentence['pos'][0]
                newTerm.pos_end = pos_end + sentence['pos'][0]
                newTerm.text = sentence['text'].encode('utf-8')[i:pos_end].decode("utf-8")
                newTerm.norm_form = data['results'][0]['result']['fields'][0]['terms'][term_i]['term']
                #print("in text:", newTerm.pos_begin, newTerm.pos_end, newTerm.text)
                newTerm.morphPos = data['results'][0]['result']['fields'][0]['terms'][term_i]['morph_pos']

                if corpus_name != CorpusType.NER_ALL_EN:
                    newTerm.morphCase = data['results'][0]['result']['fields'][0]['terms'][term_i]['morph_case']

                #GAZETEER
                entity_i = entity_to_pos_beg.get(i+sentence['pos'][0])
                if entity_i is not None:
                    last_entity_i = entity_i
                    semantic = data['results'][0]['result']['fields'][0]['entities'][entity_i]['semantic_info_str']
                    if corpus_type != "SENTIMENT" and corpus_type != "SENTIMENT-2":
                        for attr in semantic.split(":"):
                            if attr in gazeteer_filters:
                                if newTerm.semantic is None:
                                    newTerm.semantic = attr
                                else:
                                    newTerm.semantic += ':' + attr
                    else:
                        newTerm.semantic = semantic

                    #newTerm.norm_form = data['results'][0]['result']['fields'][0]['entities'][entity_i]['norm_form']
                if newTerm.semantic is None:
                    newTerm.semantic = "none"


                #NER
                curr_ne_id = ''
                ner_term_i = pos_beg_to_ner_term.get(i + sentence['pos'][0])
                if last_ner_term_i is not None:
                    if ((newTerm.pos_begin >= last_ner_term_i.pos_begin) and (newTerm.pos_end <= last_ner_term_i.pos_end)):
                        match_ne_type(last_ner_term_i, newTerm, 'I') # newTerm.eType = *-LOC, *-ORG, *-PER, O
                        newTerm.sentimentTag = "NEUTRAL-ENTITY"
                        if hasattr(last_ner_term_i, 'id'):
                            curr_ne_id = last_ner_term_i.id
                if (newTerm.eType is None) and (ner_term_i is not None):
                    if ((last_ner_term_i is not None) and (last_ner_term_i.eType == ner_term_i.eType) and ((last_ner_term_i.pos_end + 1) == newTerm.pos_begin)):
                        match_ne_type(ner_term_i, newTerm, 'I')  # newTerm.eType = *-LOC, *-ORG, *-PER, O
                    else:
                        match_ne_type(ner_term_i, newTerm, 'B')  # newTerm.eType = *-LOC, *-ORG, *-PER, O
                    newTerm.sentimentTag = "NEUTRAL-ENTITY"
                    last_ner_term_i = ner_term_i
                    if hasattr(ner_term_i, 'id'):
                        curr_ne_id = ner_term_i.id
                elif (newTerm.eType is None):
                    newTerm.eType = 'O'

                #if pos_beg_to_chunk:
                #CHUNKS
                chunk_i = pos_beg_to_chunk.get(i + sentence['pos'][0])
                if last_chunk_i is not None:
                    if ((newTerm.pos_begin >= last_chunk_i.pos_begin) and (newTerm.pos_end <= last_chunk_i.pos_end)):
                        match_chunk(newTerm, False)
                if (newTerm.chunkTag is None) and (chunk_i is not None):
                    match_chunk(newTerm, True)
                    last_chunk_i = chunk_i
                elif (newTerm.chunkTag is None):
                    newTerm.chunkTag = 'O'

                if data['results'][0]['result']['fields'][0]['terms'][term_i]["type"] == "UPPERCASED":
                    newTerm.isUpper = 'A'
                elif data['results'][0]['result']['fields'][0]['terms'][term_i]["is_uppercased"]:
                    newTerm.isUpper = 'U'
                else:
                    newTerm.isUpper = 'L'

                # SENTIMENTS
                sentim_i = pos_beg_to_sentiment.get(i + sentence['pos'][0])
                if last_sentim_i is not None:
                    if ((newTerm.pos_begin >= last_sentim_i.pos_begin) and (newTerm.pos_end <= last_sentim_i.pos_end)):
                        sentim_i = last_sentim_i
                if sentim_i is not None:
                    if sentim_i.polarity == 'negative':
                        print("Text:", newTerm.text, sentim_i.text, "Sentiment NEG", sentim_i.polarity, sentim_i.target)
                        newTerm.sentimentTag = 'NEG'
                    elif sentim_i.polarity == 'positive':
                        print("Text:", newTerm.text, sentim_i.text, "Sentiment POS", sentim_i.polarity, sentim_i.target)
                        newTerm.sentimentTag = 'POS'
                    last_sentim_i = sentim_i
                else:
                    #SENTIMENTS-ENTITY
                    w_polarity = 0
                    sentim_id_to_w = {}
                    for j in range(0, len(sentence['text'].encode('utf-8'))):
                        sentim_j = pos_beg_to_sentiment.get(j + sentence['pos'][0])
                        if sentim_j is not None:
                            #    for k, v in pos_beg_to_sentiment.items():
                            if hasattr(sentim_j, 'target'):
                                for single in sentim_j.target.split(","):  # for all sentiments
                                    if curr_ne_id != '' and curr_ne_id == single:
                                        print("Single:", single, "Text:", newTerm.text , "Sentiment:", sentim_j.text, sentim_j.id, curr_ne_id, sentim_j.polarity)
                                        if sentim_j.polarity == 'negative':
                                            #w_polarity -= 1
                                            sentim_id_to_w[sentim_j.id] = -1
                                        elif sentim_j.polarity == 'positive':
                                            #w_polarity += 1
                                            sentim_id_to_w[sentim_j.id] = +1

                    if not sentim_id_to_w:
                        w_polarity = None
                    else:
                        for k in sentim_id_to_w:
                            w_polarity += sentim_id_to_w[k]
                    if w_polarity is not None:
                        if w_polarity > 0:
                            newTerm.sentimentTag = "POS-ENTITY"
                        elif w_polarity < 0:
                            newTerm.sentimentTag = "NEG-ENTITY"
                        else: # w_polarity == 0
                            newTerm.sentimentTag = "MIX-ENTITY"
                    elif newTerm.sentimentTag is None:
                            newTerm.sentimentTag = "NEUTRAL"

                termList.append(newTerm)
                print(newTerm.text, newTerm.pos_begin, newTerm.pos_end, newTerm.morphPos, newTerm.morphCase, newTerm.chunkTag, newTerm.isUpper, newTerm.semantic, newTerm.eType)
                i = pos_end
                continue

            punct_i = punct_to_pos_beg.get(i+sentence['pos'][0])
            if (punct_i is not None) and data['results'][0]['result']['fields'][0]['special_characters'][punct_i]['text'] == '"':
                i += 1
                continue
            elif (punct_i is not None):
                newTerm = Ner()
                punct_end = data['results'][0]['result']['fields'][0]['special_characters'][punct_i]['pos'][1] - sentence['pos'][0]
                newTerm.pos_begin = i + sentence['pos'][0]
                newTerm.pos_end = punct_end + sentence['pos'][0]
                newTerm.text = sentence['text'].encode('utf-8')[i:punct_end].decode("utf-8")
                newTerm.norm_form = data['results'][0]['result']['fields'][0]['special_characters'][punct_i]['text']
                #print("in text:", newTerm.pos_begin, newTerm.pos_end, newTerm.text)
                newTerm.morphPos = "Punc"
                newTerm.morphCase = "-"
                newTerm.semantic = "none"
                newTerm.sentimentTag = "NEUTRAL"
                newTerm.predictSentiment = "NEUTRAL"

                # if last_entity_i is not None:
                #     pos = data['results'][0]['result']['fields'][0]['entities'][last_entity_i]['pos']
                #     if punct_i >= pos[0] and punct_i <= pos[0]:
                #         semantic = data['results'][0]['result']['fields'][0]['entities'][last_entity_i]['semantic_info_str']
                #     for attr in semantic.split(":"):
                #         if attr in gazeteer_filters:
                #             if newTerm.semantic is None:
                #                 newTerm.semantic = attr
                #             else:
                #                 newTerm.semantic += ':' + attr
                # if newTerm.semantic is None:
                #     newTerm.semantic = "none"

                #NER
                curr_ne_id = ''
                ner_term_i = pos_beg_to_ner_term.get(i + sentence['pos'][0])
                if last_ner_term_i is not None:
                    # i_next = i
                    # while i_next < len(sentence['text'].encode('utf-8')):
                    #     term_i_next = term_to_pos_beg.get(i_next + sentence['pos'][0])
                    #     if term_i_next is not None:
                    #         ner_term_next = pos_beg_to_ner_term.get(i_next + sentence['pos'][0])
                    #         if ner_term_next is not None and ner_term_next.id == last_ner_term_i.id and ((last_ner_term_i.pos_end + 1) == ner_term_next.pos_begin):
                    #             match_ne_type(last_ner_term_i, newTerm, 'I')  # newTerm.eType = *-LOC, *-ORG, *-PER, O
                    #             newTerm.sentimentTag = "NEUTRAL-ENTITY"
                    #             last_ner_term_i = ner_term_next
                    #             if hasattr(last_ner_term_i, 'id'):
                    #                 curr_ne_id = last_ner_term_i.id
                    #         break
                    #     i_next += 1

                    if ((newTerm.pos_begin >= last_ner_term_i.pos_begin) and (newTerm.pos_end <= last_ner_term_i.pos_end)):
                        match_ne_type(last_ner_term_i, newTerm, 'I') # newTerm.eType = *-LOC, *-ORG, *-PER, O
                        newTerm.sentimentTag = "NEUTRAL-ENTITY"
                        if hasattr(last_ner_term_i, 'id'):
                            curr_ne_id = last_ner_term_i.id

                if (newTerm.eType is None) and (ner_term_i is not None):
                    if ((last_ner_term_i is not None) and (last_ner_term_i.id == ner_term_i.id)):
                        match_ne_type(ner_term_i, newTerm, 'I')  # newTerm.eType = *-LOC, *-ORG, *-PER, O
                    else:
                        match_ne_type(ner_term_i, newTerm, 'B')  # newTerm.eType = *-LOC, *-ORG, *-PER, O
                    newTerm.sentimentTag = "NEUTRAL-ENTITY"
                    last_ner_term_i = ner_term_i
                    if hasattr(ner_term_i, 'id'):
                        curr_ne_id = ner_term_i.id
                elif (newTerm.eType is None):
                    newTerm.eType = 'O'

                #CHUNKS
                chunk_i = pos_beg_to_chunk.get(i + sentence['pos'][0])
                if last_chunk_i is not None:
                    if ((newTerm.pos_begin >= last_chunk_i.pos_begin) and (newTerm.pos_end <= last_chunk_i.pos_end)):
                        match_chunk(newTerm, False)
                if (newTerm.chunkTag is None) and (chunk_i is not None):
                    match_chunk(newTerm, True)
                    last_chunk_i = chunk_i
                elif (newTerm.chunkTag is None):
                    newTerm.chunkTag = 'O'

                if newTerm.text.isupper() and len(newTerm.text) > 1:
                    newTerm.isUpper = 'A'
                elif newTerm.text[0].isupper():
                    newTerm.isUpper = 'U'
                else:
                    newTerm.isUpper = 'L'

                termList.append(newTerm)
                print(newTerm.text, newTerm.pos_begin, newTerm.pos_end, newTerm.morphPos, newTerm.morphCase, newTerm.chunkTag, newTerm.isUpper, newTerm.semantic, newTerm.eType)
                i = punct_end
                continue
            i += 1
    return termList

def gateToCrf(corpus_type):
    source_folder = ''
    result_folder = ''
    corpus_name = ''
    if corpus_type == CorpusType.NER_1:
        source_folder = 'gateSource'
        result_folder = 'gateResultForNer1'
        corpus_name = 'ner_corpus1.txt'
    elif corpus_type == CorpusType.NER_2:
        source_folder = 'gateSourceOur2'
        result_folder = 'gateResultForNer2'
        corpus_name = 'ner_corpus2.txt'
    elif corpus_type == CorpusType.NER_2_testset:
        source_folder = 'gateSourceSmallTestSet'
        result_folder = 'gateResultSmallTestSet'
        corpus_name = 'ner_testset_rus.txt'
    elif corpus_type == CorpusType.NER_3:
        source_folder = 'gateSource3'
        result_folder = 'gateResultForNer3'
        corpus_name = 'ner_corpus3.txt'
    elif corpus_type == CorpusType.NER_4:
        source_folder = 'gateSource4'
        result_folder = 'gateResultForNer4'
        corpus_name = 'ner_corpus4.txt'
    elif corpus_type == CorpusType.NER_ALL_EN:
       source_folder = 'gateSourceAllEn'
       result_folder = 'gateResultForNerAllEn'
       corpus_name = 'ner_corpus_all.txt'
    elif corpus_type == CorpusType.NER_ALL:
       source_folder = 'gateSourceAll'
       result_folder = 'gateResultForNerAll'
       corpus_name = 'ner_corpus_all.txt'

    termLists = list()
    listFiles = os.listdir('./' + source_folder) #gateSourceOur2 gateSourc
    listFiles.sort()
    open("./corpus/" + corpus_name, 'w').close()  # clear file
    for fnum, tFile in enumerate(listFiles):
        termList = list()
        print(tFile)
        str1 = parse_document(termList, './' + source_folder + '/' + tFile)

        #creating batches
        pos_beg_to_ner_term = {}
        termList.sort(key=lambda elem: elem.begin)
        for t in termList:
            if hasattr(t, 'eType'):
                pos_beg_to_ner_term[t.pos_begin] = t
                print("ner", t.text, t.pos_begin, t.pos_end)
                print("eType", t.eType)

        json_data = []
        json_data.append({"document": {"text": str1}, "language": "ru"})
        entity_data = []
        for entity in termList:
            if hasattr(entity, 'eType'):
                item = {"original_form": entity.text}
                item['pos'] = [entity.pos_begin, entity.pos_end]
                item['semantic_info_str'] = entity.eType
                #item['polarity'] = entity.polarity
                parts_data = []

                #for part in entity.parts:
                #    item_part = {"term": part.term}
                #    item_part['pos'] = [part.begin, part.end]
                #    parts_data.append(item_part)
                item['parts'] = parts_data

                # newEnt.polarity = newTerm.polarity
                # newEnt.semantic_info_str = newTerm.eType
                # for attribute in attributes_selected:
                #    if attribute.feature == feature:
                #       item[attribute.attribute.name] = attribute.value
                entity_data.append(item)
        json_data.append({"entities": entity_data})


        #json_data = json.dumps([{"document":{"text":str1}, "language":"ru"}], ensure_ascii=False)
        filename = "./ourBatches3/" + tFile + ".json"
        print("\"ourBatches3/" + tFile + ".json\",")
        file = open(filename, 'w')
        file.write(json.dumps(json_data, ensure_ascii=False))
        file.close()
        continue

        sentenceList = PunktSentenceTokenizer().span_tokenize(str1)

        termList.sort(key=lambda elem: elem.begin)

        #pos_beg_to_ner_term = {}
        for t in termList:
            if hasattr(t, 'eType'):
                pos_beg_to_ner_term[t.pos_begin] = t
                #print("ner", t.text, t.pos_begin, t.pos_end)
                #print("eType", t.eType)

        pos_beg_to_chunk = {}
        pos_beg_to_sentiment = {}
        for t in termList:
            if t.type == 'nounPhrase':
                pos_beg_to_chunk[t.pos_begin] = t
                #print("chunk", t.text)
            elif t.type == 'sentiment':
                if hasattr(t, 'polarity') and hasattr(t, 'target'):
                    pos_beg_to_sentiment[t.pos_begin] = t
                    #print(t.target)

        if not pos_beg_to_chunk:
            json_name = result_folder + '/' + tFile + '.json_result.json'
            with open(json_name) as data_file:
                data = json.load(data_file)
            last_chunk = None
            for i, term in enumerate(data['results'][0]['result']['fields'][0]['terms']):
                if term['np_chunk_id'] != 0:
                    chunk = Term()
                    if last_chunk and term['np_chunk_id'] == last_chunk.id:
                        chunk = last_chunk
                        chunk.pos_end = term['pos'][1]
                    else:
                        chunk.pos_begin = term['pos'][0]
                        chunk.pos_end = term['pos'][1]
                        chunk.id = term['np_chunk_id']
                    last_chunk = chunk
                    pos_beg_to_chunk[chunk.pos_begin] = chunk

        npList = list()
        find_npChunks(termList, npList)

        terms = to_terms(str1)

        allPunctlist = list()
        find_all_punct(terms, allPunctlist)

        addList = list()
        make_addition_list(addList, npList, terms)

        npList.extend(addList)
        npList.sort(key=lambda elem: elem.begin)


        for p in npList:
            print(p.text, p.mark, p.begin, p.end)
        corpList = make_corpus_from_training_data(tFile, result_folder + '/', pos_beg_to_ner_term, pos_beg_to_chunk, pos_beg_to_sentiment, corpus_type)

        file = open("./corpus/" + corpus_name, 'a')
        line = ''
        for term in corpList:
            if term.text == '\n' or term.text == '\r':
                line = '\n'
            elif corpus_type == CorpusType.NER_ALL_EN:
                line = term.text + ' ' + term.morphPos + ' ' + term.chunkTag + ' ' + term.isUpper + ' ' + term.semantic + ' ' + term.eType + '\n'  # term.pos_begin, term.pos_end, #
            else:
                line = term.text + ' ' + term.morphPos + ' ' + term.morphCase + ' ' + term.chunkTag + ' ' + term.isUpper + ' ' + term.semantic + ' ' + term.eType + '\n'  # term.pos_begin, term.pos_end, #

                #if term.predictSentiment:
                #line = term.text + '\t' + term.norm_form + ' ' + term.morphPos + ' ' + term.morphCase + ' ' + term.chunkTag + ' ' + term.isUpper + ' ' + term.semantic + ' ' + term.sentimentTag +  '\n' #term.pos_begin, term.pos_end, #SENTIMENT
            # print(line.strip('\n'))
            file.write(line)
        file.close()


def opencorporaToCrf():
    #list_plaintexts = [f for f in os.listdir('./opencorpora') if os.path.isfile('./opencorpora/' + f)]
    #list_spans = [f for f in os.listdir('./opencorpora/spans') if os.path.isfile('./opencorpora/spans/' + f)]
    list_tokens = [f for f in os.listdir('./opencorpora/tokens') if os.path.isfile('./opencorpora/tokens/' + f)]
    open("./corpus/opencorpora_corpus.txt", 'w').close()  # clear file
    all_ners_by_type = {}
    for tFile in list_tokens:
        filename = tFile.strip('.tokens')
        f = open('./opencorpora/tokens/' + filename + '.tokens', 'r')
        terms = {}  # key: term_id, value: [pos_beg, pos_end, term]
        term_beg_to_id = {}  # key: pos_beg, value: term_id
        for line in f.readlines():
            splitted = re.split(" ", line)
            if len(splitted) >= 4:
                terms[int(splitted[0])] = [int(splitted[1]), int(splitted[2]), splitted[3]]
                term_beg_to_id[int(splitted[1])] = int(splitted[0])
        f.close()
        print(terms)

        spans = {}  # key: span_id  value: [pos_beg, length, term_id, terms_count]
        f = open('./opencorpora/spans/' + filename + '.spans', 'r')
        for line in f.readlines():
            splitted = re.split(" ", line)
            if len(splitted) >= 4:
                spans[int(splitted[0])] = splitted[2:5]
        f.close()
        print("spans", spans)

        objects = {}  # key: obj_id  value: [type, span_id]
        f = open('./opencorpora/objects/' + filename + '.objects', 'r')
        for line in f.readlines():
            line = re.split("\#", line)[0]
            splitted = re.split(" ", line)
            if len(splitted) >= 3:
                objects[int(splitted[0])] = splitted[1:]
        f.close()
        print("objects", objects)

        tok_id_to_ner_term = {}
        for k in objects:
            nerTerm = NamedEntity()
            type = objects[k][0] #
            type = type.lower()
            if type == 'loc' or type == 'locorg':
                nerTerm.eType = 'location'
            elif type == 'org':
                nerTerm.eType = 'organization'
            elif type == 'person':
                nerTerm.eType = 'person'
            else:
                nerTerm.eType = type
            nerTerm.id = k
            id_first = id_last = int(objects[k][1])

            min_beg = int(spans[id_first][0])
            max_end = int(spans[id_last][0]) + int(spans[id_last][1])
            term_ids = []
            for span_id in objects[k][1:]:
                if span_id:
                    min_beg = min(min_beg, int(spans[int(span_id)][0]))
                    max_end = max(max_end, int(spans[int(span_id)][0]) + int(spans[int(span_id)][1]))
                    term_ids.append(int(spans[int(span_id)][2]))
                    #print("NER: ", spans[int(span_id)][0], spans[int(span_id)][1], type)
            #nerTerm.text = None #TODO
            nerTerm.begin = min_beg
            nerTerm.end = max_end
            nerTerm.type = 'namedEntity'

            for id in term_ids:
                tok_id_to_ner_term[id] = nerTerm


        f = open('./opencorpora/' + filename + '.txt', 'r')
        i = 0
        i_bytes = 0
        pos_beg_to_ner_term = {}
        for line in f.readlines():
            for j, v in enumerate(line):
                term_id = term_beg_to_id.get(i+j)
                if term_id is not None:
                    ner_term_i = tok_id_to_ner_term.get(term_id)
                    if ner_term_i is not None:
                        ner_term_i.pos_begin = i_bytes + len(line[:(ner_term_i.begin-i)].encode('utf-8'))
                        ner_term_i.pos_end = i_bytes + len(line[:(ner_term_i.end-i)].encode('utf-8'))
                        ner_term_i.text = line[(ner_term_i.begin-i):(ner_term_i.end-i)]
                        pos_beg_to_ner_term[ner_term_i.pos_begin] = ner_term_i
                        print("ner:", ner_term_i.pos_begin, ner_term_i.pos_end, ner_term_i.text, ner_term_i.eType)
                        if all_ners_by_type.get(ner_term_i.eType) is None:
                            all_ners_by_type[ner_term_i.eType] = 0
                        all_ners_by_type[ner_term_i.eType] += 1

            if line != '\n':
                i_bytes += len(line.encode('utf-8'))
            i += len(line)
        f.close()

        #print('\"'+'devsetBatches/'+tFile+'.json\",')
        # f = open('./txt_testset/' + tFile, 'r')
        # content = f.read()
        # content = content.replace('\n\n', '\n')
        # json_data = json.dumps([{"document":{"text":content}, "language":"ru"}], ensure_ascii=False)
        # filename = "./temp/" + tFile + ".json"
        # new_file = open(filename, 'w')
        # new_file.write(json_data)
        # new_file.close()
        # print(json_data)
        #f.close()


        json_name = 'opencorporaResult/' + filename + '.txt.json_result.json'
        with open(json_name) as data_file:
            data = json.load(data_file)
        pos_beg_to_chunk = {}
        last_chunk = None
        for i, term in enumerate(data['results'][0]['result']['fields'][0]['terms']):
            if term['np_chunk_id'] != 0:
                chunk = Term()
                if last_chunk and term['np_chunk_id'] == last_chunk.id:
                    chunk = last_chunk
                    chunk.pos_end = term['pos'][1]
                else:
                    chunk.pos_begin = term['pos'][0]
                    chunk.pos_end = term['pos'][1]
                    chunk.id = term['np_chunk_id']
                last_chunk = chunk
                pos_beg_to_chunk[chunk.pos_begin] = chunk

        pos_beg_to_sentiment = {}
        corpList = make_corpus_from_training_data(filename + '.txt', 'opencorporaResult/', pos_beg_to_ner_term, pos_beg_to_chunk, pos_beg_to_sentiment, 'OPENCORP')

        file = open("./corpus/opencorpora_corpus.txt", 'a')
        line = ''
        for term in corpList:
            if term.text == '\n':
                line = '\n'
            else:
                line = term.text + ' ' + term.morphPos + ' ' + term.morphCase + ' ' + term.chunkTag + ' ' + term.isUpper + ' ' + term.semantic + ' ' + term.eType + '\n'  # term.pos_begin, term.pos_end, #
                #line = term.text + ' ' + term.morphPos + ' ' + term.morphCase + ' ' + term.chunkTag + ' ' + term.isUpper + ' ' + term.semantic + ' ' + term.sentimentTag +  '\n' #term.pos_begin, term.pos_end, #SENTIMENT
            # print(line.strip('\n'))
            file.write(line)
        file.close()





##-----------------------------------------------------
gateToCrf(CorpusType.SENTI_ES)
#gateToCrf(CorpusType.NER_1)
#gateToCrf(CorpusType.NER_2)
#gateToCrf(CorpusType.NER_3)
# gateToCrf(CorpusType.NER_ALL)
# gateToCrf(CorpusType.NER_2_testset)
# opencorporaToCrf()
#wikipediaToCrf()