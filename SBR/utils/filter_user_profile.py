import random

from sentence_splitter import SentenceSplitter


def filter_user_profile_random_sentences(dataset_config, user_info):
    sent_splitter = SentenceSplitter(language='en')

    user_info = user_info.map(sentencize_function_torchtext, fn_kwargs={
        'case_sensitive': dataset_config['case_sensitive'],
        'normalize_negation': dataset_config['normalize_negation'],
        'sentencizer': sent_splitter
    },
                              batched=True)

    user_info = user_info.to_pandas()
    user_info['text'] = user_info['sentences_text'].apply(lambda x: " ".join(random.sample(list(x), len(x))))
    user_info = user_info.drop(columns=['sentences_text'])
    return user_info


def sentencize_function_torchtext(samples, sentencizer=None, doc_desc_field="text",
                                        case_sensitive=True, normalize_negation=True):
    sent_ret = []

    for text in samples[doc_desc_field]:
        sents = []
        for s in sentencizer.split(text=text):
            if not case_sensitive:
                s = s.lower()
            if normalize_negation:
                s = s.replace("n't", " not")
            sents.append(s)
        sent_ret.append(sents)

    return {f"sentences_{doc_desc_field}": sent_ret}
