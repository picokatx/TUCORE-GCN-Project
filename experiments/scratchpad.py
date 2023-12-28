def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    print("#examples", len(examples))

    features = [[]]
    for (ex_index, example) in enumerate(examples):
        tokens_a, tokens_a_speaker_ids, tokens_a_mention_ids = tokenize(example.text_a, tokenizer, 0)
        tokens_b, tokens_b_speaker_ids = tokenize(example.text_b, tokenizer)
        tokens_c, tokens_c_speaker_ids = tokenize(example.text_c, tokenizer)

        _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 4, tokens_a_speaker_ids, tokens_b_speaker_ids, tokens_c_speaker_ids, tokens_a_mention_ids)
        tokens_b_mention_ids = [max(tokens_a_mention_ids) + 1 for _ in range(len(tokens_b))]
        tokens_c_mention_ids = [max(tokens_a_mention_ids) + 2 for _ in range(len(tokens_c))]

        tokens_b = tokens_b + ["[SEP]"] + tokens_c # roberta
        tokens_b_speaker_ids = tokens_b_speaker_ids + [0] + tokens_c_speaker_ids # roberta
        tokens_b_mention_ids = tokens_b_mention_ids + [0] + tokens_c_mention_ids # roberta

        tokens = []
        segment_ids = []
        speaker_ids = []
        mention_ids = []
        tokens.append("[CLS]") # roberta
        segment_ids.append(0)
        speaker_ids.append(0)
        mention_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        speaker_ids = speaker_ids + tokens_a_speaker_ids
        mention_ids = mention_ids + tokens_a_mention_ids
        tokens.append("[SEP]") # roberta
        segment_ids.append(0)
        speaker_ids.append(0)
        mention_ids.append(0)

        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        speaker_ids = speaker_ids + tokens_b_speaker_ids
        mention_ids = mention_ids + tokens_b_mention_ids
        tokens.append("[SEP]") # roberta
        segment_ids.append(1)
        speaker_ids.append(0)
        mention_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            speaker_ids.append(0)
            mention_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(speaker_ids) == max_seq_length
        assert len(mention_ids) == max_seq_length

        label_id = example.label 
        
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [x for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info(
                    "speaker_ids: %s" % " ".join([str(x) for x in speaker_ids]))
            logger.info(
                    "mention_ids: %s" % " ".join([str(x) for x in mention_ids]))


        features[-1].append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id,
                        speaker_ids=speaker_ids,
                        mention_ids=mention_ids))
        if len(features[-1]) == 1:
            features.append([])

    if len(features[-1]) == 0:
        features = features[:-1]
    print('#features', len(features))
    return features