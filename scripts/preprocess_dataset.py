from scripts.all_imports import *


class Preprocess_dataset(object):
    def __init__(self, args) -> None:
        self.args = args
    
    def preprocess_text(self, text: str, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' '):
        """
        Preprocess text as per Keras Tokenizer preprocess code. 
        Tokenize by just sentence.split()
        Whole process is similar to Keras Tokenizer
        """
        text = text.lower() # lower case
        maketrans = str.maketrans
        translate_dict = {c: split for c in filters}
        translate_map = maketrans(translate_dict) 
        text = text.translate(translate_map) # remove all punctuations and replace them with whitespace (because puntuations mark as a whitespace between words)
        return text
    
    def conjunction_analysis(self, dataset):
        """
        Count the sentences labeled with a particular rule like A-but-B in the dataset during dataset creation
        Perform a conjunction analysis for that rule in the sentences
        Check if both counts are equal
        If not equal, remove the datapoints which has the rule label but fails on its conjunction analysis
        """
        indices_to_remove = []
        no_rule_sentences = dataset.loc[dataset["rule_label"]==0]["sentence"]
        but_sentences = dataset.loc[dataset["rule_label"]==1]["sentence"]
        yet_sentences = dataset.loc[dataset["rule_label"]==2]["sentence"]
        though_sentences = dataset.loc[dataset["rule_label"]==3]["sentence"]
        while_sentences = dataset.loc[dataset["rule_label"]==4]["sentence"]

        for sentence in no_rule_sentences: # Check for any rule structure in no rule sentences and remove any sentence containing a rule structure
            tokenized_sentence = sentence.split()
            if ('but' in tokenized_sentence and tokenized_sentence.index('but') != 0 and tokenized_sentence.index('but') != -1 and tokenized_sentence.count('but') == 1):
                index_to_remove = list(no_rule_sentences.index[no_rule_sentences==sentence])[0]
                indices_to_remove.append(index_to_remove)
            elif ('yet' in tokenized_sentence and tokenized_sentence.index('yet') != 0 and tokenized_sentence.index('yet') != -1 and tokenized_sentence.count('yet') == 1):
                index_to_remove = list(no_rule_sentences.index[no_rule_sentences==sentence])[0]
                indices_to_remove.append(index_to_remove)
            elif ('though' in tokenized_sentence and tokenized_sentence.index('though') != 0 and tokenized_sentence.index('though') != -1 and tokenized_sentence.count('though') == 1):
                index_to_remove = list(no_rule_sentences.index[no_rule_sentences==sentence])[0]
                indices_to_remove.append(index_to_remove)
            elif ('while' in tokenized_sentence and tokenized_sentence.index('while') != 0 and tokenized_sentence.index('while') != -1 and tokenized_sentence.count('while') == 1):
                index_to_remove = list(no_rule_sentences.index[no_rule_sentences==sentence])[0]
                indices_to_remove.append(index_to_remove)

        for sentence in but_sentences:
            tokenized_sentence = sentence.split()
            if ('but' in tokenized_sentence and tokenized_sentence.index('but') != 0 and tokenized_sentence.index('but') != -1 and tokenized_sentence.count('but') == 1):
                continue
            else:
                index_to_remove = list(but_sentences.index[but_sentences==sentence])[0]
                indices_to_remove.append(index_to_remove)

        for sentence in yet_sentences:
            tokenized_sentence = sentence.split()
            if ('yet' in tokenized_sentence and tokenized_sentence.index('yet') != 0 and tokenized_sentence.index('yet') != -1 and tokenized_sentence.count('yet') == 1):
                continue
            else:
                index_to_remove = list(yet_sentences.index[yet_sentences==sentence])[0]
                indices_to_remove.append(index_to_remove)

        for sentence in though_sentences:
            tokenized_sentence = sentence.split()
            if ('though' in tokenized_sentence and tokenized_sentence.index('though') != 0 and tokenized_sentence.index('though') != -1 and tokenized_sentence.count('though') == 1):
                continue
            else:
                index_to_remove = list(though_sentences.index[though_sentences==sentence])[0]
                indices_to_remove.append(index_to_remove)

        for sentence in while_sentences:
            tokenized_sentence = sentence.split()
            if ('while' in tokenized_sentence and tokenized_sentence.index('while') != 0 and tokenized_sentence.index('while') != -1 and tokenized_sentence.count('while') == 1):
                continue
            else:
                index_to_remove = list(while_sentences.index[while_sentences==sentence])[0]
                indices_to_remove.append(index_to_remove)

        dataset = dataset.drop(indices_to_remove)
        dataset = dataset.reset_index(drop=True)

        return dataset
    
    def create_rule_masks(self, dataset):
        """
        create rule masks for each sentence in the dataset
        """
        rule_label_masks = []
        for index, sentence in enumerate(list(dataset['sentence'])):
            tokenized_sentence = sentence.split()
            rule_label = dataset['rule_label'][index]
            contrast = dataset['contrast'][index]
            try:
                if rule_label == 1 and contrast == 1:
                    a_part_tokenized_sentence = tokenized_sentence[:tokenized_sentence.index("but")]
                    b_part_tokenized_sentence = tokenized_sentence[tokenized_sentence.index("but")+1:]
                    rule_label_mask = [0]*len(a_part_tokenized_sentence) + [0]*len(["but"]) + [1]*len(b_part_tokenized_sentence)
                    rule_label_masks.append(rule_label_mask)

                elif rule_label == 2 and contrast == 1:
                    a_part_tokenized_sentence = tokenized_sentence[:tokenized_sentence.index("yet")]
                    b_part_tokenized_sentence = tokenized_sentence[tokenized_sentence.index("yet")+1:]
                    rule_label_mask = [0]*len(a_part_tokenized_sentence) + [0]*len(["yet"]) + [1]*len(b_part_tokenized_sentence)
                    rule_label_masks.append(rule_label_mask)

                elif rule_label == 3 and contrast == 1:
                    a_part_tokenized_sentence = tokenized_sentence[:tokenized_sentence.index("though")]
                    b_part_tokenized_sentence = tokenized_sentence[tokenized_sentence.index("though")+1:]
                    rule_label_mask = [1]*len(a_part_tokenized_sentence) + [0]*len(["though"]) + [0]*len(b_part_tokenized_sentence)
                    rule_label_masks.append(rule_label_mask)

                elif rule_label == 4 and contrast == 1:
                    a_part_tokenized_sentence = tokenized_sentence[:tokenized_sentence.index("while")]
                    b_part_tokenized_sentence = tokenized_sentence[tokenized_sentence.index("while")+1:]
                    rule_label_mask = [1]*len(a_part_tokenized_sentence) + [0]*len(["while"]) + [0]*len(b_part_tokenized_sentence)
                    rule_label_masks.append(rule_label_mask)
                
                else:
                    mask_length = len(tokenized_sentence)
                    rule_label_mask = [1]*mask_length
                    rule_label_masks.append(rule_label_mask)
            except:
                mask_length = len(tokenized_sentence)
                rule_label_mask = [1]*mask_length
                rule_label_masks.append(rule_label_mask)
        
        dataset["rule_mask"] = rule_label_masks

        return dataset

    def preprocess_covid_19_tweets(self, dataset) -> dict:

        # convert into dataframe for fast processing
        dataset = pd.DataFrame(dataset)

        # Select columns
        dataset = dataset[['preprocessed_tweet','sentiment_label', 'rule_label', 'contrast']]

        # Select no rule, but, yet, though and while rule sentences
        dataset = dataset.loc[(dataset["rule_label"]=="not_applicable")|(dataset["rule_label"]=="A-but-B")|(dataset["rule_label"]=="A-yet-B")|(dataset["rule_label"]=="A-though-B")|(dataset["rule_label"]=="A-while-B")]
        
        # Converting str values to int values for rule label and contrast columns
        dataset['rule_label'] = dataset['rule_label'].map({'not_applicable': 0, 'A-but-B': 1, 'A-yet-B': 2, 'A-though-B': 3, 'A-while-B': 4})
        dataset['contrast'] = dataset['contrast'].map({'not_applicable': 0, 'no_contrast': 0, 'contrast': 1})

        # Renaming the preprocessed_tweet column to sentence
        dataset = dataset.rename(columns={'preprocessed_tweet': 'sentence'})

        # Converting -1 value for negative sentiment to 0
        dataset['sentiment_label'].replace({-1: 0}, inplace=True)

        # Balance the dataset between one rule and no rule
        dataset_one_rule = dataset.loc[dataset["rule_label"]!=0]
        dataset_no_rule_pos = dataset.loc[(dataset["rule_label"]==0)&(dataset["sentiment_label"]==1)]
        dataset_no_rule_neg = dataset.loc[(dataset["rule_label"]==0)&(dataset["sentiment_label"]==0)]
        dataset_no_rule_sample_pos = dataset_no_rule_pos.sample(n=22318, random_state=self.args.seed_value)
        dataset_no_rule_sample_neg = dataset_no_rule_neg.sample(n=34318, random_state=self.args.seed_value)
        dataset = pd.concat([dataset_one_rule, dataset_no_rule_sample_pos, dataset_no_rule_sample_neg])
        dataset = dataset.sample(frac=1, random_state=self.args.seed_value).reset_index(drop=True)

        # Preprocess sentences (VECTORIZE)
        preprocessed_sentences = [self.preprocess_text(sentence) for sentence in list(dataset['sentence'])]
        dataset["sentence"] = preprocessed_sentences

        # Perform conjunction analysis on sentences (VECTORIZE)
        dataset = self.conjunction_analysis(dataset)
        
        # Create rule masks (VECTORIZE)
        dataset = self.create_rule_masks(dataset)

        # Convet back into dict
        dataset = dataset.to_dict('list')
        
        return dataset
    
    def preprocess(self) -> dict:

        if self.args.dataset_name == "covid-19_tweets":

            # load the pickle file as dict
            raw_dataset = pickle.load(open(self.args.dataset_dir+"raw_dataset.pickle", "rb"))

            # preprocess
            preprocessed_dataset = self.preprocess_covid_19_tweets(raw_dataset)

            # return the dict
            return preprocessed_dataset
