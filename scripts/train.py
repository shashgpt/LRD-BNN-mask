from scripts.all_imports import *
from scripts.train_scripts.dataset_division import Dataset_division
from scripts.train_scripts.word_vectors import Word_vectors
from scripts.train_scripts.models.bnsm import *


DEVICE = torch.device(0)
class Train(object):
    def __init__(self, args) -> None:
        self.args = args

    def vectorize(self, sentences, word_index):
        """
        1) Tokenize each preprocessed sentence in dataset as sentence.split()
        2) Encode each tokenized sentence as per vocabulary
        3) Right pad each encoded tokenized sentence with 0 upto max_tokenized_sentence_len the dataset 
        """
        # vocab = [key for key in word_index.keys()]
        # max_sent_len = 0
        # encoded_tokenized_sentences = []
        # for sentence in sentences:
        #     tokenized_sentence = sentence.split()
        #     encoded_tokenized_sentence = [vocab.index(word) for word in tokenized_sentence]
        #     if len(encoded_tokenized_sentence) > max_sent_len:
        #         max_sent_len = len(encoded_tokenized_sentence)
        #     encoded_tokenized_sentences.append(encoded_tokenized_sentence)
        # for index, encoded_tokenized_sentence in enumerate(encoded_tokenized_sentences):
        #     padding_length = max_sent_len - len(encoded_tokenized_sentence[index])
        #     encoded_tokenized_sentences[index] = encoded_tokenized_sentences[index] + ([0] * padding_length)
        # return np.array(encoded_tokenized_sentences)

        # Pytorch code
        vocab = torchtext.vocab.vocab(word_index)
        vocab.set_default_index(0)
        max_sent_len = 0
        encoded_tokenized_sentences = []
        for sentence in sentences:
            tokenized_text = sentence.split()
            encoded_tokenized_sentence = vocab(tokenized_text)
            if len(encoded_tokenized_sentence) > max_sent_len:
                max_sent_len = len(encoded_tokenized_sentence)
            encoded_tokenized_sentences.append(encoded_tokenized_sentence)
        for index, _ in enumerate(encoded_tokenized_sentences):
            padding_length = max_sent_len - len(encoded_tokenized_sentences[index])
            encoded_tokenized_sentences[index] = encoded_tokenized_sentences[index] + ([0] * padding_length)
        encoded_tokenized_sentences = np.array(encoded_tokenized_sentences)
        return encoded_tokenized_sentences


    def train_model(self, preprocessed_dataset: dict):

        # create word embeddings
        if os.path.exists(self.args.dataset_dir+self.args.word_embeddings+".npy") and os.path.exists(self.args.dataset_dir+self.args.word_embeddings+"_word_index.pickle"):
            word_vectors = np.load(open(self.args.dataset_dir+self.args.word_embeddings+".npy", "rb"))
            word_index = pickle.load(open(self.args.dataset_dir+self.args.word_embeddings+"_word_index.pickle", "rb"))
        else:
            word_vectors, word_index = Word_vectors(self.args).create_word_vectors(preprocessed_dataset)
            with open(self.args.dataset_dir+self.args.word_embeddings+".npy", "wb") as handle:
                np.save(handle, word_vectors)
            with open(self.args.dataset_dir+self.args.word_embeddings+"_word_index.pickle", "wb") as handle:
                pickle.dump(word_index, handle)

        # create train, val and test_dataset
        train_dataset, val_dataset, test_dataset = Dataset_division(self.args).train_val_test_split(preprocessed_dataset, divide_into_rule_sections=True)

        # create model
        if self.args.model_name == "brnn":
            model = BNN(self.args, word_vectors).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)   
            # model.summary(line_length = 150)

        # train model
        train_sentences = self.vectorize(train_dataset["sentence"], word_index)
        train_sentiment_labels = np.array(train_dataset["sentiment_label"])
        val_sentences = self.vectorize(val_dataset["val_dataset"]["sentence"], word_index)
        val_sentiment_labels = np.array(val_dataset["val_dataset"]["sentiment_label"])
        train_dataset = (train_sentences, train_sentiment_labels)
        val_dataset = (val_sentences, val_sentiment_labels)
        model.fit(train_dataset, val_dataset, model, optimizer)
    
        # save the model
    
    # def evaluate_model(self, preprocessed_dataset: dict):


